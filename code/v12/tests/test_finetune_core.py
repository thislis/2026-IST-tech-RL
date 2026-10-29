"""Synthetic fine-tuning orchestration checks; no Unity or real training."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from blackout_v12 import finetune
from blackout_v12.critic import Critic
from blackout_v12.io import atomic_torch, file_hash, read_json, state_hash
from blackout_v12.model import TaskScorer


def report(opponents, pairs, split, winning=True, **changes):
    rows = []
    for opponent in opponents:
        for replicate in range(pairs):
            for side in (0, 1):
                rows.append(dict(valid=True, natural_terminal=True, score_valid=True,
                    split_audit={'valid': True}, opponent=opponent, replicate=replicate, side=side,
                    raw_points=1. if winning or opponent['id'] != 'scripted-default' else .5,
                    score_points=1., requested_seed=replicate + (100 if split == 'confirmation' else 1),
                    initial_map_hash=f'{split}-{replicate}', split=split))
    rows[0].update(changes)
    return dict(rows=rows, completeness=True, interrupted=False, rows_path='synthetic-rows.json')


class FineTuneCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def configuration(self, root):
        return dict(max_steps=1, eval_interval=1, dev_pairs=1, confirmation_pairs=1,
                    data_directory=str(root/'data'), actor_threads=1, learner_threads=1,
                    opponents=[dict(id=name, family='scripted') for name in finetune.SCRIPTED_IDS])

    def source(self, root):
        source = root/'source'/'latest.pt'
        atomic_torch(source, dict(arm='F', scorer_state=TaskScorer().state_dict(),
            critic_state=Critic().state_dict(), actor_optimizer={'source_only': True},
            critic_optimizer={'source_only': True}, counters={'env_steps': 99999}))
        return source

    def fake_runtime(self, calls, live):
        class Pool:
            def __init__(self, workers, visible=True):
                calls.append(('open', workers, visible))
            def __enter__(self):
                live[0] += 1
                return self
            def __exit__(self, *args):
                live[0] -= 1
                calls.append(('close',))
            def run(self, tasks):
                calls.append(('wave', copy.deepcopy(tasks)))
                rows = []
                for task in tasks:
                    event = dict(features=torch.zeros(5, 2, 64), prior=torch.zeros(5, 2),
                        legal=torch.ones(5, 2, dtype=torch.bool), capacities=torch.tensor([5, 5]),
                        assignment=torch.zeros(5, dtype=torch.long), temperature=torch.tensor(1.),
                        old_log_prob=torch.tensor(0.), critic_features=torch.zeros(64),
                        step=0, duration=1, reward=0., on_policy=True)
                    episode = dict(valid=True, natural_terminal=True, on_policy=True,
                        policy_sha256=task['policy_sha256'], episode_id=task['id'],
                        opponent=task['opponent'], side=task['side'], split='train',
                        requested_seed=task['requested_seed'], action_seed=task['action_seed'],
                        split_audit={'valid': True}, env_steps=1, event_count=1, events=[event])
                    path = Path(task['directory'])/'episode.pt'
                    atomic_torch(path, episode)
                    rows.append(dict(episode_id=task['id'], trajectory=str(path)))
                return list(reversed(rows))
        return Pool

    def test_strict_target_keeps_invalids_draws_and_paired_maps_in_denominator(self):
        opponents = [dict(id=name) for name in finetune.SCRIPTED_IDS]
        perfect = report(opponents, 1, 'dev')
        result = finetune.strict_report(perfect, opponents, 1)
        self.assertTrue(result['all_wins'])
        self.assertEqual(result['wins'], 18)
        for changes in ({'raw_points': .5}, {'valid': False}, {'score_valid': False},
                        {'split_audit': {'valid': False}}, {'initial_map_hash': 'wrong-map'},
                        {'requested_seed': -1}):
            with self.subTest(changes=changes):
                result = finetune.strict_report(report(opponents, 1, 'dev', **changes), opponents, 1)
                self.assertFalse(result['all_wins'])
                self.assertEqual(result['planned_games'], 18)
                self.assertLess(result['opponent_macro_win_rate'], 1.)
        # A valid terminal score with a different numerical winner stays
        # separate from the provided-runner objective.
        result = finetune.strict_report(report(opponents, 1, 'dev', score_points=0.), opponents, 1)
        self.assertTrue(result['all_wins'])
        duplicate = copy.deepcopy(perfect)
        duplicate['rows'].append(copy.deepcopy(duplicate['rows'][0]))
        self.assertFalse(finetune.strict_report(duplicate, opponents, 1)['all_wins'])

    def test_adaptation_and_every_wave_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = finetune._resolved_config(self.configuration(root), root/'run', 16)
            dev = finetune.strict_report(report(cfg['opponents'], 1, 'dev', winning=False), cfg['opponents'], 1)
            weights = finetune.sampling_weights(dev, cfg)
            self.assertAlmostEqual(sum(weights.values()), 1.)
            self.assertGreater(weights['scripted-default'], weights['scripted-noop'])
            self.assertGreaterEqual(min(weights.values()), .05)
            counters = dict(wave=0, episodes=0,
                opponent_episodes={name: 0 for name in finetune.SCRIPTED_IDS},
                opponent_sides={name: [0, 0] for name in finetune.SCRIPTED_IDS})
            tasks, _ = finetune._wave_plan(root/'run'/'waves'/'wave_000000', cfg, 17,
                counters, {'latest_dev': dev}, TaskScorer().state_dict(), 'identity')
            self.assertEqual(len(tasks), 16)
            self.assertEqual({task['opponent']['id'] for task in tasks}, set(finetune.SCRIPTED_IDS))
            self.assertTrue(all(not task['deterministic'] for task in tasks))
            self.assertTrue(all(Path(task['directory']).is_relative_to(root/'data') for task in tasks))
            self.assertTrue(all(4_100_000 <= task['requested_seed'] <= 4_999_999 for task in tasks))
            self.assertEqual(counters['episodes'], 0)  # Planning does not commit allocation counters.

    def test_budget_dev_selection_and_single_failed_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            source_hash = file_hash(source)
            cfg = self.configuration(root)
            calls, live, evaluations = [], [0], []

            def evaluate(directory, config, checkpoint, arm, opponents, pairs, split, workers):
                self.assertEqual(live[0], 0, 'evaluation must close all training environments')
                evaluations.append(split)
                return report(opponents, pairs, split, winning=split == 'dev' and len(evaluations) > 1)

            def update(scorer, critic, actor_opt, critic_opt, batch, arm, config):
                self.assertEqual(actor_opt.state, {})  # The source optimizer was not imported.
                with torch.no_grad():
                    next(scorer.parameters()).add_(.001)
                return {'updates': 1}

            with patch('blackout_v12.finetune_workers.PersistentEpisodePool', self.fake_runtime(calls, live)), \
                    patch.object(finetune, 'evaluate', side_effect=evaluate), \
                    patch.object(finetune, 'update', side_effect=update), \
                    patch('blackout_v12.env.make_env') as unity:
                result = finetune.run(root/'run', cfg, source, learner_device='cpu')
                resumed = finetune.run(root/'run', cfg, source, learner_device='cpu')
                unity.assert_not_called()
            self.assertEqual(evaluations, ['dev', 'dev', 'confirmation'])
            self.assertEqual(result, resumed)
            self.assertEqual(result['env_steps'], 16)
            self.assertEqual(result['overshoot'], 15)
            self.assertTrue(result['best_dev']['all_wins'])
            self.assertFalse(result['empirical_target_met'])
            self.assertEqual(result['target_status'], 'unmet')
            self.assertFalse(result['confirmation_used_for_training'])
            self.assertEqual(file_hash(source), source_hash)
            self.assertEqual(read_json(root/'run'/'progress.json')['status'], 'complete')
            self.assertEqual([entry[0] for entry in calls], ['open', 'wave', 'close'])
            self.assertGreater(result['counters']['events'], 0)
            latest = torch.load(root/'run'/'latest.pt', weights_only=True)
            self.assertGreater(latest['state']['latest_update']['actor_parameter_delta_l2'], 0.)

    def test_stop_after_optimizer_rolls_back_wave_and_resumes_same_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            cfg = dict(self.configuration(root), stop_file=str(root/'STOP'))
            calls, live, evaluation_calls, stop_once = [], [0], [], [True]
            saved_pre_update = []

            def evaluate(directory, config, checkpoint, arm, opponents, pairs, split, workers):
                self.assertEqual(live[0], 0)
                evaluation_calls.append(split)
                return report(opponents, pairs, split, winning=len(evaluation_calls) > 1)

            def update(scorer, critic, actor_opt, critic_opt, batch, arm, config):
                saved_pre_update.append(finetune._rng_state('cpu'))
                with torch.no_grad():
                    next(scorer.parameters()).add_(torch.rand(()))
                random.random()
                np.random.rand()
                if stop_once[0]:
                    stop_once[0] = False
                    Path(cfg['stop_file']).touch()
                return {'updates': 1}

            with patch('blackout_v12.finetune_workers.PersistentEpisodePool', self.fake_runtime(calls, live)), \
                    patch.object(finetune, 'evaluate', side_effect=evaluate), \
                    patch.object(finetune, 'update', side_effect=update), \
                    patch('blackout_v12.env.make_env') as unity:
                with self.assertRaises(InterruptedError):
                    finetune.run(root/'run', cfg, source, learner_device='cpu')
                stopped = torch.load(root/'run'/'latest.pt', weights_only=True)
                self.assertEqual(stopped['counters']['env_steps'], 0)
                self.assertEqual(stopped['counters']['episodes'], 0)
                self.assertEqual(stopped['counters']['wave'], 0)
                original = torch.load(source, weights_only=True)
                self.assertEqual(state_hash(stopped['scorer_state']), state_hash(original['scorer_state']))
                self.assertTrue(torch.equal(stopped['rng_state']['torch'], saved_pre_update[0]['torch']))
                self.assertEqual(read_json(root/'run'/'progress.json')['status'], 'interrupted')
                Path(cfg['stop_file']).unlink()
                result = finetune.run(root/'run', cfg, source, learner_device='cpu')
                unity.assert_not_called()
            waves = [entry[1] for entry in calls if entry[0] == 'wave']
            self.assertEqual(waves[0], waves[1])
            self.assertEqual(evaluation_calls, ['dev', 'dev', 'confirmation'])
            self.assertTrue(torch.equal(saved_pre_update[0]['torch'], saved_pre_update[1]['torch']))
            self.assertEqual(saved_pre_update[0]['python'], saved_pre_update[1]['python'])
            self.assertTrue(torch.equal(saved_pre_update[0]['numpy']['keys'], saved_pre_update[1]['numpy']['keys']))
            self.assertEqual(result['env_steps'], 16)
            self.assertTrue(result['empirical_target_met'])

    def test_invalid_roster_splits_and_resume_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self.configuration(root)
            with self.assertRaisesRegex(ValueError, 'nine distinct'):
                finetune._resolved_config(dict(cfg, opponents=cfg['opponents'][:-1]), root/'run', 16)
            splits = copy.deepcopy(finetune.FRESH_SPLITS)
            splits['confirmation'] = splits['dev']
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                finetune._resolved_config(dict(cfg, splits=splits), root/'run', 16)
            source = self.source(root)
            with patch.object(finetune, 'evaluate', side_effect=lambda directory, config, checkpoint, arm,
                    opponents, pairs, split, workers: report(opponents, pairs, split)), \
                    patch('blackout_v12.finetune_workers.PersistentEpisodePool') as pool:
                result = finetune.run(root/'run', cfg, source, learner_device='cpu')
                self.assertTrue(result['empirical_target_met'])
                self.assertEqual(result['env_steps'], 0)
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    finetune.run(root/'run', dict(cfg, actor_lr=2e-4), source, learner_device='cpu')
                altered = torch.load(source, weights_only=True)
                altered['modified'] = True
                atomic_torch(source, altered)
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    finetune.run(root/'run', cfg, source, learner_device='cpu')
                pool.assert_not_called()


if __name__ == '__main__':
    unittest.main()
