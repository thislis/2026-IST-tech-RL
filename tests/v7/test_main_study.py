"""Main-study budget, pilot isolation, provenance and detached-entrypoint regressions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import connectome_main as main
from scripts import connectome_experiments as runner
from scripts import pilot_evaluation as evaluation
from blackout_rl.v7_registry import load_config


class MainStudyTests(unittest.TestCase):
    def test_registered_budgets_and_runs_are_separate_from_pilots(self):
        total = 0
        identities = set()
        for suite, expected_count, seeds in [('v7-1', 20, [11, 22, 33, 44, 55]),
                                             ('v7-2', 3, [11, 22, 33])]:
            jobs = json.loads(main.QUEUES[suite].read_text())['jobs']
            self.assertEqual(len(jobs), expected_count)
            variants = set()
            for job in jobs:
                cfg = load_config(main.ROOT / job['config'])
                self.assertTrue(cfg['experiment_id'].endswith('_main_2m_v1'))
                self.assertEqual(cfg['training']['max_environment_steps'], 2_000_000)
                self.assertEqual(cfg['training']['independent_training_seeds'], seeds)
                self.assertIn(job['seed'], seeds)
                identity = (cfg['experiment_id'], job['seed'])
                self.assertNotIn(identity, identities)
                identities.add(identity)
                self.assertGreaterEqual(cfg['training']['opponent_schedule'][-1]['until'], 2_000_000)
                total += cfg['training']['max_environment_steps']
                if suite == 'v7-1':
                    variants.add(cfg['model']['variant'])
                else:
                    self.assertEqual(cfg['controller']['training_mode'], 'F1_readout_ppo')
                    self.assertEqual(cfg['controller']['remaining_team_policy'], 'fixed_scripted_active_slots_v2')
            if suite == 'v7-1':
                self.assertEqual(variants, {'mlp', 'matched_mlp', 'rewired', 'graph'})
        self.assertEqual(total, 46_000_000)
        for name in ('v7_1_flywire', 'v7_2_readout_ppo'):
            self.assertLess(load_config(main.ROOT / f'configs/v7/{name}.yaml')['training']['max_environment_steps'], 2_000_000)

    def test_registration_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text('budget: 2000000\n')
            snapshot = root / 'snapshot.zip'
            snapshot.write_bytes(b'registered source snapshot')
            record = dict(schema='blackout.v7.main_study.v1', sources={},
                          files={'config.yaml': main.sha256(config)}, source_snapshot='snapshot.zip',
                          source_snapshot_sha256=main.sha256(snapshot))
            registration = root / 'registration.json'
            registration.write_text(json.dumps(record))
            with patch.object(main, 'ROOT', root), patch.object(main, 'REGISTRATION', registration):
                main.validate_registration()
                config.write_text('budget: 100\n')
                with self.assertRaisesRegex(ValueError, 'file changed'):
                    main.validate_registration()

    def test_stage_uses_main_queue_and_current_sources(self):
        with patch.object(sys, 'argv', ['connectome_main.py', '--stage', 'v7-1', '--check']), \
                patch.object(main, 'validate_registration'), patch.object(evaluation, 'main') as run:
            main.main()
            run.assert_called_once()
            self.assertIn('--current-runtime', sys.argv)
            self.assertIn('--pipeline', sys.argv)
            self.assertEqual(sys.argv[sys.argv.index('--queue') + 1], str(main.QUEUES['v7-1']))
            self.assertEqual(sys.argv[sys.argv.index('--output-dir') + 1], str(main.OUTPUTS['v7-1']))

    def test_detached_worker_reenters_main_instead_of_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experiment.pid').write_text('123')
            with patch.object(sys, 'argv', ['start_connectome_main.sh']), \
                    patch.object(runner, 'DIRECTORY', root), patch.object(runner, 'ENTRYPOINT', Path(main.__file__)), \
                    patch.object(runner, 'command', main.command), patch.object(runner.subprocess, 'run'), \
                    patch.object(runner.subprocess, 'Popen') as popen:
                popen.return_value.poll.return_value = None
                popen.return_value.pid = 123
                runner.main()
                args = popen.call_args
                self.assertEqual(args.args[0][2], str(Path(main.__file__)))
                self.assertTrue(args.kwargs['start_new_session'])
                self.assertEqual(len(args.kwargs['pass_fds']), 1)


if __name__ == '__main__':
    unittest.main()
