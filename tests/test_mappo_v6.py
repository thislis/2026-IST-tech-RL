"""V6 contracts using synthetic transitions only; never start a Unity process."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from blackout_rl.batching import team_agents, canonical_agents
from blackout_rl.mappo_v6 import (V6Model, V6Policy, PlannerFeatures, correction_actions,
    apply_joint_action, joint_distribution, model_from_payload, CONTEXT_SIZE, SCHEMA_VERSION)
from blackout_rl.mappo_v6_training import FailureSeedSampler, V6Collector, update_v6
from blackout_rl.mappo_v6_artifacts import save_v6_checkpoint, load_v6_checkpoint, export_v6
from blackout_rl.mappo_curriculum_v6 import validate_seed_splits, DEFAULT_STAGES, gate_passed
from blackout_rl.policy import DeterministicCheckpointPolicy, NoOpPolicy
from blackout_rl.ppo import PPOConfig
from blackout_rl.representation import absorption_phase_features, GlobalLocalMapEncoder
from blackout_rl.scripted_fsm import ScriptedTeamController
from blackout_rl.team_state import Role
from blackout_rl.training_reward import TrainingRewardConfig
from scripts.train_mappo_planner_residual_v6 import build_parser, preflight, run, recover_log_tails
from tests.test_ppo_training import small_model
from tests.test_scripted_fsm import observations, BATTERIES

ROOT = Path(__file__).resolve().parents[1]


def all_observations(time_left=.9):
    obs = observations(batteries=BATTERIES, time_left=time_left)
    for i in range(5):
        vector = obs[f"unit_{i}"]["vector"].copy()
        vector[:90].reshape(10,9)[:,2] *= -1
        graphic = obs[f"unit_{i}"]["graphic"].copy()
        graphic[..., [2,3,4,5]] = graphic[..., [3,2,5,4]]
        obs[f"unit_{i+5}"] = {"vector":vector, "graphic":graphic}
    return obs


class SyntheticEnv:
    def __init__(self, **kwargs):
        self.reset_seeds = []
        self.agents = list(canonical_agents())
        self.steps = 0
    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        self.agents = list(canonical_agents())
        return all_observations(), {}
    def step(self, actions):
        if set(actions) != set(canonical_agents()):
            raise AssertionError("ten actions required")
        self.steps += 1
        ended = self.steps == 3
        flags = {a:ended for a in canonical_agents()}
        info = {"score_0":self.steps/100, "score_1":0., "time_left":.9-self.steps/21000}
        if ended:
            info.update(score_0=0., winner=0)
            self.agents = []
        return (all_observations(.9-self.steps/21000), {a:0. for a in canonical_agents()}, flags,
                {a:False for a in canonical_agents()}, {a:dict(info) for a in canonical_agents()})
    def close(self):
        pass


def model_optimizer():
    model = V6Model(small_model())
    return model, torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)


def write_checkpoint(path, model, optimizer):
    save_v6_checkpoint(path, model, optimizer, training={"schema_version":SCHEMA_VERSION,
        "config":{"seed":10600, "learning_rate":1e-4}, "global_step":0, "update":0,
        "runtime_state":{}, "run_id":"unit-test"}, source={})


def isolated_args(root, *extra):
    argv = ["--log-dir",str(root/"logs"), "--snapshot-dir",str(root/"snapshots"),
            "--export-dir",str(root/"export"), "--max-env-steps","4", "--rollout-steps","2",
            "--curriculum-scale","0.00001", "--eval-every","2", "--save-every","2"]
    for name in ("latest", "target-best", "stage-best", "target"):
        argv += [f"--{name}-checkpoint",str(root/(name+".pt"))]
    args = build_parser().parse_args(argv+list(extra))
    args.log_dir.mkdir()
    return args


def evaluation_summary(rate, seeds, step):
    return dict(episodes=2*len(seeds), wins=round(2*len(seeds)*rate), draws=0,
        losses=2*len(seeds)-round(2*len(seeds)*rate), win_rate=rate,
        mean_model_score_diff=20. if rate >= .8 else 0., global_step=step,
        by_model_side={"A":{"win_rate":rate}, "B":{"win_rate":rate}})


class V6Contracts(unittest.TestCase):
    def test_clock_matches_actual_twenty_second_boundary(self):
        phase = absorption_phase_features(torch.tensor([1-20/420, 1-5/420]))
        self.assertTrue(torch.allclose(phase[0], torch.tensor([0.,1.]), atol=1e-5))
        self.assertTrue(torch.allclose(phase[1], torch.tensor([1.,0.]), atol=1e-5))

    def test_crop_uses_top_down_row_for_world_y(self):
        image = torch.arange(64).reshape(1,1,8,8).float()
        crop = GlobalLocalMapEncoder(channels=1, crop_size=3).local_crop(image, torch.tensor([[.3125,.8125]]))
        self.assertEqual(float(crop[0,0,1,1]), float(image[0,0,1,2]))

    def test_corrections_preserve_continuous_direction_and_cover_stopped_compass(self):
        planner = np.tile(np.asarray([.6,.8], dtype=np.float32), (5,1))
        corrected = correction_actions(planner)
        self.assertTrue(np.allclose(corrected[:,5], -planner))
        self.assertTrue(np.allclose(corrected[:,4], 0))
        stopped = correction_actions(np.zeros((5,2), dtype=np.float32))
        self.assertEqual(len(np.unique(stopped[0].round(5), axis=0)), 8)
        for index in range(41):
            output = apply_joint_action(index, planner, corrected)
            self.assertLessEqual(int(np.any(output != planner, axis=1).sum()), 1)

    def test_joint_behavior_logprob_ratio_and_exploration_floor(self):
        logits = torch.full((2,41), -100., requires_grad=True)
        logits = torch.cat((torch.zeros(2,1), logits[:,1:]), -1)
        valid = torch.ones(2,41, dtype=torch.bool)
        valid[:,10:20] = False
        dist = joint_distribution(logits, valid, .15)
        self.assertTrue(torch.allclose(1-dist.probs[:,0], torch.full((2,),.15)))
        self.assertTrue(torch.equal(dist.probs[:,10:20], torch.zeros(2,10)))
        selected = torch.tensor([1,25])
        ratio = (joint_distribution(logits, valid, .15).log_prob(selected)-dist.log_prob(selected)).exp()
        self.assertTrue(torch.equal(ratio, torch.ones(2)))
        (-dist.log_prob(selected).mean()).backward()

    def test_all_invalid_corrections_fall_back_without_nan(self):
        valid = torch.zeros(41,dtype=torch.bool)
        valid[0] = True
        dist = joint_distribution(torch.zeros(41), valid, .3)
        self.assertEqual(int(dist.sample()), 0)
        self.assertTrue(bool(torch.isfinite(dist.entropy())))

    def test_real_planner_context_and_greedy_baseline_both_sides(self):
        model, _ = model_optimizer()
        obs = all_observations()
        for side in (0,1):
            features = PlannerFeatures(side)
            data, mask, planner, _ = features.prepare(obs, model)
            self.assertEqual(data.shape, (5,32+CONTEXT_SIZE))
            self.assertEqual(mask.shape, (41,))
            reference = ScriptedTeamController(side, roles=(Role.WORKER,)*3+(Role.GUARD,)*2, chase_radius_cells=48)
            actions = reference.act(obs, team_agents(side))
            policy = V6Policy(model, team=side)
            result = policy.act(obs, tuple(reversed(team_agents(side))))
            for name in team_agents(side):
                self.assertTrue(np.array_equal(result[name], actions[name]))
            repeat = policy.act(obs, team_agents(side))
            for name in result:
                self.assertTrue(np.array_equal(result[name], repeat[name]))

    def test_seed_priorities_are_train_only_and_resume_rng(self):
        sampler = FailureSeedSampler([1,2,3], 42)
        for _ in range(20):
            sampler.observe(1, "loss")
            sampler.observe(2, "win")
        self.assertGreater(sampler.probabilities()[0], sampler.probabilities()[1])
        self.assertGreaterEqual(float(sampler.probabilities().min()), .4/3)
        restored = FailureSeedSampler([1,2,3], 9, sampler.state_dict())
        self.assertEqual([sampler.next_seed() for _ in range(20)], [restored.next_seed() for _ in range(20)])
        with self.assertRaises(ValueError):
            sampler.observe(3101, "loss")

    def test_splits_reject_confirmation_leakage(self):
        splits = json.loads((ROOT/"configs/seed_splits_v6.json").read_text())
        validate_seed_splits(splits)
        splits["train"].append(splits["confirmation"][0])
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_seed_splits(splits)

    def test_baseline_relative_readiness_does_not_require_nine_wins(self):
        baseline = dict(win_rate=.8, mean_model_score_diff=21.5,
                        by_model_side={"A":{"win_rate":1.}, "B":{"win_rate":.6}})
        self.assertTrue(gate_passed(DEFAULT_STAGES[1], baseline, baseline, stage_steps=51200))
        self.assertFalse(gate_passed(DEFAULT_STAGES[-1], baseline, baseline, stage_steps=200704))

    def test_collector_continuity_team_gae_and_frozen_encoder(self):
        model, optimizer = model_optimizer()
        env = SyntheticEnv()
        opponent = NoOpPolicy()
        opponent.reset = lambda:None
        collector = V6Collector(env, model, opponent, team=0,
            sampler=FailureSeedSampler([1,2], 1), reward_config=TrainingRewardConfig())
        batch, metrics = collector.collect(1, exploration=0., gamma=.99, gae_lambda=.95, force_planner=True)
        self.assertEqual(int(batch["action"][0]), 0)
        batch, metrics = collector.collect(5, exploration=.15, gamma=.99, gae_lambda=.95)
        self.assertEqual(collector.counts["terminal_episodes"], 2)
        self.assertEqual(len(env.reset_seeds), 3)
        self.assertEqual(batch["features"].shape[:2], (5,5))
        self.assertEqual(batch["old_log_prob"].shape, (5,))
        before = {k:v.clone() for k,v in model.actor_model.state_dict().items()}
        head = model.residual[-1].weight.detach().clone()
        result = update_v6(model, optimizer, batch, PPOConfig(update_epochs=1, minibatch_size=5))
        self.assertEqual(result["samples"], 5)
        self.assertTrue(np.isfinite(result["value_loss"]))
        self.assertFalse(torch.equal(head, model.residual[-1].weight))
        self.assertTrue(all(torch.equal(v, model.actor_model.state_dict()[k]) for k,v in before.items()))

    def test_checkpoint_save_does_not_perturb_rng_and_policy_routes_v6(self):
        model, optimizer = model_optimizer()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"checkpoint.pt"
            rng = torch.get_rng_state().clone()
            write_checkpoint(path, model, optimizer)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            loaded, _, _ = load_v6_checkpoint(path)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k,v in model.state_dict().items()))
            policy = DeterministicCheckpointPolicy(path, team=0)
            self.assertIsNotNone(policy._v6_policy)
            self.assertEqual(len(policy.act(all_observations(), team_agents(0))), 5)

    def test_two_file_export_matches_project_on_sequence_and_reset(self):
        model, optimizer = model_optimizer()
        # Exercise a non-fallback head: packaging only the planner must fail this test.
        with torch.no_grad():
            model.residual[-1].bias.fill_(-10)
            model.residual[-1].bias[0] = 5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root/"source.pt"
            write_checkpoint(checkpoint, model, optimizer)
            target = root/"standalone"
            export_v6(checkpoint, target)
            self.assertEqual({p.name for p in target.iterdir()}, {"policy.py", "checkpoint.pt"})
            spec = importlib.util.spec_from_file_location("v6_export_test", target/"policy.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            exported = module.load_policy(target/"checkpoint.pt")
            for side in (0,1):
                reference = V6Policy(model, team=side)
                for time_left in (.9, .899, .898, 1., .999):
                    obs = all_observations(time_left)
                    selected = {a:obs[a] for a in reversed(team_agents(side))}
                    actual = exported.act(selected)
                    expected = reference.act(obs, team_agents(side))
                    for name in expected:
                        self.assertTrue(np.array_equal(actual[name], expected[name]))
            # Fresh isolated interpreter can load using only the exported files;
            # no project directory, no installed blackout-env, no Unity imports.
            code = "import sys, importlib.util; s=importlib.util.spec_from_file_location('policy',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); p=m.load_policy(sys.argv[2]); assert not any(n.startswith(('blackout_rl','blackout_env','mlagents')) for n in sys.modules); print('standalone_ok')"
            completed = subprocess.run([sys.executable, "-I", "-c", code, str(target/"policy.py"), str(target/"checkpoint.pt")],
                cwd=target, text=True, capture_output=True, check=True)
            self.assertIn("standalone_ok", completed.stdout)

    def test_preflight_does_not_start_unity_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            args = build_parser().parse_args(["--log-dir", str(Path(temp)/"logs"), "--check"])
            with patch("scripts.train_mappo_planner_residual_v6.ContractBlackOutEnv", side_effect=AssertionError("Unity forbidden")):
                preflight(args)
            self.assertFalse(args.log_dir.exists())

    def test_crash_ahead_log_records_are_preserved_before_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"training.jsonl"
            path.write_text('committed\nuncommitted\n')
            recovered = recover_log_tails(root, {"training.jsonl":len('committed\n')})
            self.assertEqual(path.read_text(), 'committed\n')
            self.assertEqual(Path(recovered[0]).read_text(), 'uncommitted\n')

    def test_runner_logs_stages_and_resume_without_unity(self):
        def fake_evaluate(args, checkpoint, seeds, opponent_id, split, step, label):
            win_rate = .8 if opponent_id == "base_scripted" else .5
            return dict(episodes=2*len(seeds), wins=round(2*len(seeds)*win_rate), draws=0,
                losses=round(2*len(seeds)*(1-win_rate)), win_rate=win_rate,
                mean_model_score_diff=21.5 if opponent_id=="base_scripted" else 0., global_step=step,
                by_model_side={"A":{"win_rate":1. if win_rate==.8 else .6},
                               "B":{"win_rate":.6 if win_rate==.8 else .4}})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = ["--log-dir",str(root/"logs"), "--snapshot-dir",str(root/"snapshots"),
                    "--export-dir",str(root/"export"), "--max-env-steps","4", "--rollout-steps","2",
                    "--curriculum-scale","0.00001", "--eval-every","2", "--save-every","2"]
            for name in ("latest", "target-best", "stage-best", "target"):
                argv += [f"--{name}-checkpoint",str(root/(name+".pt"))]
            args = build_parser().parse_args(argv)
            args.log_dir.mkdir()
            checked = preflight(args)
            with patch("scripts.train_mappo_planner_residual_v6.ContractBlackOutEnv", SyntheticEnv), patch(
                    "scripts.train_mappo_planner_residual_v6.evaluate", side_effect=fake_evaluate):
                self.assertEqual(run(args, checked), 0)
                summary = json.loads((args.log_dir/"run_summary.json").read_text())
                self.assertEqual(summary["status"], "budget_exhausted")
                records = [json.loads(line) for line in (args.log_dir/"training.jsonl").read_text().splitlines()]
                self.assertTrue(any(r.get("record_type")=="stage_transition" for r in records))
                self.assertTrue((args.log_dir/"source_snapshot.zip").is_file())
                # A crash can leave an alias ahead of the saved runtime state.
                args.target_best_checkpoint.write_bytes(b"uncommitted alias")
                args.resume_latest = True
                args.max_env_steps = 6
                checked = preflight(args)
                self.assertEqual(run(args, checked), 0)
                torch.load(args.target_best_checkpoint, weights_only=True)
            summary = json.loads((args.log_dir/"run_summary.json").read_text())
            self.assertEqual(summary["global_step"], 6)

    def test_runner_confirmed_rollback_restores_baseline_and_stops_at_limit(self):
        def fake_evaluate(args, checkpoint, seeds, opponent_id, split, step, label):
            rate = .8 if opponent_id == "base_scripted" else .5 if label.startswith("baseline") else .2
            return evaluation_summary(rate, seeds, step)
        with tempfile.TemporaryDirectory() as temp:
            args = isolated_args(Path(temp), "--max-rollbacks", "1")
            checked = preflight(args)
            with patch("scripts.train_mappo_planner_residual_v6.ContractBlackOutEnv", SyntheticEnv), patch(
                    "scripts.train_mappo_planner_residual_v6.evaluate", side_effect=fake_evaluate):
                self.assertEqual(run(args, checked), 3)
            summary = json.loads((args.log_dir/"run_summary.json").read_text())
            self.assertEqual(summary["status"], "rollback_limit")
            self.assertEqual(summary["runtime_state"]["rollback_count"], 1)
            self.assertEqual(summary["latest_target_evaluation"]["win_rate"], .5)
            checkpoint = torch.load(args.latest_checkpoint, weights_only=True)
            immutable = summary["runtime_state"]["artifact_identities"]["target_best"]["path"]
            baseline = torch.load(immutable, weights_only=True)
            self.assertTrue(all(torch.equal(v, baseline["v6_model_state"][k]) for k,v in checkpoint["v6_model_state"].items()))
            args.resume_latest = True
            with self.assertRaisesRegex(ValueError, "rollback_limit"):
                preflight(args)

    def test_target_qualification_exports_and_test_result_does_not_select(self):
        def fake_evaluate(args, checkpoint, seeds, opponent_id, split, step, label):
            rate = .0 if split == "test" else .8 if opponent_id == "base_scripted" else .5 if label.startswith("baseline") else .9
            return evaluation_summary(rate, seeds, step)
        with tempfile.TemporaryDirectory() as temp:
            args = isolated_args(Path(temp))
            checked = preflight(args)
            with patch("scripts.train_mappo_planner_residual_v6.ContractBlackOutEnv", SyntheticEnv), patch(
                    "scripts.train_mappo_planner_residual_v6.evaluate", side_effect=fake_evaluate):
                self.assertEqual(run(args, checked), 0)
            summary = json.loads((args.log_dir/"run_summary.json").read_text())
            self.assertEqual(summary["status"], "target_reached")
            self.assertEqual(summary["runtime_state"]["final_test_evaluation"]["win_rate"], 0.)
            self.assertTrue(args.target_checkpoint.is_file())
            self.assertEqual({p.name for p in args.export_dir.iterdir()}, {"policy.py", "checkpoint.pt"})
            # Interrupted finalization may safely repeat this exact export.
            export_v6(args.target_checkpoint, args.export_dir)


if __name__ == "__main__":
    unittest.main()
