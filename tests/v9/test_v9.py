"""Critical v9 tests; all games are synthetic and never launch Unity."""
import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from blackout_v9.benchmark import fixture, WorkerTuner
from blackout_v9.collector import collect_episode
from blackout_v9.contracts import config
from blackout_v9.export import export_policy, verify_export
from blackout_v9.evaluation import evaluate
from blackout_v9.io import atomic_json, atomic_torch, cpu_state, file_hash, load_checkpoint
from blackout_v9.league import League
from blackout_v9.opponents import StrategyOpponent, HistoricalOpponent
from blackout_v9.policy import MyPolicy
from blackout_v9.ppo import optimizers, update
from blackout_v9.rewards import gae_episode, shape_episode
from blackout_v9.runtime import run_tasks
from blackout_v9.statistics import summarize, wilson
from blackout_v9.training import train_run
from blackout_v9.trajectory import Episode, EpisodeWriter, WaveBatch, pack
from blackout_v9.value import Critic


def observations(step=0, terminal=False):
    obs = {}
    graphic = np.zeros((96, 96, 11), np.float32); graphic[..., 0] = 1
    graphic[0, :, 0] = 0; graphic[0, :, 1] = 1
    for observer in range(10):
        vector = np.zeros(96, np.float32)
        units = vector[:90].reshape(10, 9)
        units[:, :2] = .2 + np.arange(10)[:, None]*.03 + step*.002
        units[:, 2] = np.asarray([1]*5+[-1]*5) * (1 if observer < 5 else -1)
        units[:, 3] = 1
        if step == 1:
            units[0, 3] = 0; units[0, 4] = 1
        vector[90] = 1
        vector[93 if observer < 5 else 94] = .01 if step >= 2 and not terminal else 0
        vector[95] = 1-step/21000
        g = graphic.copy()
        vector.setflags(write=False); g.setflags(write=False)
        obs[f"unit_{observer}"] = {"vector": vector, "graphic": g}
    return obs


class Game:
    def __init__(self, length=3, fail=None, partial=False):
        self.possible_agents = [f"unit_{i}" for i in range(10)]
        self.length, self.fail, self.partial = length, fail, partial
        self.closed, self.steps, self.resets = False, 0, 0
    def reset(self, seed=None):
        self.resets += 1; self.agents = self.possible_agents.copy()
        return observations(), {}
    def step(self, actions):
        self.steps += 1
        assert set(actions) == set(self.possible_agents)
        if self.steps == self.fail: raise RuntimeError("synthetic transport failure")
        done = self.steps >= self.length
        if done: self.agents = []
        terms = {n: done for n in self.possible_agents}
        if self.partial: terms["unit_0"] = True
        raw = {n: (.2 + float(done) if i < 5 else -.2-float(done)) for i, n in enumerate(self.possible_agents)}
        return observations(self.steps, done), raw, terms, {n: False for n in self.possible_agents}, {n: {"winner": 1} for n in self.possible_agents}
    def close(self): self.closed = True


def small_config():
    cfg = copy.deepcopy(config())
    cfg.update(workers=1, worker_cap=1, max_episode_steps=5, training_steps=6, pilot_steps=6, evaluate_every=100,
               history_every=3, episode_wall_seconds=10, stall_seconds=5, stop_grace_seconds=.5, disk_stop_gib=0)
    cfg["ppo"].update(epochs=1, minibatch=2)
    return cfg


def task_for(directory, cfg, side=0, ident="fixture", policy_sha="fixture-policy"):
    return {"id": ident, "kind": "collect", "config": cfg, "side": side, "action_seed": 11,
            "requested_seed": 99, "trajectory": str(Path(directory) / ident / "trajectory"),
            "invalid_path": str(Path(directory) / ident / "invalid.json"),
            "policy_sha256": policy_sha, "opponent": {"kind": "noop"}}


class V9Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(11)

    def test_contract_permutation_split_and_all_channels_gradient(self):
        p = MyPolicy()
        v, g = fixture()
        original = (v.clone(), g.clone())
        for n in (0, 1, 3, 5, 10):
            vv, gg = v.repeat(2, 1)[:n], g.repeat(2, 1, 1, 1)[:n]
            a = p(vv, gg)
            self.assertEqual(a.shape, (n, 2)); self.assertEqual(a.dtype, torch.float32)
            self.assertTrue(torch.isfinite(a).all() and (a.abs() <= 1).all())
        order = torch.tensor([3, 0, 4, 2, 1])
        logs = p.log_probs(v, g)
        torch.testing.assert_close(p.log_probs(v[order], g[order]), logs[order], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(torch.cat([p.log_probs(v[i:i+1], g[i:i+1]) for i in range(5)]), logs, atol=2e-6, rtol=1e-5)
        with self.assertRaises(TypeError): p(v.double(), g)
        bad = v.clone(); bad[0, 0] = torch.nan
        with self.assertRaises(ValueError): p(bad, g)
        dense = torch.rand_like(g, requires_grad=True)
        (-p.log_probs(v, dense)[:, 1].mean()).backward()
        self.assertTrue((dense.grad.abs().sum((0, 2, 3)) > 0).all())
        self.assertFalse(any(isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)) for m in p.modules()))
        torch.testing.assert_close(v, original[0]); torch.testing.assert_close(g, original[1])

    def test_decoder_not_changed_by_train_eval_and_mixture_normalized(self):
        p = MyPolicy(); p.configure("C")
        v, g = fixture()
        torch.manual_seed(31); a = p.train()(v, g)
        torch.manual_seed(31); b = p.eval()(v, g)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        logs, features, gates = p.team_distribution(v[None], g[:1])
        torch.testing.assert_close(logs.exp().sum(-1), torch.ones(1, 5))
        torch.testing.assert_close(gates.sum(-1), torch.ones(1, 5))
        torch.testing.assert_close(logs[0], p.log_probs(v, g), atol=2e-6, rtol=1e-5)

    def test_two_file_isolated_actual_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = MyPolicy().eval(); bundle = export_policy(p, Path(tmp) / "submission")
            result = verify_export(bundle, p, fixture())
            self.assertTrue(result["passed"] and result["provided_loader"])
            self.assertEqual({p.name for p in bundle.iterdir()}, {"policy.py", "checkpoint.pt"})

    def test_reward_bound_loops_discount_and_gae_boundaries(self):
        for length in (2, 21000, 100000):
            phi = np.sin(np.arange(length))*.25
            for gamma in (1., .99995):
                rewards = shape_episode(phi, 1, gamma)
                total = float(np.dot(gamma**np.arange(length), rewards))
                self.assertAlmostEqual(total, gamma**(length-1)-phi[0], places=5)
        a, r = gae_episode(np.array([0., 1.]), np.zeros(2), 1., 1.)
        np.testing.assert_array_equal(r, [1., 1.])
        b, r = gae_episode(np.array([-1.]), np.zeros(1), 1., 1.)
        np.testing.assert_array_equal(r, [-1.])
        with self.assertRaises(ValueError): shape_episode([float("nan")], 1)

    def test_popart_preserves_unnormalized_value(self):
        c = Critic(); v = torch.rand(7, 5, 96)
        before = c(v).detach()
        c.update_scale(torch.tensor([-10., 20., 2.]), beta=.5)
        torch.testing.assert_close(c(v), before, atol=1e-6, rtol=1e-5)

    def test_graphic_codec_nonbinary_is_exact(self):
        vectors, graphic = pack(observations(), [f"unit_{i}" for i in range(5)])
        other = graphic.copy(); other[0, 5, 5] = .75; other[1, 5, 5] = .25
        with tempfile.TemporaryDirectory() as tmp:
            writer = EpisodeWriter(Path(tmp) / "episode", 2)
            for g in (graphic, other):
                writer.add(vectors=vectors, graphic=g, actions=np.zeros(5, np.int64), log_probs=np.zeros(5, np.float32),
                           values=0., potentials=0., aux_targets=np.zeros(3, np.float32))
            with self.assertRaises(ValueError): writer.commit(1, 1, 1, {"natural_terminal": False})
            writer.commit(1, 1, 1, {"natural_terminal": True, "policy_sha256": "one"})
            ep = Episode(Path(tmp) / "episode")
            np.testing.assert_array_equal(ep.graphic([0, 1]), np.stack([graphic, other]))

    def test_complete_episode_raw_outcome_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = Game(); task = task_for(tmp, small_config())
            result = collect_episode(MyPolicy().eval(), Critic().eval(), StrategyOpponent("noop"), task, lambda: game)
            self.assertTrue(game.closed); self.assertEqual(game.resets, 1)
            self.assertEqual(result["outcome"], 1)
            self.assertEqual(result["wrapper_terminal_proxy"]["unit_0"], 1)  # Incorrect proxy deliberately ignored.
            self.assertLessEqual(abs(result["train_return"]), 1.25)
            self.assertEqual(result["behavior"]["delivery_proxy"], 1)
            self.assertEqual(result["behavior"]["pickup_proxy"], 1)

    def test_watchdog_transport_partial_terminal_never_commit(self):
        for game in (Game(length=100), Game(fail=2), Game(partial=True)):
            with tempfile.TemporaryDirectory() as tmp:
                p, c = MyPolicy().eval(), Critic().eval()
                before = cpu_state(p); task = task_for(tmp, small_config())
                with self.assertRaises((RuntimeError, ValueError)):
                    collect_episode(p, c, StrategyOpponent("noop"), task, lambda: game)
                self.assertFalse((Path(task["trajectory"]) / "committed.json").exists())
                self.assertTrue(Path(task["invalid_path"]).exists()); self.assertTrue(game.closed)
                for k, v in p.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_ppo_updates_encoder_and_shared_team_agent_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = MyPolicy().eval(), Critic().eval(); cfg = small_config()
            task = task_for(tmp, cfg)
            collect_episode(p, c, StrategyOpponent("noop"), task, Game)
            batch = WaveBatch([task["trajectory"]])
            rows = batch.minibatch(np.arange(len(batch)), "cpu")
            logs = p.team_distribution(rows["vectors"], rows["graphic"])[0]
            selected = logs.gather(-1, rows["actions"][..., None]).squeeze(-1)
            torch.testing.assert_close(selected, rows["log_probs"], atol=2e-6, rtol=1e-5)
            opts = optimizers(p, c, cfg["ppo"])
            metrics = update(p, c, *opts, batch, cfg["ppo"], np.random.default_rng(1), "A")
            self.assertGreater(metrics["parameter_delta"]["patch.weight"], 0)

    def test_official_runner_side_swap_uses_actual_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_policy(MyPolicy(), Path(tmp) / "bundle")
            for side in (0, 1):
                task = {"bundle": str(bundle), "config": small_config(), "side": side,
                        "replicate": 0, "action_seed": 1, "opponent": {"kind": "noop"}}
                result = evaluate(task, Game)
                self.assertEqual(result["winner"], side)

    @staticmethod
    def mock_tasks(tasks, directory, cfg, stop=lambda: False, progress=lambda **_: None):
        results = []
        for task in tasks:
            path = Path(directory) / task["id"]; path.mkdir(parents=True)
            task = dict(task, trajectory=str(path / "trajectory"), invalid_path=str(path / "invalid.json"))
            atomic_json(path / "task.json", task)
            payload = torch.load(task["policy_path"], weights_only=True)
            p, c = MyPolicy().eval(), Critic().eval()
            p.load_state_dict(payload["policy_state"]); c.load_state_dict(payload["critic_state"])
            result = collect_episode(p, c, StrategyOpponent("noop"), task, Game)
            row = {"id": task["id"], "task": task, "directory": str(path), "valid": True,
                   "result": result, "error": None, "attempt_steps": 3, "interrupted": False}
            atomic_json(path / "attempt.json", row); results.append(row)
        return results

    def test_training_checkpoint_resume_and_registration_rejection(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as tmp, patch("blackout_v9.training.run_tasks", self.mock_tasks):
            policy, result = train_run(tmp, cfg, "B", 11, "registered", "cpu")
            self.assertTrue(result["complete"]); self.assertEqual(result["step"], 6)
            saved, _ = load_checkpoint(Path(tmp) / "checkpoints")
            before = cpu_state(policy)
            resumed, again = train_run(tmp, cfg, "B", 11, "registered", "cpu")
            self.assertEqual(again["step"], 6)
            for k, v in resumed.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            self.assertIsNone(saved["state"]["pending_wave"])
            with self.assertRaisesRegex(ValueError, "provenance"):
                train_run(tmp, cfg, "B", 11, "changed", "cpu")

    def test_invalid_wave_never_changes_optimizer_or_weights(self):
        cfg = small_config()
        def invalid(tasks, directory, config, *args):
            result = self.mock_tasks(tasks, directory, config)
            result[0].update(valid=False, error="synthetic invalid terminal", result=None)
            return result
        with tempfile.TemporaryDirectory() as tmp, patch("blackout_v9.training.run_tasks", invalid):
            with self.assertRaisesRegex(RuntimeError, "invalid episode gate"):
                train_run(tmp, cfg, "A", 11, "registered", "cpu")
            saved, _ = load_checkpoint(Path(tmp) / "checkpoints")
            torch.manual_seed(11); reference = MyPolicy()
            self.assertEqual(saved["step"], 0)
            self.assertFalse(saved["actor_optimizer"]["state"])
            for k, v in reference.state_dict().items(): torch.testing.assert_close(v, saved["policy_state"][k], rtol=0, atol=0)

    def test_runtime_detached_group_and_external_watchdog(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tasks([{"id": "detached"}], Path(tmp) / "a", cfg, worker_module="tests.v9.runtime_fixture")
            self.assertTrue(result[0]["valid"])
            row = result[0]["result"]
            self.assertEqual(row["pid"], row["session"]); self.assertEqual(row["pid"], row["group"])
            self.assertNotEqual(row["group"], os.getpgrp()); self.assertFalse(row["stdin_isatty"])
            cfg.update(episode_wall_seconds=2, stall_seconds=2)
            result = run_tasks([{"id": "hung", "hang": True}], Path(tmp) / "b", cfg, worker_module="tests.v9.runtime_fixture")
            self.assertFalse(result[0]["valid"]); self.assertIn("watchdog", result[0]["error"])

    def test_statistics_no_zero_uncertainty_claim_and_worker_cap(self):
        self.assertGreater(wilson(0, 60)[1], 0)
        rows = [{"winner": 1, "opponent": {"kind": "target"}, "side": s, "initial_observation_sha256": str(s)} for s in (0, 1)]
        report = summarize(rows, 1)
        self.assertEqual(report["invalid_rate"], 1/3); self.assertTrue(report["map_independence_unverified"])
        tuner = WorkerTuner(1, 1)
        self.assertEqual(tuner.observe(1, 3, 1, 0), 1)
        tuner = WorkerTuner(4, 8)
        self.assertLess(tuner.observe(4, 100, 1, 40), 4)

    def test_historical_target_accepts_uncanonical_environment_dict_order(self):
        from tests.test_scripted_fsm import observations as scripted_observations, BATTERIES
        obs = scripted_observations(batteries=BATTERIES)
        names = ["unit_3", "unit_2", "unit_4", "unit_1", "unit_0"]
        result = HistoricalOpponent("target").act({n: obs[n] for n in names}, names)
        self.assertEqual(set(result), set(names))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_real_optimizer_checkpoint_and_cpu_export(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as tmp, patch("blackout_v9.training.run_tasks", self.mock_tasks):
            policy, result = train_run(tmp, cfg, "C", 22, "mps-test", "mps")
            self.assertTrue(result["complete"])
            saved, _ = load_checkpoint(Path(tmp) / "checkpoints")
            self.assertEqual(saved["policy_state"]["patch.weight"].device.type, "cpu")
            self.assertTrue(saved["actor_optimizer"]["state"])
            self.assertTrue(torch.isfinite(policy(*fixture())).all())

    def test_actual_background_supervisor_returns_and_stops(self):
        from blackout_v9 import runner
        from blackout_v9.io import ROOT
        original_popen = subprocess.Popen
        children = []
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); directory = temp / "fixture"
            (temp / "logs/v9").mkdir(parents=True)
            def launch(command, **kwargs):
                if any(str(x).endswith("v9_experiments.py") for x in command):
                    code = "import sys,runpy; sys.path.insert(0," + repr(str(ROOT)) + "); runpy.run_module('tests.v9.supervisor_fixture',run_name='__main__')"
                    command = [sys.executable, "-u", "-c", code, *command[3:]]
                    kwargs["env"] = dict(kwargs["env"], V9_FIXTURE_DIRECTORY=str(directory))
                    process = original_popen(command, **kwargs); children.append(process)
                    return process
                return original_popen(command, **kwargs)
            try:
                with patch.object(runner, "ROOT", temp), patch.object(runner, "directory_for", return_value=directory), \
                     patch.object(runner, "check", return_value={"passed": True}), \
                     patch.object(runner.subprocess, "Popen", launch), patch.object(sys, "argv", ["v9", "--name", "fixture"]):
                    runner.main()
                self.assertEqual(len(children), 1)
                self.assertIsNone(children[0].poll())
                self.assertTrue(runner.is_active(directory))
                self.assertEqual(os.getsid(children[0].pid), children[0].pid)
                (directory / "stop.request").touch()
                self.assertEqual(children[0].wait(timeout=15), 0)
                self.assertFalse(runner.is_active(directory))
                self.assertEqual(json.loads((directory / "status.json").read_text())["state"], "stopped")
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate(); child.wait(timeout=10)

    def test_entire_study_gates_arms_confirmation_and_submission_synthetic(self):
        from blackout_v9 import study
        cfg = small_config()
        cfg.update(arms=["A", "B", "C"], seeds=[11], confirmation_seeds=[44],
                   dev_replicates=1, test_replicates=1, evaluation_families=["noop"])
        def tasks(items, directory, cfg, stop=lambda: False, progress=lambda **_: None):
            if not items or items[0]["kind"] == "collect":
                return self.mock_tasks(items, directory, cfg, stop, progress)
            rows = []
            for task in items:
                path = Path(directory) / task["id"]; path.mkdir(parents=True)
                atomic_json(path / "task.json", task)
                result = evaluate(task, Game)
                row = {"id": task["id"], "task": task, "directory": str(path), "valid": True,
                       "result": result, "error": None, "attempt_steps": 3, "step_count_complete": True, "interrupted": False}
                atomic_json(path / "attempt.json", row); rows.append(row)
            return rows
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(study, "ROOT", Path(tmp)), patch.object(study, "verify_original", return_value={}), \
             patch.object(study, "benchmark_inference", return_value={"selected_actor_threads": 1}), \
             patch.object(study, "choose_learner", return_value={"selected_device": "cpu"}), \
             patch.object(study, "run_tasks", tasks), patch("blackout_v9.training.run_tasks", tasks), \
             patch("blackout_v9.windows.visible_windows", return_value={"verified": True, "onscreen_owned_windows": []}):
            directory = Path(tmp) / "study"
            result = study.run_study(directory, cfg, "study-fixture", lambda: False, lambda **kw: None, lambda: 0.)
            self.assertTrue(result["complete"] and result["format_ready"])
            self.assertEqual(set(result["results"]), {"A", "B", "C"})
            self.assertEqual(result["main_training_steps"], 18)
            self.assertEqual(result["confirmation_steps"], 6)
            self.assertFalse(result["external_submission_sent"])
            self.assertFalse(result["official_server_certified"])
            self.assertEqual({p.name for p in Path(result["submission"]).iterdir()}, {"policy.py", "checkpoint.pt"})
            self.assertTrue(json.loads((directory / "lifecycle/result.json").read_text())["passed"])


if __name__ == "__main__": unittest.main()
