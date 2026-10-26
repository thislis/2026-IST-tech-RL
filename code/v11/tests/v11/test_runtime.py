"""Focused runtime contracts; these tests never launch Unity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from blackout_v11.benchmark import apply_profile, benchmark_workers
from blackout_v11.io import atomic_json, digest, read_json
from blackout_v11.runtime import run_tasks


class RuntimeContracts(unittest.TestCase):
    def test_devices_are_independent_and_profile_does_not_mutate_registration(self):
        cfg = {"ppo": {"minibatch_sequences": 16}, "device": "auto", "workers": 2}
        before = copy.deepcopy(cfg)
        profile = {"actor": {"selected_actor_device": "cpu", "selected_planner_device": "mps",
                              "selected_actor_threads": 2},
                   "learner": {"selected_device": "mps", "selected_batch_sequences": 8},
                   "workers": {"selected_workers": 4}}
        resolved = apply_profile(cfg, profile)
        self.assertEqual(cfg, before)
        self.assertEqual((resolved["resolved_actor_device"], resolved["resolved_planner_device"],
                          resolved["resolved_learner_device"]), ("cpu", "mps", "mps"))
        self.assertEqual(resolved["runtime_profile_sha256"], digest(profile))

    def test_worker_choice_uses_throughput_and_five_percent_tie_rule(self):
        cfg = {"workers": 2, "worker_cap": 8, "memory_soft_gib": 32,
               "benchmark_opponent": {"kind": "noop"}}
        totals = {2: 500, 4: 1000, 8: 1040}
        def tasks(task_list, directory, probe_cfg, stop, progress):
            count = probe_cfg["workers"]
            return [dict(valid=True, attempt_steps=totals[count]/count, error=None,
                         result={"natural_terminal": False}) for _ in task_list]
        with tempfile.TemporaryDirectory() as temp, \
             patch("blackout_v11.runtime.run_tasks", side_effect=tasks), \
             patch("blackout_v11.benchmark.shutil.disk_usage", return_value=SimpleNamespace(free=1024**4)), \
             patch("blackout_v11.benchmark.time.perf_counter", side_effect=[0,10,10,20,20,30]):
            report = benchmark_workers(temp, cfg)
        self.assertEqual(report["selected_workers"], 4)
        self.assertTrue(all(row["training_steps"] == 0 for row in report["probes"]))

    def test_worker_probe_skips_configuration_that_cannot_fit_a_training_wave(self):
        cfg = {"workers": 2, "worker_cap": 8, "memory_soft_gib": 32,
               "benchmark_opponent": {"kind": "noop"}, "benchmark_workers": [2, 8]}
        def tasks(task_list, directory, probe_cfg, stop, progress):
            self.assertEqual(probe_cfg["workers"], 2)
            return [dict(valid=True, attempt_steps=100, error=None,
                         result={"natural_terminal": False}) for _ in task_list]
        with tempfile.TemporaryDirectory() as temp, \
             patch("blackout_v11.runtime.run_tasks", side_effect=tasks), \
             patch("blackout_v11.benchmark.shutil.disk_usage", return_value=SimpleNamespace(free=64*1024**3)), \
             patch("blackout_v11.benchmark.time.perf_counter", side_effect=[0,10]):
            report = benchmark_workers(temp, cfg)
        self.assertEqual(report["selected_workers"], 2)
        self.assertTrue(report["probes"][1]["skipped"])
        self.assertFalse(report["probes"][1]["valid"])

    def test_episode_accounting_counts_environment_steps_not_decision_events(self):
        cfg = dict(workers=1, episode_wall_seconds=20, stall_seconds=20, stop_grace_seconds=1)
        def launch(command, **kwargs):
            task_path = Path(command[-1])
            task = read_json(task_path)
            atomic_json(task_path.parent / "result.json", dict(task_sha256=digest(task),
                        result=dict(valid=True, env_steps=123, length=9)))
            self.assertTrue(kwargs["start_new_session"])
            class Process:
                pid = 901234
                def poll(self): return 0
            return Process()
        with tempfile.TemporaryDirectory() as temp, \
             patch("blackout_v11.runtime.process_birth", return_value="birth"), \
             patch("blackout_v11.runtime.owned_members", return_value=[]), \
             patch("blackout_v11.runtime.terminate_owned"), \
             patch("blackout_v11.runtime.subprocess.Popen", side_effect=launch):
            rows = run_tasks([dict(id="one", kind="collect")], temp, cfg)
        self.assertEqual(rows[0]["attempt_steps"], 123)
        self.assertTrue(rows[0]["valid"])


if __name__ == "__main__":
    unittest.main()
