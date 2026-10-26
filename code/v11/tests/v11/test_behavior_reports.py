import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from agent_league.trace import Trace
from blackout_v11.behavior_reports import aggregate, enforce_retention, inspect_episode, write_report
from blackout_v11.io import atomic_json, read_json


def summary(steps=10):
    return dict(schema="blackout.v11.behavior.v1", steps=steps, complete=True, error=None,
        metadata=dict(arm="F", side=0, opponent_id="v6", kind="collect", task_source="study/pilot/F_s11/task.json"),
        league=dict(sides=[], timeline=[], event_truth="observation proxies"),
        situations=dict(side0={"holding:carrying": dict(unit_steps=10, stop=2, turn=1, command_move=8,
            movement_observed_steps=4, observed_move=3, pickup_proxy=1, delivery_proxy=1, respawn_proxy=0)}, side1={}))


class BehaviorReportTests(unittest.TestCase):
    def test_retention_preserves_active_traces_summaries_and_other_studies(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp) / "study"
            root = study / "behavior/episodes"
            for name, stamp, closed in (("old", 1, True), ("new", 2, True), ("active", 3, False)):
                episode = root / name
                trace = episode / "trace"
                trace.mkdir(parents=True)
                (trace / "payload").write_bytes(b"a" * 1024)
                atomic_json(episode / "summary.json", summary())
                if closed:
                    atomic_json(trace / "index.json", dict(schema="blackout.league.trace.v2", complete=name != "old"))
                    os.utime(trace / "index.json", (stamp, stamp))
            outside = Path(temp) / "other_study"
            (outside / "trace").mkdir(parents=True)
            (outside / "trace/payload").write_bytes(b"keep")
            (root / "escape").symlink_to(outside, target_is_directory=True)
            newer_size = sum(p.stat().st_size for p in (root / "new/trace").iterdir())
            report = enforce_retention(study, newer_size + 1024)
            self.assertEqual(report["pruned"], ["old"])
            self.assertTrue(read_json(root / "old/summary.json")["trace_pruned"])
            self.assertEqual(read_json(root / "old/summary.json")["steps"], 10)
            self.assertTrue((root / "new/trace/index.json").exists())
            self.assertTrue((root / "active/trace/payload").exists())
            self.assertEqual((outside / "trace/payload").read_bytes(), b"keep")
            again = enforce_retention(study, 0)
            self.assertTrue(again["over_budget"])
            self.assertEqual(again["after_bytes"], 1024)
            self.assertTrue((root / "active/trace/payload").exists())

    def test_rates_use_observed_denominator_and_disclose_missing_reused_history(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)
            atomic_json(study / "behavior/episodes/abc/summary.json", summary())
            probe = summary(1000)
            probe["metadata"]["kind"] = "probe"
            probe["situations"]["side0"]["holding:carrying"]["stop"] = 10
            atomic_json(study / "behavior/episodes/probe/summary.json", probe)
            atomic_json(study / "continuation_import.json", dict(total_reused_games=600, behavior_history_available=False))
            report = aggregate(study)
            row = report["controlled_team_situations"]["holding:carrying"]
            self.assertEqual(row["rates"]["stop"], .2)
            self.assertEqual(row["rates"]["observed_move"], .75)
            self.assertEqual(row["events_per_1000_observed_unit_steps"]["delivery_proxy"], 250)
            self.assertEqual(report["coverage"]["reused_episodes_without_behavior"], 600)
            self.assertEqual(report["coverage"]["recorded_episodes"], 1)
            self.assertEqual(report["coverage"]["calibration_probes_excluded"], 1)
            self.assertEqual(report["coverage"]["recorded_env_steps"], 10)
            self.assertEqual(report["groups"]["stage"]["pilot"]["episodes"], 1)
            paths = write_report(study)
            self.assertIn("600", Path(paths["markdown"]).read_text())
            self.assertTrue(Path(paths["json"]).is_file())

    def test_step_inspection_decodes_shared_xor_chunk_and_reports_pruning(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)
            episode = study / "behavior/episodes/abcdef1234"
            trace = Trace(episode / "trace", dict(chunk_steps=2))
            before = np.arange(960, dtype=np.float32).reshape(10, 96) / 1000
            after = before + np.float32(.001)
            actions = np.zeros((10, 2), np.float32)
            actions[0, 0] = 1
            trace.append(step=0, before=before, after=after, actions=actions, rewards=np.zeros(10))
            trace.close(complete=True)
            atomic_json(episode / "summary.json", summary(1))
            found = inspect_episode(study, "abcdef", 0)
            np.testing.assert_array_equal(np.asarray(found["transition"]["before"], np.float32), before)
            np.testing.assert_array_equal(np.asarray(found["transition"]["after"], np.float32), after)
            self.assertEqual(found["transition"]["actions"][0], [1., 0.])
            with self.assertRaisesRegex(ValueError, "not retained"):
                inspect_episode(study, "abcdef", 5)
            enforce_retention(study, 0)
            with self.assertRaisesRegex(ValueError, "pruned"):
                inspect_episode(study, "abcdef", 0)
            self.assertEqual(inspect_episode(study, "abcdef")["steps"], 1)


if __name__ == "__main__":
    unittest.main()
