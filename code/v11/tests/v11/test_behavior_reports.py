import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from agent_league.trace import Trace
from blackout_v11 import behavior_reports
from blackout_v11.behavior_reports import aggregate, enforce_retention, inspect_episode, notify_retention_episode, write_report
from blackout_v11.io import atomic_json, read_json


def summary(steps=10):
    return dict(schema="blackout.v11.behavior.v1", steps=steps, complete=True, error=None,
        metadata=dict(arm="F", side=0, opponent_id="v6", kind="collect", task_source="study/pilot/F_s11/task.json"),
        league=dict(sides=[], timeline=[], event_truth="observation proxies"),
        situations=dict(side0={"holding:carrying": dict(unit_steps=10, stop=2, turn=1, command_move=8,
            movement_observed_steps=4, observed_move=3, pickup_proxy=1, delivery_proxy=1, respawn_proxy=0)}, side1={}))


class BehaviorReportTests(unittest.TestCase):
    def test_incremental_retention_does_not_rescan_closed_history(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)
            root = study / "behavior/episodes"
            def closed(name):
                episode = root / name
                (episode / "trace").mkdir(parents=True)
                (episode / "trace/payload").write_bytes(b"a" * 1024)
                atomic_json(episode / "summary.json", summary())
                atomic_json(episode / "trace/index.json", dict(schema="blackout.league.trace.v2"))
                return episode
            old = closed("old")
            first = enforce_retention(study)
            self.assertEqual(first["mode"], "reconcile")
            self.assertEqual(first["traces_measured"], 1)
            new = closed("new")
            notify_retention_episode(study, new)
            with patch("blackout_v11.behavior_reports._bytes", wraps=behavior_reports._bytes) as measure:
                second = enforce_retention(study)
                self.assertEqual([call.args[0] for call in measure.call_args_list], [(new / "trace").resolve()])
            with patch("blackout_v11.behavior_reports._bytes", side_effect=AssertionError("closed trace rescanned")), \
                 patch.object(Path, "iterdir", side_effect=AssertionError("episode history enumerated")):
                third = enforce_retention(study)
            self.assertEqual(second["mode"], "incremental")
            self.assertEqual(third["before_bytes"], second["after_bytes"])
            self.assertTrue((old / "trace/index.json").exists())

    def test_writer_notices_keep_active_trace_safe_until_close(self):
        from blackout_v11.behavior import SituationRecorder
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)
            behavior = study / "behavior"
            episode = behavior / "episodes/active"
            cfg = dict(behavior_output_root=str(behavior))
            recorder = SituationRecorder(episode, cfg)
            self.assertTrue((behavior / ".retention_pending/active.json").exists())
            active = enforce_retention(study, 0)
            self.assertTrue(active["over_budget"])
            self.assertTrue((episode / "trace/events.jsonl.gz").exists())
            recorder.close(complete=False)
            self.assertTrue((behavior / ".retention_pending/active.json").exists())
            closed = enforce_retention(study, 0)
            self.assertEqual(closed["pruned"], ["active"])
            self.assertFalse(closed["over_budget"])
            self.assertTrue(read_json(episode / "summary.json")["trace_pruned"])

    def test_restart_recovers_unacknowledged_close_and_partial_prune(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)
            episode = study / "behavior/episodes/interrupted"
            trace = episode / "trace"
            trace.mkdir(parents=True)
            (trace / "payload").write_bytes(b"a" * 1024)
            atomic_json(episode / "summary.json", summary())
            atomic_json(trace / "index.json", dict(schema="blackout.league.trace.v2"))
            enforce_retention(study)
            def partial_delete(path):
                (path / "index.json").unlink()
                raise OSError("interrupted deletion")
            with patch("blackout_v11.behavior_reports.shutil.rmtree", side_effect=partial_delete):
                failed = enforce_retention(study, 0)
            self.assertTrue(failed["errors"])
            self.assertEqual(read_json(episode / "summary.json")["retention"]["state"], "failed")
            # Simulate a fresh supervisor and a legacy close without a notice.
            behavior_reports._RETENTION_READY.discard(((study / "behavior/episodes").resolve(), os.getpid()))
            missed = study / "behavior/episodes/missed"
            (missed / "trace").mkdir(parents=True)
            (missed / "trace/payload").write_bytes(b"keep")
            atomic_json(missed / "summary.json", summary())
            atomic_json(missed / "trace/index.json", dict(schema="blackout.league.trace.v2"))
            resumed = enforce_retention(study)
            self.assertEqual(resumed["mode"], "reconcile")
            self.assertIn("interrupted", resumed["pruned"])
            self.assertTrue(read_json(episode / "summary.json")["trace_pruned"])
            self.assertEqual(resumed["indexed_traces"], 1)
            self.assertTrue((missed / "trace/payload").exists())

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
