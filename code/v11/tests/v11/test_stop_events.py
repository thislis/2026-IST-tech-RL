"""Stop provenance tests: temporary files and mocked supervisor, no Unity."""
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from blackout_v11.io import read_json
from blackout_v11.stop_events import StopRecorder, append_jsonl, manual_stop, read_events


LIMITS = dict(memory_soft_gib=32, disk_stop_gib=25)


def resource(rss=2.):
    return dict(process_rss_gib=rss, mps_memory={"available": False, "driver_bytes": 0},
                conservative_unified_gib=rss, memory_pressure={"critical": False},
                disk_free_bytes=100 * 1024**3, active_workers=0, active=[],
                phase="train", run="R_s11", step=123, limits=LIMITS)


class StopEventsTests(unittest.TestCase):
    def test_concurrent_append_keeps_every_record_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stop_events.jsonl"
            with ThreadPoolExecutor(max_workers=2) as writers:
                list(writers.map(lambda n: append_jsonl(path, {"n": n}), range(12)))
            self.assertEqual(sorted(row["n"] for row in read_events(temp)), list(range(12)))

    def test_append_recovers_partial_tail_and_fsyncs_each_event(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stop_events.jsonl"
            path.write_bytes(b'{"interrupted":')
            with patch("blackout_v11.stop_events.os.fsync", wraps=os.fsync) as sync:
                append_jsonl(path, {"event": "one"})
                append_jsonl(path, {"event": "two"})
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(read_events(temp), [{"event": "one"}, {"event": "two"}])
            self.assertTrue(path.read_bytes().startswith(b'{"interrupted":\n'))

    def test_cli_persists_request_before_touch_and_supervisor_identifies_it(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = StopRecorder(temp, LIMITS, "one", "study")
            original = append_jsonl
            def append_before_touch(path, row):
                self.assertFalse((Path(temp) / "stop.request").exists())
                original(path, row)
            with patch("blackout_v11.stop_events.append_jsonl", side_effect=append_before_touch):
                requested = manual_stop(temp, resource=resource(), limits=LIMITS,
                                        session_id="one", operation="study")
            recorder.observe(resource())
            rows = read_events(temp)
            self.assertEqual([r["event"] for r in rows], ["stop_requested", "stop_observed"])
            self.assertEqual(rows[0]["requester_pid"], os.getpid())
            self.assertEqual(rows[1]["request_event_id"], requested["event_id"])
            self.assertEqual(recorder.terminal_fields()["stop_origin"], "manual_cli")

    def test_new_session_keeps_history_without_attributing_old_cli_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            manual_stop(temp, resource=resource(), limits=LIMITS, session_id="old")
            (Path(temp) / "stop.request").unlink()
            before = (Path(temp) / "stop_events.jsonl").read_bytes()
            resumed = StopRecorder(temp, LIMITS, "new", "study")
            self.assertIsNone(resumed.current_stop)
            (Path(temp) / "stop.request").touch()
            resumed.observe(resource())
            self.assertEqual(resumed.current_stop["origin"], "external_stopfile")
            self.assertEqual(resumed.current_stop["session_id"], "new")
            self.assertTrue((Path(temp) / "stop_events.jsonl").read_bytes().startswith(before))

    def test_signal_is_lightweight_and_flushes_at_terminal_without_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = StopRecorder(temp, LIMITS, "signal", "study")
            recorder.signal_handler(signal.SIGTERM, None)
            self.assertTrue(recorder.stop_path.exists())
            self.assertFalse((Path(temp) / "stop_events.jsonl").exists())
            fields = recorder.finish("stopped", resource())
            rows = read_events(temp)
            self.assertEqual([r["event"] for r in rows], ["stop_requested", "stop_completed"])
            self.assertEqual(rows[0]["signum"], signal.SIGTERM)
            self.assertEqual(rows[0]["signal_name"], "SIGTERM")
            self.assertEqual(fields["stop_reason"], "received SIGTERM")
            self.assertEqual(fields["stop_resource"]["step"], 123)

    def test_resource_guard_keeps_trigger_snapshot_after_cleanup_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = StopRecorder(temp, LIMITS, "guard", "study")
            recorder.guard("unified memory soft limit reached", resource(33))
            recorder.guard("unified memory soft limit reached", resource(34))
            fields = recorder.finish("stopped", resource(1))
            self.assertEqual(fields["stop_resource"]["process_rss_gib"], 33)
            self.assertEqual(len(read_events(temp)), 2)
            self.assertEqual(read_events(temp)[-1]["resource"]["process_rss_gib"], 1)

    def test_resource_history_is_rate_limited_and_append_only_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = StopRecorder(temp, LIMITS, "one", "study")
            self.assertTrue(recorder.sample(resource(), now=0))
            self.assertFalse(recorder.sample(resource(), now=29))
            self.assertTrue(recorder.sample(resource(), now=30))
            resumed = StopRecorder(temp, LIMITS, "two", "study")
            self.assertTrue(resumed.sample(resource(), now=0))
            rows = [json.loads(line) for line in (Path(temp) / "resources.jsonl").read_text().splitlines()]
            self.assertEqual([r["session_id"] for r in rows], ["one", "one", "two"])
            self.assertEqual(rows[0]["resource"]["run"], "R_s11")

    def test_failure_preserves_prior_stop_cause_and_records_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = StopRecorder(temp, LIMITS, "one", "study")
            recorder.signal_handler(signal.SIGINT, None)
            recorder.failure(RuntimeError("cleanup failed"), resource(), stage="cleanup")
            fields = recorder.finish("failed", resource())
            self.assertEqual(fields["stop_reason"], "received SIGINT")
            self.assertEqual(fields["last_stop"]["reason"], "cleanup failed")
            self.assertEqual(read_events(temp)[1]["exception_type"], "RuntimeError")

    def test_supervisor_resource_stop_preserves_reason_in_terminal_status(self):
        from blackout_v11.runner import supervise
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = SimpleNamespace(benchmark=False, smoke=False, start_token="runner",
                                   supervisor_fd=os.open(directory / "lock", os.O_CREAT | os.O_RDWR, 0o600),
                                   machine_fd=os.open(directory / "machine", os.O_CREAT | os.O_RDWR, 0o600))
            with patch("blackout_v11.runner.signal.signal"), \
                 patch("torch.set_num_threads"), patch("torch.set_num_interop_threads"), \
                 patch("blackout_v11.runner.shutil.which", return_value=None), \
                 patch("blackout_v11.runner.clean_orphans"), \
                 patch("blackout_v11.runner.process_birth", return_value="birth"), \
                 patch("blackout_v11.runner.prepare_registration", return_value="registered"), \
                 patch("blackout_v11.runner.check", return_value={}), \
                 patch("blackout_v11.runner.finish_behavior_report"), \
                 patch("blackout_v11.runner.resource_snapshot", side_effect=[resource(33), resource(33), resource(1), resource(1)]):
                supervise(args, dict(LIMITS), directory)
            final = read_json(directory / "status.json")
            self.assertEqual(final["state"], "stopped")
            self.assertEqual(final["stop_reason"], "unified memory soft limit reached")
            self.assertEqual(final["stop_resource"]["process_rss_gib"], 33)
            self.assertEqual(final["resource"]["process_rss_gib"], 1)
            self.assertEqual(read_events(temp)[-1]["event"], "stop_completed")


if __name__ == "__main__":
    unittest.main()
