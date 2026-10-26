"""Report generation never launches experiments or changes their outcome."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blackout_v11.io import atomic_json, read_json
from blackout_v11.runner import finish_behavior_report, main


class ReportCommandTests(unittest.TestCase):
    def test_report_only_bypasses_experiment_registration_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("sys.argv", ["v11", "--name", "example", "--report"]), \
                 patch("blackout_v11.runner.directory_for", return_value=root), \
                 patch("blackout_v11.runner.prepare_registration", side_effect=AssertionError("registration")), \
                 patch("blackout_v11.runner.subprocess.Popen", side_effect=AssertionError("launch")), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((root / "behavior_report.md").is_file())
            self.assertEqual(read_json(root / "behavior_report.json")["coverage"]["recorded_episodes"], 0)
            self.assertFalse((root / "status.json").exists())

    def test_automatic_report_failure_preserves_experiment_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome = dict(state="failed", operation="study", error="original study failure")
            atomic_json(root / "status.json", outcome)
            with patch("blackout_v11.behavior_reports.write_report", side_effect=OSError("disk full")), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = finish_behavior_report(root)
            self.assertEqual(report["state"], "failed")
            self.assertEqual(read_json(root / "status.json"), outcome)
            self.assertEqual(read_json(root / "behavior_report_status.json")["error"], "disk full")


if __name__ == "__main__":
    unittest.main()
