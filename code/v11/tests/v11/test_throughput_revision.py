"""Queue continuation preserves the two prior approvals and exact resume boundary."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blackout_v11.io import atomic_json, digest, file_hash, read_json
from blackout_v11.revision import (
    THROUGHPUT_ARCHIVE,
    THROUGHPUT_FIX_ARCHIVE,
    THROUGHPUT_PROTECTED,
    create_evaluation_revision,
    create_observability_revision,
    create_throughput_revision,
    create_throughput_fix_revision,
    throughput_overrides,
    validated_revision,
)
from blackout_v11.runner import prepare_registration


class ThroughputRevisionTests(unittest.TestCase):
    def fixture(self, root):
        original = dict(config={"arms": ["R", "G", "F", "D"], "seeds": [11, 22, 33],
                                "confirmation_seeds": [44, 55], "training_steps": 2097152},
                        packages={"torch": "same"}, contract="same",
                        sources={f"code/v11/blackout_v11/{name}.py": "original" for name in
                                 ("policy", "training", "runner", "revision", "benchmark", "behavior", "behavior_reports")})
        for name, value in {"registration.json": original, "runtime_profile.json": {"workers": 8},
                            "task_plan.json": {"training_steps": 34340864},
                            "status.json": {"state": "stopped", "operation": "study"}}.items():
            atomic_json(root / name, value)
        folder = root / "runs/D_s33/checkpoints"
        folder.mkdir(parents=True)
        checkpoint = folder / "temporary.pt"
        checkpoint.write_bytes(b"original policy, critic, Adam state and RNG")
        checksum = file_hash(checkpoint)
        checkpoint = checkpoint.rename(folder / (checksum + ".pt"))
        atomic_json(folder / "latest.json", {"sha256": checksum, "step": 553144})
        evaluation = copy.deepcopy(original)
        evaluation["sources"]["code/v11/blackout_v11/training.py"] = "evaluation-accounting"
        with patch("blackout_v11.revision.registration", return_value=evaluation):
            create_evaluation_revision(root, original["config"])
        atomic_json(root / "observability_revisions/stop-events-v1/before_source_registration.json", evaluation)
        before = copy.deepcopy(evaluation)
        before["sources"]["code/v11/blackout_v11/runner.py"] = "logging"
        before["sources"]["code/v11/blackout_v11/revision.py"] = "logging"
        before["sources"]["code/v11/blackout_v11/stop_events.py"] = "logger"
        with patch("blackout_v11.revision.registration", return_value=before):
            create_observability_revision(root, original["config"])
        archive = root / THROUGHPUT_ARCHIVE
        atomic_json(archive / "before_source_registration.json", before)
        atomic_json(archive / "continuation_boundary.json", dict(run="D_s33", step=553144,
                    checkpoint_sha256=checksum, completed_runs=["D_s11", "D_s22"]))
        current = copy.deepcopy(before)
        for name in ("training", "runner", "revision", "benchmark", "behavior", "behavior_reports"):
            current["sources"][f"code/v11/blackout_v11/{name}.py"] = "queue16"
        return original, before, current, checkpoint

    def test_preserves_prior_receipts_registration_checkpoint_and_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, before, current, checkpoint = self.fixture(root)
            protected = {name: file_hash(root / name) for name in THROUGHPUT_PROTECTED}
            checkpoint_hash = file_hash(checkpoint)
            boundary_hash = file_hash(root / THROUGHPUT_ARCHIVE / "continuation_boundary.json")
            with patch("blackout_v11.revision.registration", return_value=current):
                receipt = create_throughput_revision(root, original["config"])
                self.assertEqual(create_throughput_revision(root, original["config"]), receipt)
                self.assertEqual(throughput_overrides(root, original["config"]), {"wave_episodes": 16})
            with patch("blackout_v11.runner.registration", return_value=current):
                self.assertEqual(prepare_registration(root, original["config"]), digest(original))
            self.assertEqual(protected, {name: file_hash(root / name) for name in THROUGHPUT_PROTECTED})
            self.assertEqual(file_hash(checkpoint), checkpoint_hash)
            self.assertEqual(file_hash(root / THROUGHPUT_ARCHIVE / checkpoint.name), checkpoint_hash)
            self.assertEqual(file_hash(root / THROUGHPUT_ARCHIVE / "continuation_boundary.json"), boundary_hash)
            self.assertTrue(receipt["training_procedure_changed"])
            self.assertTrue(receipt["training_budget_unchanged"])
            self.assertTrue(receipt["training_seeds_unchanged"])
            self.assertTrue(receipt["evaluation_schedule_unchanged"])
            self.assertNotIn("wave_episodes", read_json(root / "registration.json")["config"])
            self.assertEqual(read_json(root / "status.json")["state"], "stopped")

    def test_defaults_distinguish_fresh_and_previously_approved_studies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(throughput_overrides(root, {}), {})
            original, before, current, _ = self.fixture(root)
            with patch("blackout_v11.revision.registration", return_value=before):
                self.assertEqual(throughput_overrides(root, original["config"]), {"wave_episodes": 8})
            with patch("blackout_v11.revision.registration", return_value=current):
                with self.assertRaises(ValueError):
                    throughput_overrides(root, original["config"])

    def test_config_package_contract_and_non_allowed_source_changes_rejected(self):
        for kind in ("config", "packages", "contract", "policy", "runtime"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, _, current, _ = self.fixture(root)
                if kind == "config":
                    current["config"]["wave_episodes"] = 16
                elif kind == "packages":
                    current["packages"]["torch"] = "different"
                elif kind == "contract":
                    current["contract"] = "different"
                else:
                    current["sources"][f"code/v11/blackout_v11/{kind}.py"] = "different"
                with patch("blackout_v11.revision.registration", return_value=current):
                    with self.assertRaises(ValueError):
                        create_throughput_revision(root, original["config"])
                self.assertFalse((root / "throughput_revision.json").exists())

    def test_requires_inactive_stopped_study(self):
        for kind in ("active", "running", "benchmark"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, _, current, _ = self.fixture(root)
                if kind != "active":
                    atomic_json(root / "status.json", dict(state="running" if kind == "running" else "stopped",
                                operation="benchmark" if kind == "benchmark" else "study"))
                with patch("blackout_v11.revision.registration", return_value=current), \
                     patch("blackout_v11.runner.is_active", return_value=kind == "active"):
                    with self.assertRaises(ValueError):
                        create_throughput_revision(root, original["config"])
                self.assertFalse((root / "throughput_revision.json").exists())

    def test_requires_exact_captured_resume_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, _, current, checkpoint = self.fixture(root)
            atomic_json(checkpoint.parent / "latest.json", dict(sha256=checkpoint.stem, step=553145))
            with patch("blackout_v11.revision.registration", return_value=current):
                with self.assertRaisesRegex(ValueError, "checkpoint moved"):
                    create_throughput_revision(root, original["config"])
            self.assertFalse((root / "throughput_revision.json").exists())

    def test_rejects_later_source_override_receipt_and_archive_tampering(self):
        kinds = ("source", "override", "extra_override", "float_override", "evaluation_receipt", "logging_receipt",
                 "archived_receipt", "archive_checkpoint", "boundary")
        for kind in kinds:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, _, current, checkpoint = self.fixture(root)
                with patch("blackout_v11.revision.registration", return_value=current):
                    receipt = create_throughput_revision(root, original["config"])
                archive = root / THROUGHPUT_ARCHIVE
                if kind == "source":
                    current["sources"]["code/v11/blackout_v11/training.py"] = "later"
                elif "override" in kind:
                    receipt["runtime_overrides"] = ({"wave_episodes": 8} if kind == "override" else
                                                   {"wave_episodes": 16.0} if kind == "float_override" else
                                                   {"wave_episodes": 16, "workers": 16})
                    atomic_json(root / "throughput_revision.json", receipt)
                elif kind in ("evaluation_receipt", "logging_receipt"):
                    name = "evaluation_revision.json" if kind == "evaluation_receipt" else "observability_revision.json"
                    value = read_json(root / name)
                    value["created_at"] += 1
                    atomic_json(root / name, value)
                elif kind == "archived_receipt":
                    atomic_json(archive / "observability_revision.json", {"changed": True})
                elif kind == "archive_checkpoint":
                    (archive / checkpoint.name).write_bytes(b"changed")
                else:
                    value = read_json(archive / "continuation_boundary.json")
                    value["completed_runs"].append("changed")
                    atomic_json(archive / "continuation_boundary.json", value)
                with self.assertRaises(ValueError):
                    validated_revision(root, original["config"], current)

    def test_live_checkpoint_progress_does_not_move_archived_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, _, current, checkpoint = self.fixture(root)
            with patch("blackout_v11.revision.registration", return_value=current):
                receipt = create_throughput_revision(root, original["config"])
            atomic_json(checkpoint.parent / "latest.json", dict(sha256="later-commit", step=600000))
            checkpoint.unlink()
            validated_revision(root, original["config"], current)
            self.assertEqual(receipt["checkpoint_boundary"]["step"], 553144)

    def fix_fixture(self, root):
        original, _, before, checkpoint = self.fixture(root)
        with patch("blackout_v11.revision.registration", return_value=before):
            create_throughput_revision(root, original["config"])
        archive = root / THROUGHPUT_FIX_ARCHIVE
        atomic_json(archive / "before_source_registration.json", before)
        (archive / "throughput_revision.json").write_bytes((root / "throughput_revision.json").read_bytes())
        current = copy.deepcopy(before)
        current["sources"]["code/v11/blackout_v11/evaluation.py"] = "exclude-only-wave-episodes"
        current["sources"]["code/v11/blackout_v11/revision.py"] = "pin-evaluation-config-fix"
        return original, current, checkpoint

    def test_fix_preserves_prior_approvals_checkpoint_and_queue_override(self):
        for state in ("failed", "stopped"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, current, checkpoint = self.fix_fixture(root)
                atomic_json(root / "status.json", dict(state=state, operation="study"))
                protected = {name: file_hash(root / name) for name in (*THROUGHPUT_PROTECTED, "throughput_revision.json")}
                checkpoint_hash = file_hash(checkpoint)
                with patch("blackout_v11.revision.registration", return_value=current):
                    receipt = create_throughput_fix_revision(root, original["config"])
                    self.assertEqual(create_throughput_fix_revision(root, original["config"]), receipt)
                    self.assertEqual(throughput_overrides(root, original["config"]), {"wave_episodes": 16})
                with patch("blackout_v11.runner.registration", return_value=current):
                    self.assertEqual(prepare_registration(root, original["config"]), digest(original))
                self.assertEqual(protected, {name: file_hash(root / name) for name in protected})
                self.assertEqual(file_hash(checkpoint), checkpoint_hash)
                self.assertEqual(receipt["evaluation_config_ignored_keys"], ["wave_episodes"])
                self.assertEqual(read_json(root / "status.json")["state"], state)

    def test_fix_rejects_config_packages_contract_and_other_source_changes(self):
        for kind in ("config", "packages", "contract", "evaluation_schedule", "training", "study"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, current, _ = self.fix_fixture(root)
                if kind == "config":
                    current["config"]["dev_replicates"] = 100
                elif kind == "packages":
                    current["packages"]["torch"] = "changed"
                elif kind == "contract":
                    current["contract"] = "changed"
                else:
                    current["sources"][f"code/v11/blackout_v11/{kind}.py"] = "changed"
                with patch("blackout_v11.revision.registration", return_value=current):
                    with self.assertRaises(ValueError):
                        create_throughput_fix_revision(root, original["config"])
                self.assertFalse((root / "throughput_fix_revision.json").exists())

    def test_fix_requires_inactive_failed_or_stopped_study(self):
        for kind in ("active", "running", "complete", "benchmark"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, current, _ = self.fix_fixture(root)
                atomic_json(root / "status.json", dict(state=kind if kind in ("running", "complete") else "failed",
                            operation="benchmark" if kind == "benchmark" else "study"))
                with patch("blackout_v11.revision.registration", return_value=current), \
                     patch("blackout_v11.runner.is_active", return_value=kind == "active"):
                    with self.assertRaises(ValueError):
                        create_throughput_fix_revision(root, original["config"])
                self.assertFalse((root / "throughput_fix_revision.json").exists())

    def test_fix_rejects_later_evaluator_previous_receipt_or_scope_changes(self):
        for kind in ("source", "previous_receipt", "archived_receipt", "ignored_keys"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original, current, _ = self.fix_fixture(root)
                with patch("blackout_v11.revision.registration", return_value=current):
                    receipt = create_throughput_fix_revision(root, original["config"])
                if kind == "source":
                    current["sources"]["code/v11/blackout_v11/evaluation.py"] = "later-evaluation-change"
                elif kind == "ignored_keys":
                    receipt["evaluation_config_ignored_keys"].append("dev_replicates")
                    atomic_json(root / "throughput_fix_revision.json", receipt)
                else:
                    path = root / (THROUGHPUT_FIX_ARCHIVE if kind == "archived_receipt" else "") / "throughput_revision.json"
                    value = read_json(path)
                    value["created_at"] += 1
                    atomic_json(path, value)
                with self.assertRaises(ValueError):
                    validated_revision(root, original["config"], current)


if __name__ == "__main__":
    unittest.main()
