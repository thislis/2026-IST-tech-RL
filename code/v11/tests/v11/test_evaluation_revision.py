"""Evaluation changes preserve exact checkpoint and training provenance."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from blackout_v11.io import atomic_json, digest, file_hash, read_json
from blackout_v11.revision import create_evaluation_revision, validated_revision
from blackout_v11.runner import prepare_registration


class EvaluationRevisionTests(unittest.TestCase):
    def fixture(self, root):
        original = dict(config={"training_steps": 2097152, "arms": ["R", "G", "F", "D"], "seeds": [11, 22, 33],
                                "confirmation_seeds": [44, 55]}, packages={"torch": "same"},
                        sources={"code/v11/blackout_v11/policy.py": "policy",
                                 "code/v11/blackout_v11/study.py": "old"})
        current = copy.deepcopy(original)
        current["sources"]["code/v11/blackout_v11/study.py"] = "new"
        for name, value in {"registration.json": original, "runtime_profile.json": {"measured": True},
                            "task_plan.json": {"total_learning_steps": 34340864},
                            "status.json": {"state": "stopped", "operation": "study"}}.items():
            atomic_json(root / name, value)
        folder = root / "runs/R_s11/checkpoints"
        folder.mkdir(parents=True)
        file = folder / "temporary.pt"
        file.write_bytes(b"unchanged weights, critic, optimizers and RNG")
        checksum = file_hash(file)
        file.rename(folder / (checksum + ".pt"))
        atomic_json(folder / "latest.json", {"sha256": checksum, "step": 1822457})
        return original, current, folder / (checksum + ".pt")

    def test_revision_keeps_original_identity_and_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, current, checkpoint = self.fixture(root)
            original_hash, checkpoint_hash = file_hash(root / "registration.json"), file_hash(checkpoint)
            with patch("blackout_v11.revision.registration", return_value=current):
                receipt = create_evaluation_revision(root, original["config"])
                self.assertEqual(create_evaluation_revision(root, original["config"]), receipt)
            with patch("blackout_v11.runner.registration", return_value=current):
                self.assertEqual(prepare_registration(root, original["config"]), digest(original))
            self.assertEqual(file_hash(root / "registration.json"), original_hash)
            self.assertEqual(file_hash(checkpoint), checkpoint_hash)
            archived = Path(receipt["archive"]) / checkpoint.relative_to(root)
            self.assertEqual(file_hash(archived), checkpoint_hash)
            self.assertEqual(read_json(root / "status.json")["state"], "stopped")

    def test_training_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, current, _ = self.fixture(root)
            for kind in ("config", "actor"):
                invalid = copy.deepcopy(current)
                if kind == "config":
                    invalid["config"]["training_steps"] = 100
                else:
                    invalid["sources"]["code/v11/blackout_v11/policy.py"] = "changed"
                with patch("blackout_v11.revision.registration", return_value=invalid):
                    with self.assertRaises(ValueError):
                        create_evaluation_revision(root, original["config"])
            self.assertFalse((root / "evaluation_revision.json").exists())

    def test_revision_rejects_subsequent_code_and_runtime_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, current, _ = self.fixture(root)
            with patch("blackout_v11.revision.registration", return_value=current):
                create_evaluation_revision(root, original["config"])
            changed = copy.deepcopy(current)
            changed["sources"]["code/v11/blackout_v11/study.py"] = "changed again"
            with self.assertRaises(ValueError):
                validated_revision(root, original["config"], changed)
            atomic_json(root / "runtime_profile.json", {"measured": False})
            with self.assertRaises(ValueError):
                validated_revision(root, original["config"], current)


if __name__ == "__main__":
    unittest.main()
