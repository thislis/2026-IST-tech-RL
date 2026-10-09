"""BASE-R13/R15 experiment registry and evaluation matrix tests."""

from __future__ import annotations

from project_paths import project_root, project_path


import copy
from datetime import datetime, timezone
import json
import tempfile
import unittest
from pathlib import Path

import torch

from blackout_rl import (
    ExperimentIdentity,
    SubmissionPolicy,
    checkpoint_payload,
    load_checkpoint,
    read_registry,
    register_checkpoint,
)
from blackout_rl.logging_schema import SERIES_SCHEMA_VERSION, validate_series_log
from eval.evaluator import summarize_episodes
from eval.policy_matrix import build_policy_matrix, validate_policy_matrix


ROOT = project_root()
RANDOM_SERIES = project_path('logs/pre_v1/prep14_random_paired_5seeds.json', root=ROOT)
REGISTRY = project_path('logs/shared/experiments/registry.jsonl', root=ROOT)
MATRIX = project_path('logs/pre_v1/base_r15_policy_matrix_seed1401.json', root=ROOT)


class ExperimentRegistryTests(unittest.TestCase):
    def test_registered_checkpoint_binds_config_seed_git_and_opponent(self) -> None:
        model = SubmissionPolicy(
            hidden_dim=16, entity_dim=4, context_dim=4, graphic_dim=8,
            slot_embedding_dim=4,
        )
        identity = ExperimentIdentity(
            run_id="registry-test-run",
            config={"algorithm": "IPPO", "learning_rate": 3e-4},
            seed=1313,
            git_sha="a" * 40,
            opponent_id="scripted-battery-v1",
        )
        payload = checkpoint_payload(
            model,
            global_step=500,
            training_seed=1313,
            experiment=identity.checkpoint_metadata(),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            registry = Path(directory) / "registry.jsonl"
            entry = register_checkpoint(checkpoint, payload, registry_path=registry)
            restored_model, restored = load_checkpoint(checkpoint)
            entries = read_registry(registry)
            action = restored_model(
                torch.zeros(5, 96), torch.zeros(5, 11, 96, 96)
            )
        self.assertEqual(tuple(action.shape), (5, 2))
        self.assertEqual(restored["experiment"], identity.checkpoint_metadata())
        self.assertEqual(entries, [entry])
        self.assertEqual(entry["seed"], 1313)
        self.assertEqual(entry["git_sha"], "a" * 40)
        self.assertEqual(entry["opponent_id"], "scripted-battery-v1")
        self.assertEqual(entry["config"], identity.config)

    def test_registry_rejects_duplicate_run_id(self) -> None:
        model = SubmissionPolicy(hidden_dim=16, entity_dim=4, context_dim=4, graphic_dim=8)
        identity = ExperimentIdentity(
            run_id="duplicate", config={}, seed=1, git_sha="b" * 40,
            opponent_id="fixed",
        )
        payload = checkpoint_payload(
            model, global_step=0, training_seed=1,
            experiment=identity.checkpoint_metadata(),
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.jsonl"
            register_checkpoint(Path(directory) / "one.pt", payload, registry_path=registry)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                register_checkpoint(Path(directory) / "two.pt", payload, registry_path=registry)


@unittest.skipUnless(REGISTRY.exists(), "run scripts/register_base_r13_checkpoint.py first")
class RegisteredCheckpointEvidenceTests(unittest.TestCase):
    def test_registered_checkpoint_exists_and_embeds_same_identity(self) -> None:
        entries = read_registry(REGISTRY)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        checkpoint = project_path(entry["checkpoint_path"], root=ROOT)
        self.assertTrue(checkpoint.is_file())
        _, payload = load_checkpoint(checkpoint)
        experiment = payload["experiment"]
        for field in ("run_id", "seed", "git_sha", "opponent_id", "config"):
            self.assertEqual(experiment[field], entry[field])


@unittest.skipUnless(RANDOM_SERIES.exists(), "PREP-14 paired series is required")
class PolicyMatrixTests(unittest.TestCase):
    def _series(self, policy: str) -> dict:
        source = json.loads(RANDOM_SERIES.read_text())
        episodes = copy.deepcopy(
            [episode for episode in source["episodes"] if episode["seed"] == 1401]
        )
        for index, episode in enumerate(episodes):
            episode["episode_id"] = f"{policy}-episode-{index}"
            episode["pair_id"] = f"{policy}-seed-1401"
            episode["model"]["policy_id"] = f"{policy}-v1"
        series = {
            "schema_version": SERIES_SCHEMA_VERSION,
            "series_id": f"{policy}-series",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "series_config": {
                "seeds": [1401], "side_swap": True,
                "model_policy_seed_base": 1, "opponent_policy_seed_base": 2,
            },
            "episodes": episodes,
            "summary": summarize_episodes(episodes),
        }
        validate_series_log(series)
        return series

    def test_matrix_requires_common_seed_opponent_and_side_swap(self) -> None:
        matrix = build_policy_matrix(
            {policy: self._series(policy) for policy in ("random", "scripted", "ippo")},
            seeds=[1401],
        )
        validate_policy_matrix(matrix)
        self.assertEqual(set(matrix["results"]), {"random", "scripted", "ippo"})
        self.assertEqual(matrix["matrix_config"]["seeds"], [1401])
        for result in matrix["results"].values():
            self.assertEqual(result["summary"]["episodes"], 2)
            self.assertTrue(
                result["summary"]["side_bias_diagnostics"][
                    "evaluator_side_attribution_passed"
                ]
            )


@unittest.skipUnless(MATRIX.exists(), "run eval/policy_matrix.py first")
class PolicyMatrixEvidenceTests(unittest.TestCase):
    def test_recorded_matrix_has_common_seed_opponent_and_side_swap(self) -> None:
        matrix = json.loads(MATRIX.read_text())
        validate_policy_matrix(matrix)
        config = matrix["matrix_config"]
        self.assertEqual(config["seeds"], [1401])
        self.assertEqual(config["opponent_policy_seeds"], {"1401": 515403})
        self.assertEqual(config["opponent_id"], "random-v1")
        self.assertEqual(matrix["ranking"], ["scripted", "random", "ippo"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
