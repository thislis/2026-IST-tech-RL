"""PREP-04~07 regression tests for parsers and recorded live-Unity evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from blackout_env import load_checkpoint as load_upstream_checkpoint

from blackout_rl import (
    GRAPHIC_CHANNEL_NAMES,
    CanonicalTeamModel,
    ReferenceActorCritic,
    SubmissionPolicy,
    VECTOR_SIZE,
    canonical_agents,
    categorical_action,
    checkpoint_payload,
    hwc_to_chw,
    load_checkpoint,
    parse_graphic,
    parse_vector,
    stack_observations,
    save_checkpoint,
    team_agents,
    team_model_input,
)
from blackout_rl.logging_schema import (
    EPISODE_SCHEMA_VERSION,
    model_result,
    validate_episode_log,
    validate_series_log,
)
from blackout_rl.reward import ScoreDeltaRewardTracker
from eval.evaluator import summarize_episodes


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "logs" / "prep04_07_contract.json"
PREP14_PATH = ROOT / "logs" / "prep14_random_paired_5seeds.json"


def synthetic_vector() -> np.ndarray:
    vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
    for index in range(10):
        start = index * 9
        vector[start : start + 2] = (index / 10.0, (9 - index) / 10.0)
        vector[start + 2] = 1.0 if index < 5 else -1.0
        vector[start + 3 + (index % 6)] = 1.0
    vector[90 + 1] = 1.0
    vector[93:] = (0.25, 0.5, 0.75)
    return vector


def synthetic_graphic() -> np.ndarray:
    ids = np.arange(16, dtype=np.int64).reshape(4, 4) % len(GRAPHIC_CHANNEL_NAMES)
    return np.eye(len(GRAPHIC_CHANNEL_NAMES), dtype=np.float32)[ids]


class ObservationParserTests(unittest.TestCase):
    def test_observation_shape_and_dtype(self) -> None:
        parsed = parse_vector(synthetic_vector())
        graphic = parse_graphic(synthetic_graphic())
        self.assertEqual(parsed.positions.shape, (10, 2))
        self.assertEqual(parsed.holding_item_one_hot.shape, (10, 6))
        self.assertEqual(parsed.self_class_id, 1)
        self.assertEqual((parsed.own_score, parsed.opponent_score, parsed.time_left), (0.25, 0.5, 0.75))
        self.assertEqual(graphic.channels.shape, (4, 4, 11))
        self.assertEqual(graphic.channels.dtype, np.float32)

    def test_model_input_layout_hwc_to_chw(self) -> None:
        graphic = synthetic_graphic()
        chw = hwc_to_chw(graphic)
        self.assertEqual(chw.shape, (11, 4, 4))
        self.assertTrue(np.array_equal(chw[3], graphic[..., 3]))
        self.assertTrue(chw.flags.c_contiguous)

    def test_field_ranges_and_team_perspective(self) -> None:
        parsed = parse_vector(synthetic_vector())
        self.assertTrue(np.array_equal(parsed.team_signs[:5], np.ones(5, dtype=np.float32)))
        self.assertTrue(np.array_equal(parsed.team_signs[5:], -np.ones(5, dtype=np.float32)))
        self.assertTrue(np.array_equal(parsed.holding_item_ids, np.arange(10) % 6))

    def test_canonical_batch_does_not_use_dict_order(self) -> None:
        vector = synthetic_vector()
        graphic = synthetic_graphic()
        observations = {
            agent: {"vector": vector.copy(), "graphic": graphic.copy()}
            for agent in reversed(canonical_agents())
        }
        batch = stack_observations(observations)
        self.assertEqual(batch.agent_names, canonical_agents())
        self.assertTrue(np.array_equal(batch.slot_ids, np.array([0, 1, 2, 3, 4] * 2)))
        self.assertEqual(batch.vectors.shape, (10, 96))
        self.assertEqual(batch.graphics_chw.shape, (10, 11, 4, 4))


class EvaluationContractTests(unittest.TestCase):
    def test_score_delta_sign(self) -> None:
        tracker = ScoreDeltaRewardTracker()
        increase = tracker.update_points((5, 0))
        self.assertEqual(increase.score_delta, (5, 0))
        self.assertAlmostEqual(increase.competitive_reward[0], 0.05)
        self.assertAlmostEqual(increase.competitive_reward[1], -0.05)

        decrease = tracker.update_points((2, 0))
        self.assertEqual(decrease.score_delta, (-3, 0))
        self.assertLess(decrease.competitive_reward[0], 0.0)
        self.assertGreater(decrease.competitive_reward[1], 0.0)

    def test_score_delta_theft_and_terminal(self) -> None:
        tracker = ScoreDeltaRewardTracker()
        tracker.reset((5, 0))
        theft = tracker.update_points((2, 0))
        stolen_deposit = tracker.update_points((2, 3))
        terminal = tracker.update_points(terminated=True, winner=1)
        self.assertEqual(theft.score_delta, (-3, 0))
        self.assertEqual(stolen_deposit.score_delta, (0, 3))
        self.assertLess(theft.total_reward[0], 0.0)
        self.assertLess(stolen_deposit.total_reward[0], 0.0)
        self.assertEqual(terminal.current_score, (2, 3))
        self.assertEqual(terminal.score_delta, (0, 0))
        self.assertEqual(terminal.terminal_bonus, (-1.0, 1.0))

    def test_logging_schema_round_trip(self) -> None:
        digest = "a" * 64
        record = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": "episode-1",
            "pair_id": "seed-1",
            "pair_index": 0,
            "seed": 1,
            "policy_seeds": {"model": 10, "opponent": 11},
            "side_assignment": {"model_team": 0, "opponent_team": 1},
            "model": {"checkpoint_sha256": digest},
            "opponent": {"checkpoint_sha256": digest},
            "environment": {"executable_sha256": digest},
            "episode_length": {"steps": 123},
            "score": {"team_a": 5, "team_b": 3},
            "winner": {"team": 0},
            "model_result": "win",
            "rewards": {},
            "termination": {"score_is_preterminal": False},
        }
        validate_episode_log(record)
        round_tripped = json.loads(json.dumps(record))
        validate_episode_log(round_tripped)
        self.assertEqual(round_tripped, record)

    def test_checkpoint_round_trip(self) -> None:
        torch.manual_seed(1213)
        model = SubmissionPolicy(hidden_dim=32, slot_embedding_dim=8)
        payload = checkpoint_payload(
            model,
            global_step=1234,
            training_seed=1213,
            source={"game_commit": "d2220a7", "python_api_commit": "6ba7d99"},
        )
        vector = torch.zeros((5, 96), dtype=torch.float32)
        graphic = torch.zeros((5, 11, 96, 96), dtype=torch.float32)
        before = model(vector, graphic)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, payload)
            loaded, restored = load_checkpoint(checkpoint)
            after = loaded(vector, graphic)
            upstream = load_upstream_checkpoint(
                SubmissionPolicy,
                str(checkpoint),
                vector_size=96,
                n_channels=11,
                hidden_dim=32,
                slot_embedding_dim=8,
            )
            upstream_after = upstream._net(vector, graphic)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(before, upstream_after))
        self.assertEqual(restored["training"], {"global_step": 1234, "seed": 1213})
        self.assertEqual(restored["schema_version"], "blackout.checkpoint.v1")

        schema = json.loads((ROOT / "schemas" / "checkpoint_v1.schema.json").read_text())
        self.assertIn("policy_state", schema["required"])
        self.assertIn("observation_contract", schema["required"])

    def test_paired_seed_evaluation(self) -> None:
        episodes = [
            {
                "model_result": model_result(0, 0),
                "pair_id": "seed-1",
                "pair_index": 0,
                "seed": 1,
                "winner": {"team": 0},
                "score": {
                    "team_a": 5,
                    "team_b": 3,
                    "model_minus_opponent": 2,
                },
                "side_assignment": {"model_side": "A", "model_team": 0},
            },
            {
                "model_result": model_result(1, 1),
                "pair_id": "seed-1",
                "pair_index": 1,
                "seed": 1,
                "winner": {"team": 1},
                "score": {
                    "team_a": 2,
                    "team_b": 6,
                    "model_minus_opponent": 4,
                },
                "side_assignment": {"model_side": "B", "model_team": 1},
            },
        ]
        summary = summarize_episodes(episodes)
        self.assertEqual((summary["wins"], summary["draws"], summary["losses"]), (2, 0, 0))
        self.assertEqual(summary["mean_model_score_diff"], 3.0)
        self.assertEqual(summary["by_model_side"]["A"]["wins"], 1)
        self.assertEqual(summary["by_model_side"]["B"]["wins"], 1)
        self.assertEqual(summary["by_physical_side"]["A"]["wins"], 1)
        self.assertEqual(summary["by_physical_side"]["B"]["wins"], 1)
        self.assertEqual(summary["side_bias_diagnostics"]["physical_win_rate_gap_a_minus_b"], 0.0)
        self.assertTrue(summary["side_bias_diagnostics"]["evaluator_side_attribution_passed"])
        self.assertFalse(summary["winner_derived_from_unity_shaping"])


class ModelInterfaceTests(unittest.TestCase):
    def team_observations(self, team: int = 0) -> dict[str, dict[str, np.ndarray]]:
        vector = synthetic_vector()
        tile_ids = np.arange(96 * 96, dtype=np.int64).reshape(96, 96) % 11
        graphic = np.eye(11, dtype=np.float32)[tile_ids]
        return {
            agent: {"vector": vector.copy(), "graphic": graphic.copy()}
            for agent in reversed(team_agents(team))
        }

    def test_actor_critic_input_output_and_slot_contract(self) -> None:
        batch = team_model_input(self.team_observations(), team=0)
        self.assertEqual(batch.agent_names, team_agents(0))
        self.assertEqual(tuple(batch.vector.shape), (5, 96))
        self.assertEqual(tuple(batch.graphic.shape), (5, 11, 96, 96))
        self.assertEqual(batch.vector.dtype, torch.float32)
        self.assertEqual(batch.graphic.dtype, torch.float32)
        self.assertEqual(batch.slot_id.dtype, torch.int64)
        self.assertTrue(torch.equal(batch.slot_id, torch.arange(5)))

        model = ReferenceActorCritic(hidden_dim=32, slot_embedding_dim=8)
        output = model(batch.vector, batch.graphic, batch.slot_id)
        self.assertEqual(tuple(output.action_logits.shape), (5, 9))
        self.assertEqual(tuple(output.value.shape), (5,))

    def test_categorical_action_adapter(self) -> None:
        action = categorical_action(torch.arange(9, dtype=torch.int64))
        self.assertEqual(tuple(action.shape), (9, 2))
        self.assertTrue(torch.equal(action[0], torch.zeros(2)))
        self.assertTrue(bool(torch.all(action >= -1.0) and torch.all(action <= 1.0)))
        self.assertTrue(
            torch.allclose(torch.linalg.vector_norm(action[1:], dim=-1), torch.ones(8))
        )

    def test_canonical_submission_adapter_ignores_dict_order(self) -> None:
        policy = SubmissionPolicy(hidden_dim=32, slot_embedding_dim=8)
        observations = self.team_observations(team=1)
        adapter = CanonicalTeamModel(policy, team=1)
        actions = adapter.act(observations)
        self.assertEqual(tuple(actions), team_agents(1))
        for action in actions.values():
            self.assertEqual(action.shape, (2,))
            self.assertEqual(action.dtype, np.float32)
            self.assertTrue(np.all((-1.0 <= action) & (action <= 1.0)))


@unittest.skipUnless(PREP14_PATH.exists(), "run eval/paired_series.py for PREP-14 first")
class PairedRandomEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series = json.loads(PREP14_PATH.read_text())

    def test_random_paired_series_side_attribution(self) -> None:
        validate_series_log(self.series)
        summary = self.series["summary"]
        self.assertEqual(summary["episodes"], 10)
        self.assertEqual(summary["by_model_side"]["A"]["episodes"], 5)
        self.assertEqual(summary["by_model_side"]["B"]["episodes"], 5)
        self.assertTrue(summary["side_bias_diagnostics"]["balanced_model_side_exposure"])
        self.assertTrue(
            summary["side_bias_diagnostics"]["model_result_attribution_consistent"]
        )
        self.assertTrue(summary["side_bias_diagnostics"]["evaluator_side_attribution_passed"])
        self.assertFalse(summary["winner_derived_from_unity_shaping"])

    def test_random_paired_series_has_complete_terminal_episodes(self) -> None:
        pairs: dict[str, list[dict]] = {}
        for episode in self.series["episodes"]:
            pairs.setdefault(episode["pair_id"], []).append(episode)
            self.assertEqual(episode["episode_length"]["steps"], 21_003)
            self.assertTrue(episode["termination"]["all_agents_terminated"])
            self.assertFalse(episode["termination"]["truncated"])
        self.assertEqual(len(pairs), 5)
        for episodes in pairs.values():
            self.assertEqual(
                {episode["side_assignment"]["model_team"] for episode in episodes},
                {0, 1},
            )


@unittest.skipUnless(EVIDENCE_PATH.exists(), "run scripts/verify_prep04_07.py first")
class LiveUnityEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE_PATH.read_text())

    def test_reset_step_termination_contract(self) -> None:
        contract = self.evidence["pettingzoo_contract"]
        self.assertTrue(contract["reset_ok"])
        self.assertTrue(contract["step_ok"])
        self.assertTrue(contract["termination_ok"])
        self.assertTrue(contract["no_truncation"])
        self.assertEqual(contract["terminal_agent_count"], 10)

    def test_seed_reproduces_initial_map(self) -> None:
        self.assertTrue(self.evidence["seed_reproducibility"]["same_seed_equal"])

    def test_agent_batch_order(self) -> None:
        order = self.evidence["agent_identity_and_order"]
        self.assertTrue(order["canonical_batch_stable"])
        self.assertEqual(order["team_batch_sizes"], [5, 5])
        self.assertEqual(order["canonical_agents"], list(canonical_agents()))
        self.assertEqual(order["team_a_agents"], list(team_agents(0)))
        self.assertEqual(order["team_b_agents"], list(team_agents(1)))

    def test_team_side_swap(self) -> None:
        self.assertTrue(self.evidence["agent_identity_and_order"]["team_perspective_swap_ok"])

    def test_zero_action_stops_unit(self) -> None:
        self.assertLessEqual(self.evidence["action_semantics"]["zero"]["magnitude"], 1e-7)

    def test_nonzero_action_is_normalized(self) -> None:
        action = self.evidence["action_semantics"]
        self.assertTrue(action["small_large_equal_speed"])
        self.assertTrue(action["oversized_action_clipped"])
        self.assertTrue(action["eight_directions_match"])

    def test_terminal_winner_matches_game_result(self) -> None:
        terminal = self.evidence["terminal"]
        self.assertTrue(terminal["winner_matches_score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
