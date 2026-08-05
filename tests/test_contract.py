"""PREP-04~07 regression tests for parsers and recorded live-Unity evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from blackout_rl import (
    GRAPHIC_CHANNEL_NAMES,
    VECTOR_SIZE,
    canonical_agents,
    hwc_to_chw,
    parse_graphic,
    parse_vector,
    stack_observations,
    team_agents,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "logs" / "prep04_07_contract.json"


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
