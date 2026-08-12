"""BASE-R10~R12 reward, frozen opponent, and BC warm-start tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from blackout_rl import (
    FrozenScriptedOpponent,
    IPPOActorCritic,
    RewardMode,
    ScriptedTrajectoryDataset,
    TeamTrainingReward,
    TrainingRewardConfig,
    action_vector_to_index,
    evaluate_behavior_cloning,
    run_bc_warm_start_experiment,
)
from blackout_rl.trajectory import decode_graphic


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TRAJECTORY = ROOT / "logs" / "base_s08_s10_role_fsm_trajectory.jsonl"
HELD_OUT_TRAJECTORY = ROOT / "logs" / "base_s13_coordination_trajectory.jsonl"
BC_EXPERIMENT_LOG = ROOT / "logs" / "base_r12_bc_warm_start.json"


def transition_inputs(
    *,
    score_a: float,
    score_b: float,
    unity_reward: float = 0.0,
    terminated: bool = False,
    winner: int | None = None,
) -> tuple[dict, dict, dict, dict, tuple[str, ...]]:
    agents = tuple(f"unit_{index}" for index in range(5))
    rewards = {agent: unity_reward + index for index, agent in enumerate(agents)}
    terminations = {agent: terminated for agent in agents}
    truncations = {agent: False for agent in agents}
    info = {"score_0": score_a, "score_1": score_b}
    if winner is not None:
        info["winner"] = winner
    infos = {agent: dict(info) for agent in agents}
    return rewards, terminations, truncations, infos, agents


class TeamRewardTests(unittest.TestCase):
    def test_score_delta_is_shared_and_terminal_bonus_uses_winner(self) -> None:
        reward = TeamTrainingReward(learning_team=0)
        first = reward(*transition_inputs(score_a=0.05, score_b=0.02, unity_reward=10.0))
        self.assertTrue(all(value == 0.03 for value in first.values()))
        terminal = reward(
            *transition_inputs(
                score_a=0.0,
                score_b=0.0,
                unity_reward=100.0,
                terminated=True,
                winner=0,
            )
        )
        self.assertTrue(all(value == 1.0 for value in terminal.values()))
        self.assertEqual(reward.tracker.score, (5, 2))

    def test_unity_shaping_can_be_off_on_or_combined(self) -> None:
        inputs = transition_inputs(score_a=0.02, score_b=0.0, unity_reward=3.0)
        score_only = TeamTrainingReward(
            0, TrainingRewardConfig(mode=RewardMode.SCORE_DELTA)
        )(*inputs)
        unity_only = TeamTrainingReward(
            0, TrainingRewardConfig(mode=RewardMode.UNITY_SHAPING)
        )(*inputs)
        combined = TeamTrainingReward(
            0,
            TrainingRewardConfig(
                mode=RewardMode.COMBINED,
                score_delta_weight=2.0,
                unity_shaping_weight=0.5,
            ),
        )(*inputs)
        agents = inputs[-1]
        self.assertEqual([score_only[agent] for agent in agents], [0.02] * 5)
        self.assertEqual([unity_only[agent] for agent in agents], [3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(
            [combined[agent] for agent in agents],
            [1.54, 2.04, 2.54, 3.04, 3.54],
        )


@unittest.skipUnless(TRAIN_TRAJECTORY.exists(), "scripted trajectory evidence is required")
class FrozenOpponentTests(unittest.TestCase):
    def test_scripted_opponent_has_no_trainable_state_and_config_stays_frozen(self) -> None:
        step = next(
            record
            for record in map(json.loads, TRAIN_TRAJECTORY.read_text().splitlines())
            if record.get("record_type") == "step"
        )
        graphic_ids = decode_graphic(step["observation"]["team_graphic"])
        graphic = np.eye(11, dtype=np.float32)[graphic_ids]
        observations = {
            agent: {
                "vector": np.asarray(vector, dtype=np.float32),
                "graphic": graphic,
            }
            for agent, vector in step["observation"]["vectors"].items()
        }
        opponent = FrozenScriptedOpponent(team=0, seed=1100)
        fingerprint = opponent.frozen_fingerprint
        actions = opponent.act(observations, tuple(observations))
        self.assertEqual(opponent.trainable_parameters, ())
        self.assertEqual(opponent.frozen_fingerprint, fingerprint)
        self.assertEqual(set(actions), set(observations))
        opponent.reset()
        repeated = opponent.act(observations, tuple(observations))
        self.assertEqual(opponent.frozen_fingerprint, fingerprint)
        for agent in actions:
            self.assertTrue(np.array_equal(actions[agent], repeated[agent]))


@unittest.skipUnless(
    TRAIN_TRAJECTORY.exists() and HELD_OUT_TRAJECTORY.exists(),
    "scripted trajectory evidence is required",
)
class BehaviorCloningTests(unittest.TestCase):
    def test_action_quantization_covers_noop_and_diagonal(self) -> None:
        self.assertEqual(action_vector_to_index((0.0, 0.0)), 0)
        self.assertEqual(action_vector_to_index((10.0, 10.0)), 2)
        self.assertEqual(action_vector_to_index((-2.0, 0.0)), 5)

    def test_dataset_decodes_agent_rows_and_bc_improves_training_nll(self) -> None:
        dataset = ScriptedTrajectoryDataset([HELD_OUT_TRAJECTORY])
        self.assertEqual(len(dataset), 31 * 5)
        vector, graphic, slot_id, action_index = dataset[0]
        self.assertEqual(tuple(vector.shape), (96,))
        self.assertEqual(tuple(graphic.shape), (11, 96, 96))
        self.assertEqual(slot_id.dtype, torch.int64)
        self.assertEqual(action_index.dtype, torch.int64)

        torch.manual_seed(1212)
        model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
        )
        before = evaluate_behavior_cloning(model, dataset, batch_size=64)
        _, result = run_bc_warm_start_experiment(
            model,
            dataset,
            dataset,
            epochs=5,
            batch_size=64,
            learning_rate=2e-3,
            downstream_environment_steps=10_000,
            seed=1212,
        )
        self.assertAlmostEqual(result.scratch.loss, before.loss)
        self.assertLess(result.bc_pretrained.loss, result.scratch.loss)
        self.assertEqual(
            result.downstream_environment_step_budget,
            {"scratch": 10_000, "bc_pretrained": 10_000},
        )
        self.assertEqual(result.demonstration_sample_budget, 5 * len(dataset))


@unittest.skipUnless(BC_EXPERIMENT_LOG.exists(), "run scripts/run_base_r12_bc.py first")
class BCExperimentEvidenceTests(unittest.TestCase):
    def test_recorded_experiment_uses_separate_data_and_equal_environment_steps(self) -> None:
        experiment = json.loads(BC_EXPERIMENT_LOG.read_text())
        self.assertEqual(experiment["schema_version"], "blackout.bc_experiment.v1")
        self.assertTrue(
            set(experiment["train_sources"]).isdisjoint(experiment["held_out_sources"])
        )
        self.assertEqual(
            experiment["downstream_environment_step_budget"]["scratch"],
            experiment["downstream_environment_step_budget"]["bc_pretrained"],
        )
        self.assertLess(
            experiment["bc_pretrained"]["loss"], experiment["scratch"]["loss"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
