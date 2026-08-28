from __future__ import annotations

import unittest
import numpy as np
import torch

from blackout_rl.batching import canonical_agents
from blackout_rl.ippo_model import IPPOActorCritic
from blackout_rl.mappo import (
    CENTRAL_VECTOR_SIZE, AblationArm, CentralizedStateBuilder, JointRolloutBuffer,
    MAPPOParallelRolloutCollector,
    compare_ippo_mappo, initialize_mappo_from_ippo, mappo_update, team_alive_mask,
    initialize_planner_residual_head,
)
from blackout_rl.ppo import PPOConfig
from tests.test_ppo_training import MockParallelEnv, small_model
from blackout_rl.policy import NoOpPolicy
from blackout_rl.training_reward import TeamTrainingReward


class CentralStateTests(unittest.TestCase):
    def test_builder_collects_all_units_classes_scores_and_map(self) -> None:
        state = CentralizedStateBuilder().build(MockParallelEnv()._observations(), learning_team=0)
        self.assertEqual(state.vector.shape, (1, CENTRAL_VECTOR_SIZE))
        self.assertEqual(state.graphic.shape, (1, 11, 16, 16))

    def test_missing_opponent_observation_fails_closed(self) -> None:
        obs = MockParallelEnv()._observations(); obs.pop("unit_9")
        with self.assertRaisesRegex(KeyError, "all ten"):
            CentralizedStateBuilder().build(obs, learning_team=0)


class MAPPOModelTests(unittest.TestCase):
    def test_ippo_actor_is_bit_identical_after_mappo_initialization(self) -> None:
        ippo = small_model().eval(); mappo = initialize_mappo_from_ippo(ippo).eval()
        vector = torch.zeros(5, 96); graphic = torch.zeros(5, 11, 96, 96); slots = torch.arange(5)
        with torch.no_grad():
            expected = ippo(vector, graphic, slots).action_logits
            actual = mappo.decentralized_action_logits(vector, graphic, slots)
        self.assertTrue(torch.equal(expected, actual))

    def test_planner_residual_head_starts_with_deterministic_fallback(self) -> None:
        model = initialize_mappo_from_ippo(small_model())
        initialize_planner_residual_head(model, fallback_logit_bias=4.0)
        vector = torch.zeros(5, 96)
        graphic = torch.zeros(5, 11, 16, 16)
        slots = torch.arange(5)

        logits = model.actor_logits(vector, graphic, slots)

        self.assertEqual(torch.argmax(logits, dim=-1).tolist(), [0] * 5)
        self.assertTrue(torch.equal(model.actor_model.actor.weight, torch.zeros_like(model.actor_model.actor.weight)))

    def test_death_and_respawn_only_mask_actor_rows(self) -> None:
        self.assertEqual(team_alive_mask(("unit_0", "unit_2"), 0).tolist(), [True, False, True, False, False])

    def test_joint_buffer_and_update_keep_team_critic_target_finite(self) -> None:
        ippo = IPPOActorCritic(hidden_dim=16, entity_dim=4, context_dim=4, graphic_dim=8, slot_embedding_dim=4)
        model = initialize_mappo_from_ippo(ippo)
        buffer = JointRolloutBuffer()
        for step, mask in enumerate(([1,1,0,1,1], [1,1,1,1,1])):
            vector = torch.zeros(5, 96); graphic = torch.zeros(5, 11, 8, 8); slots = torch.arange(5)
            central_vector = torch.zeros(1, CENTRAL_VECTOR_SIZE); central_graphic = torch.zeros(1, 11, 8, 8)
            with torch.no_grad():
                output = model(vector, graphic, slots, central_vector.expand(5,-1), central_graphic.expand(5,-1,-1,-1))
                dist = torch.distributions.Categorical(logits=output.action_logits); actions = dist.sample()
            buffer.add(vector=vector, graphic=graphic, slot_id=slots, central_vector=central_vector,
                       central_graphic=central_graphic, team_mask=torch.tensor(mask, dtype=torch.bool),
                       action_index=actions, log_prob=dist.log_prob(actions), value=output.value[:1],
                       reward=torch.ones(5), next_value=output.value[:1],
                       terminated=torch.tensor([step == 1]), truncated=torch.tensor([False]))
        batch = buffer.as_batch(gamma=.99, gae_lambda=.95)
        self.assertEqual(int(batch.team_mask.sum()), 9)
        metrics = mappo_update(model, torch.optim.Adam(model.parameters(), 1e-3), batch,
                               config=PPOConfig(update_epochs=1, minibatch_size=9))
        self.assertEqual(metrics.samples, 9)
        self.assertTrue(np.isfinite(metrics.value_loss))

    def test_ablation_rejects_mismatched_budget(self) -> None:
        ippo = AblationArm("ippo", (1,2), "fixed", 100, .4, 1)
        mappo = AblationArm("mappo", (1,2), "fixed", 101, .5, 2)
        with self.assertRaisesRegex(ValueError, "share seeds"):
            compare_ippo_mappo(ippo, mappo)

    def test_live_collector_path_records_joint_central_state(self) -> None:
        model=initialize_mappo_from_ippo(small_model())
        collector=MAPPOParallelRolloutCollector(MockParallelEnv(),model,NoOpPolicy(),learning_team=0)
        batch=collector.collect(3,seed=4).as_batch(gamma=.99,gae_lambda=.95)
        self.assertEqual(len(batch),15)
        self.assertEqual(batch.central_vector.shape,(15,CENTRAL_VECTOR_SIZE))

    def test_collector_persists_across_updates_and_reseeds_completed_episodes(self) -> None:
        env = MockParallelEnv()
        seeds = iter((3001, 3002, 3003))
        collector = MAPPOParallelRolloutCollector(
            env,
            initialize_mappo_from_ippo(small_model()),
            NoOpPolicy(),
            learning_team=0,
            reward_transform=TeamTrainingReward(0),
            episode_seed_provider=lambda: next(seeds),
        )

        collector.collect(1)
        collector.collect(2)

        self.assertEqual(env.reset_seeds, [3001, 3002])
        self.assertEqual(collector.environment_steps, 3)
        self.assertEqual(collector.episodes_completed, 1)
        self.assertEqual(collector.terminal_episodes, 1)
        self.assertEqual(collector.truncated_episodes, 0)
        self.assertEqual(collector.wins, 1)
        self.assertEqual(collector.draws, 0)
        self.assertEqual(collector.losses, 0)

    def test_teacher_labels_and_forcing_are_recorded_on_joint_rollout(self) -> None:
        env = MockParallelEnv()
        collector = MAPPOParallelRolloutCollector(
            env,
            initialize_mappo_from_ippo(small_model()),
            NoOpPolicy(),
            learning_team=0,
            teacher=NoOpPolicy(),
            teacher_forcing_probability=1.0,
            teacher_seed=42,
        )

        batch = collector.collect(1, seed=3001).as_batch(gamma=.99, gae_lambda=.95)

        self.assertIsNotNone(batch.teacher_action_index)
        self.assertTrue(torch.equal(batch.teacher_action_index, torch.zeros(5, dtype=torch.int64)))
        self.assertEqual(collector.teacher_labeled_steps, 1)
        self.assertEqual(collector.teacher_forced_steps, 1)
        for agent in collector.controlled_agents:
            self.assertTrue(np.array_equal(env.actions_seen[0][agent], np.zeros(2, dtype=np.float32)))

    def test_residual_fallback_executes_planner_and_labels_action_zero(self) -> None:
        env = MockParallelEnv()
        model = initialize_mappo_from_ippo(small_model())
        initialize_planner_residual_head(model)
        planner = NoOpPolicy()
        collector = MAPPOParallelRolloutCollector(
            env,
            model,
            NoOpPolicy(),
            learning_team=0,
            residual_base=planner,
            teacher_forcing_probability=1.0,
            teacher_seed=42,
        )

        batch = collector.collect(1, seed=3001).as_batch(gamma=.99, gae_lambda=.95)

        self.assertTrue(torch.equal(batch.teacher_action_index, torch.zeros(5, dtype=torch.int64)))
        self.assertEqual(collector.residual_fallback_actions, 5)
        self.assertEqual(collector.residual_override_actions, 0)


if __name__ == "__main__": unittest.main(verbosity=2)
