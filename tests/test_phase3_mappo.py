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
)
from blackout_rl.ppo import PPOConfig
from tests.test_ppo_training import MockParallelEnv, small_model
from blackout_rl.policy import NoOpPolicy


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


if __name__ == "__main__": unittest.main(verbosity=2)
