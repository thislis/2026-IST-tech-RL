"""BASE-R01~R05/R14 tests for the entity-aware IPPO policy vertical slice."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from blackout_rl import (
    ACTION_DIRECTIONS,
    EntityVectorEncoder,
    IPPOActorCritic,
    SemanticCNNEncoder,
    SubmissionPolicy,
    evaluate_categorical_action,
    select_categorical_action,
)


class CategoricalActionDistributionTests(unittest.TestCase):
    def test_sample_and_argmax_share_nine_way_action_contract(self) -> None:
        logits = torch.eye(9, dtype=torch.float32) * 100.0
        sampled = select_categorical_action(logits)
        deterministic = select_categorical_action(logits, deterministic=True)

        expected_index = torch.arange(9, dtype=torch.int64)
        self.assertTrue(torch.equal(sampled.index, expected_index))
        self.assertTrue(torch.equal(deterministic.index, expected_index))
        self.assertTrue(torch.equal(sampled.action, deterministic.action))
        self.assertEqual(tuple(sampled.action.shape), (9, 2))
        self.assertTrue(bool(torch.all(sampled.action >= -1.0)))
        self.assertTrue(bool(torch.all(sampled.action <= 1.0)))
        self.assertTrue(torch.isfinite(sampled.log_prob).all())
        self.assertTrue(torch.isfinite(sampled.entropy).all())

    def test_stored_action_statistics_are_ppo_reusable(self) -> None:
        torch.manual_seed(2201)
        logits = torch.randn(7, 9, requires_grad=True)
        selected = select_categorical_action(logits)
        reevaluated = evaluate_categorical_action(logits, selected.index)
        self.assertTrue(torch.equal(selected.action, reevaluated.action))
        self.assertTrue(torch.allclose(selected.log_prob, reevaluated.log_prob))

        loss = -(reevaluated.log_prob + 0.01 * reevaluated.entropy).mean()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


class EncoderTests(unittest.TestCase):
    def test_vector_encoder_separates_ten_entities_and_context(self) -> None:
        torch.manual_seed(2202)
        encoder = EntityVectorEncoder(entity_dim=12, context_dim=7)
        vector = torch.zeros(2, 96)
        baseline = encoder(vector)
        changed_vector = vector.clone()
        changed_vector[:, :9] = 1.0
        changed = encoder(changed_vector)

        self.assertEqual(tuple(baseline.entities.shape), (2, 10, 12))
        self.assertEqual(tuple(baseline.context.shape), (2, 7))
        self.assertEqual(tuple(baseline.latent.shape), (2, 127))
        self.assertFalse(torch.equal(baseline.entities[:, 0], changed.entities[:, 0]))
        self.assertTrue(torch.equal(baseline.entities[:, 1:], changed.entities[:, 1:]))
        self.assertTrue(torch.equal(baseline.context, changed.context))

    def test_semantic_cnn_has_fixed_latent_for_official_input(self) -> None:
        encoder = SemanticCNNEncoder(graphic_dim=40)
        official = encoder(torch.zeros(3, 11, 96, 96))
        alternate_spatial_size = encoder(torch.zeros(3, 11, 48, 48))
        self.assertEqual(tuple(official.shape), (3, 40))
        self.assertEqual(tuple(alternate_spatial_size.shape), (3, 40))


class SharedActorCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2203)
        self.model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
        )

    def test_shared_heads_return_slot_specific_actor_and_local_value(self) -> None:
        vector = torch.zeros(5, 96)
        graphic = torch.zeros(5, 11, 32, 32)
        slots = torch.arange(5, dtype=torch.int64)
        output = self.model(vector, graphic, slots)

        self.assertIsInstance(self.model.actor, nn.Linear)
        self.assertIsInstance(self.model.critic, nn.Linear)
        self.assertEqual(tuple(output.action_logits.shape), (5, 9))
        self.assertEqual(tuple(output.value.shape), (5,))
        self.assertFalse(torch.equal(output.action_logits[0], output.action_logits[1]))

        (output.action_logits.sum() + output.value.sum()).backward()
        slot_gradient = self.model.slot_embedding.weight.grad
        self.assertIsNotNone(slot_gradient)
        self.assertTrue(torch.all(torch.linalg.vector_norm(slot_gradient, dim=-1) > 0.0))

    def test_same_observation_and_slot_use_the_same_shared_parameters(self) -> None:
        vector = torch.zeros(2, 96)
        graphic = torch.zeros(2, 11, 32, 32)
        slots = torch.zeros(2, dtype=torch.int64)
        output = self.model(vector, graphic, slots)
        self.assertTrue(torch.equal(output.action_logits[0], output.action_logits[1]))
        self.assertEqual(output.value[0], output.value[1])


class DeterministicSubmissionPolicyTests(unittest.TestCase):
    def test_forward_returns_argmax_direction_in_official_shape_and_range(self) -> None:
        policy = SubmissionPolicy(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
        )
        with torch.no_grad():
            policy.actor_critic.actor.weight.zero_()
            policy.actor_critic.actor.bias.zero_()
            policy.actor_critic.actor.bias[2] = 10.0

        action = policy(
            torch.zeros(5, 96),
            torch.zeros(5, 11, 96, 96),
        )
        expected = ACTION_DIRECTIONS[2] / torch.linalg.vector_norm(ACTION_DIRECTIONS[2])
        self.assertEqual(tuple(action.shape), (5, 2))
        self.assertTrue(torch.allclose(action, expected.expand(5, 2)))
        self.assertTrue(bool(torch.all(action >= -1.0)))
        self.assertTrue(bool(torch.all(action <= 1.0)))

    def test_forward_rejects_non_team_batch(self) -> None:
        policy = SubmissionPolicy(hidden_dim=16, graphic_dim=8, entity_dim=4, context_dim=4)
        with self.assertRaisesRegex(ValueError, "exactly 5 agents"):
            policy(torch.zeros(4, 96), torch.zeros(4, 11, 96, 96))


if __name__ == "__main__":
    unittest.main(verbosity=2)
