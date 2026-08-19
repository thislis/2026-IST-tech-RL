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

    def test_global_local_map_encoder_runs_for_all_slots(self) -> None:
        model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            map_encoder_version="global_local_v1",
        )
        vector = torch.zeros(5, 96)
        vector[:, 2:45:9] = 1.0
        vector[:, 0:45:9] = torch.linspace(0.1, 0.9, 5)
        vector[:, 1:45:9] = torch.linspace(0.9, 0.1, 5)
        graphic = torch.zeros(5, 11, 96, 96)
        graphic[:, 6, 45:51, 45:51] = 1.0
        output = model(vector, graphic, torch.arange(5, dtype=torch.int64))
        self.assertEqual(tuple(output.action_logits.shape), (5, 9))
        self.assertEqual(model.model_config["map_encoder_version"], "global_local_v1")

    def test_entity_attention_and_auxiliary_heads_share_actor_latent(self) -> None:
        model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            entity_encoder_version="self_attention_v1",
            auxiliary_targets=("role", "holding_item"),
        )
        vector = torch.zeros(5, 96)
        blocks = vector[:, :90].reshape(5, 10, 9)
        blocks[:, :5, 2] = 1.0
        blocks[:, 5:, 2] = -1.0
        blocks[:, :, 3] = 1.0
        graphic = torch.zeros(5, 11, 32, 32)
        slots = torch.arange(5, dtype=torch.int64)
        latent = model.encode(vector, graphic, slots)
        predictions = model.predict_auxiliary(latent)
        self.assertEqual(set(predictions), {"role", "holding_item"})
        self.assertEqual(tuple(predictions["role"].shape), (5, 3))
        self.assertEqual(tuple(predictions["holding_item"].shape), (5, 6))
        loss = (
            model.actor(latent).sum()
            + predictions["role"].sum()
            + predictions["holding_item"].sum()
        )
        loss.backward()
        self.assertIsNotNone(model.entity_attention.attention.in_proj_weight.grad)
        self.assertTrue(
            torch.isfinite(model.entity_attention.attention.in_proj_weight.grad).all()
        )

    def test_unknown_attention_or_auxiliary_contract_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown entity encoder"):
            IPPOActorCritic(entity_encoder_version="transformer_future")
        with self.assertRaisesRegex(ValueError, "unknown auxiliary"):
            IPPOActorCritic(auxiliary_targets=("future_score",))

    def test_spatial_target_residual_runs_for_all_slots(self) -> None:
        model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            map_encoder_version="global_local_spatial_v2",
        )
        vector = torch.zeros(5, 96)
        vector[:, 2:45:9] = 1.0
        vector[:, 0:45:9] = 0.5
        vector[:, 1:45:9] = 0.5
        graphic = torch.zeros(5, 11, 96, 96)
        graphic[:, 6, 20:24, 70:74] = 1.0
        output = model(vector, graphic, torch.arange(5, dtype=torch.int64))
        self.assertTrue(torch.isfinite(output.action_logits).all())
        self.assertIsNotNone(model.spatial_target_residual)

    def test_multitarget_residual_runs_for_all_slots(self) -> None:
        model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            map_encoder_version="global_local_multitarget_v3",
        )
        vector = torch.zeros(5, 96)
        vector[:, 2:45:9] = 1.0
        vector[:, 0:45:9] = 0.5
        vector[:, 1:45:9] = 0.5
        graphic = torch.zeros(5, 11, 96, 96)
        graphic[:, 6, 20:24, 70:74] = 1.0
        output = model(vector, graphic, torch.arange(5, dtype=torch.int64))
        self.assertTrue(torch.isfinite(output.action_logits).all())
        self.assertIsNotNone(model.multi_target_residual)

    def test_team_relative_entity_order_is_side_invariant(self) -> None:
        model = IPPOActorCritic(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            entity_order_version="team_relative_v1",
        ).eval()
        team_a = torch.zeros(1, 96)
        blocks = team_a[:, :90].reshape(1, 10, 9)
        blocks[:, :5, 2] = 1.0
        blocks[:, 5:, 2] = -1.0
        blocks[:, :, 3] = 1.0
        team_b = team_a.clone()
        team_b_blocks = team_b[:, :90].reshape(1, 10, 9)
        team_b_blocks[:, :5, 2] = -1.0
        team_b_blocks[:, 5:, 2] = 1.0
        team_b_blocks[:, :, 3] = 1.0
        graphic = torch.zeros(1, 11, 32, 32)
        slot = torch.tensor([2])
        with torch.inference_mode():
            first = model(team_a, graphic, slot)
            second = model(team_b, graphic, slot)
        self.assertTrue(torch.equal(first.action_logits, second.action_logits))

    def test_continuous_head_submission_is_bounded(self) -> None:
        policy = SubmissionPolicy(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            action_head_version="continuous_tanh_v1",
        )
        action = policy(torch.zeros(5, 96), torch.zeros(5, 11, 32, 32))
        self.assertEqual(tuple(action.shape), (5, 2))
        self.assertTrue(torch.all((action >= -1.0) & (action <= 1.0)))

    def test_recurrent_policy_tracks_and_resets_inference_state(self) -> None:
        policy = SubmissionPolicy(
            hidden_dim=32,
            entity_dim=8,
            context_dim=8,
            graphic_dim=16,
            slot_embedding_dim=6,
            action_head_version="continuous_tanh_v1",
            recurrent_version="gru_v1",
        ).eval()
        vector = torch.zeros(5, 96)
        graphic = torch.zeros(5, 11, 32, 32)
        with torch.inference_mode():
            first = policy(vector, graphic)
            second = policy(vector, graphic)
            policy.reset_inference_state()
            reset = policy(vector, graphic)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(first, reset))

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
