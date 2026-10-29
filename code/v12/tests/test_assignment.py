import itertools
import unittest
import torch
from blackout_v12.assignment import (log_partition, map_assignment, sample_assignment,
    joint_log_prob, marginals, joint_entropy, sequential_assignment, sequential_log_prob)
from blackout_v12.model import TaskScorer


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.scores = torch.tensor([[1.2, .7, -.4, 0], [.9, 1.1, .2, 0],
                                    [-.3, .6, 1.4, 0]], dtype=torch.double)
        self.cap = torch.tensor([1, 2, 1, 3])
        self.legal = torch.ones(3, 4, dtype=torch.bool)
        self.legal[0, 2] = self.legal[2, 0] = False

    def test_partition_marginal_gradient_and_map_against_enumeration(self):
        choices = [a for a in itertools.product(range(4), repeat=3)
                   if all(self.legal[u, t] for u, t in enumerate(a))
                   and all(a.count(t) <= self.cap[t] for t in range(4))]
        self.assertEqual(len(choices), 29)
        score = torch.stack([sum(self.scores[u, t] for u, t in enumerate(a)) / .7 for a in choices])
        expected = torch.logsumexp(score, 0)
        result = log_partition(self.scores, self.cap, self.legal, .7)
        self.assertAlmostEqual(result.item(), 6.5988516569678035, places=12)
        torch.testing.assert_close(result, expected)
        actual_map = map_assignment(self.scores, self.cap, self.legal, .7)
        self.assertEqual(actual_map.tolist(), [0, 1, 2])
        probability = score.softmax(0)
        expected_marginal = torch.zeros_like(self.scores)
        for p, choice in zip(probability, choices):
            for unit, task in enumerate(choice):
                expected_marginal[unit, task] += p
        torch.testing.assert_close(marginals(self.scores, self.cap, self.legal, .7), expected_marginal)
        self.assertTrue(torch.autograd.gradcheck(lambda s: log_partition(s, self.cap, self.legal, .7),
                                                (self.scores.clone().requires_grad_(),), fast_mode=True))
        torch.testing.assert_close(joint_entropy(self.scores, self.cap, self.legal, .7),
                                   -(probability * probability.log()).sum())

    def test_forced_capacity_zero_locked_and_finite_gradients(self):
        scores = torch.randn(5, 6, requires_grad=True)
        legal = torch.zeros(5, 6, dtype=torch.bool)
        legal[range(5), range(5)] = True
        caps = torch.tensor([1, 1, 1, 1, 1, 0])
        choice = sample_assignment(scores, caps, legal)
        logp = joint_log_prob(scores, choice, caps, legal)
        self.assertAlmostEqual(logp.item(), 0, places=5)
        logp.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores.grad))
        with self.assertRaises(ValueError):
            log_partition(scores, torch.zeros(6, dtype=torch.long), legal)

    def test_batch_sampling_replay_and_permutation(self):
        scores = self.scores.expand(64, -1, -1)
        assignments = sample_assignment(scores, self.cap, self.legal, .7,
                                       generator=torch.Generator().manual_seed(17))
        old = joint_log_prob(scores, assignments, self.cap, self.legal, .7)
        new = joint_log_prob(scores.clone(), assignments, self.cap, self.legal, .7)
        torch.testing.assert_close((new-old).exp(), torch.ones_like(old))
        self.assertTrue(torch.isfinite(old).all())
        permutation = torch.tensor([2, 0, 1])
        actual = map_assignment(self.scores[permutation], self.cap, self.legal[permutation], .7)
        torch.testing.assert_close(actual, torch.tensor([0, 1, 2])[permutation])

    def test_sequential_does_not_steal_locked_reservation(self):
        scores = torch.tensor([[10., 0.], [1., 0.]])
        legal = torch.tensor([[True, True], [True, False]])
        cap = torch.tensor([1, 1])
        choice = sequential_assignment(scores, cap, legal)
        self.assertEqual(choice.tolist(), [1, 0])
        self.assertEqual(sequential_log_prob(scores, choice, cap, legal).item(), 0.)

    def test_model_size_and_prior_initialization(self):
        model = TaskScorer()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6273)
        prior = torch.randn(5, 24)
        torch.testing.assert_close(model(torch.randn(5, 24, 64), prior), prior)

    def test_cached_inference_indices_remain_safe_for_training(self):
        from blackout_v12.assignment import _topology
        _topology.cache_clear()
        with torch.inference_mode():
            map_assignment(self.scores, self.cap, self.legal)
        scores = self.scores.clone().requires_grad_()
        log_partition(scores, self.cap, self.legal).backward()
        self.assertTrue(torch.isfinite(scores.grad).all())


if __name__ == '__main__':
    unittest.main()
