"""Synthetic full-support replay and exact entropy checks; no game process."""
import itertools
import unittest
import torch
from blackout_v12.assignment import joint_entropy, joint_log_prob, sample_assignment
from blackout_v12.critic import Critic, critic_features
from blackout_v12.model import TaskScorer
from blackout_v12.ppo import event_log_prob, update
from blackout_v12.training import make_batch


class LearnedPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def episode(self, scorer, task_counts):
        events = []
        for step, tasks in enumerate(task_counts):
            features = torch.randn(5, tasks, 64)
            prior = torch.randn(5, tasks)
            legal = torch.ones(5, tasks, dtype=torch.bool)
            capacities = torch.full((tasks,), 5, dtype=torch.long)
            capacities[0] = 1
            temperature = torch.tensor(.8)
            with torch.no_grad():
                scores = scorer(features, prior)
                assignment = sample_assignment(scores, capacities, legal, temperature)
                old_log_prob = joint_log_prob(scores, assignment, capacities, legal, temperature)
            events.append(dict(features=features, prior=prior, legal=legal,
                capacities=capacities, assignment=assignment, temperature=temperature,
                old_log_prob=old_log_prob, critic_features=critic_features(features, legal),
                step=step, duration=1, reward=float(step % 2), on_policy=True))
        return dict(events=events, natural_terminal=True, on_policy=True, env_steps=len(events))

    def test_variable_task_padding_keeps_immutable_assignment_support(self):
        torch.manual_seed(61)
        scorer, critic = TaskScorer({'policy_mode':'learned'}), Critic()
        torch.nn.init.normal_(scorer.layers[-1].weight, std=.3)
        episode = self.episode(scorer, [3, 30])
        original = {key:value.clone() for key,value in episode['events'][0].items()
                    if torch.is_tensor(value)}
        batch = make_batch([episode], critic)
        self.assertEqual(batch['features'].shape, (2,5,30,64))
        self.assertEqual(batch['prior'].shape, (2,5,30))
        self.assertEqual(batch['legal'].shape, (2,5,30))
        self.assertEqual(batch['capacities'].shape, (2,30))
        self.assertEqual(batch['assignment'].shape, (2,5))
        self.assertEqual(batch['temperature'].shape, (2,))
        self.assertEqual(batch['critic_features'].shape, (2,64))
        self.assertFalse(batch['legal'][0,:,3:].any())
        self.assertEqual(batch['capacities'][0,3:].sum().item(), 0)
        self.assertEqual(batch['features'][0,:,3:].abs().sum().item(), 0.)
        self.assertEqual(batch['prior'][0,:,3:].abs().sum().item(), 0.)
        torch.testing.assert_close(event_log_prob(scorer,batch,'J'), batch['old_log_prob'],
                                   atol=2e-5, rtol=0)
        for key, value in original.items():
            torch.testing.assert_close(episode['events'][0][key], value)
        legacy = self.episode(TaskScorer(), [24,24])
        legacy_batch = make_batch([legacy], critic)
        for key in ('features','prior','legal','capacities','assignment','temperature','critic_features'):
            torch.testing.assert_close(legacy_batch[key], torch.stack([e[key] for e in legacy['events']]))

    def test_exact_entropy_gradient_matches_assignment_enumeration(self):
        scores = torch.tensor([[1.2,-.3,.5],[-.4,.9,.2]], dtype=torch.double, requires_grad=True)
        legal = torch.tensor([[True,True,False],[True,True,True]])
        capacities = torch.tensor([1,2,1])
        choices = [a for a in itertools.product(range(3), repeat=2)
                   if all(legal[u,t] for u,t in enumerate(a))
                   and all(a.count(t) <= capacities[t] for t in range(3))]
        logits = torch.stack([sum(scores[u,t] for u,t in enumerate(a)) / .7 for a in choices])
        log_probability = logits.log_softmax(0)
        expected = -(log_probability.exp() * log_probability).sum()
        actual = joint_entropy(scores, capacities, legal, .7)
        expected_gradient, = torch.autograd.grad(expected, scores, retain_graph=True)
        actual_gradient, = torch.autograd.grad(actual, scores)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_adaptive_minibatch_and_entropy_drive_learned_scorer(self):
        torch.manual_seed(62)
        scorer, critic = TaskScorer({'policy_mode':'learned'}), Critic()
        torch.nn.init.normal_(scorer.layers[-1].weight, std=.4)
        episode = self.episode(scorer, [48])
        event = episode['events'][0]
        episode['events'] = [dict(event, step=step) for step in range(3)]
        episode['env_steps'] = 3
        batch = make_batch([episode], critic)
        batch['advantages'].zero_()
        snapshot = {key:value.clone() for key,value in batch.items()}
        before = scorer.layers[-1].weight.detach().clone()
        entropy_before = joint_entropy(scorer(event['features'],event['prior']),
            event['capacities'],event['legal'],event['temperature'],create_graph=False).item()
        report = update(scorer, critic, torch.optim.SGD(scorer.parameters(),lr=.1),
            torch.optim.SGD(critic.parameters(),lr=1e-4), batch, 'F',
            dict(ppo_epochs=1,minibatch_events=4,entropy=.01,entropy_sample_events=1))
        entropy_after = joint_entropy(scorer(event['features'],event['prior']),
            event['capacities'],event['legal'],event['temperature'],create_graph=False).item()
        self.assertEqual(report['minibatch_events'],2)
        self.assertEqual(report['task_count'],48)
        self.assertEqual(report['updates'],2)
        self.assertEqual(report['entropy_coefficient'],.01)
        self.assertEqual(report['entropy_events'],1)
        self.assertGreater(report['entropy_per_unit'],0)
        self.assertAlmostEqual(report['assignment_entropy']/5,report['entropy_per_unit'])
        self.assertLessEqual(report['old_new_logprob_max_error'],2e-4)
        self.assertGreater((scorer.layers[-1].weight.detach()-before).abs().max().item(),0)
        self.assertGreater(entropy_after,entropy_before)
        for key,value in snapshot.items():
            torch.testing.assert_close(batch[key],value)
        with self.assertRaisesRegex(ValueError,'only for joint'):
            update(scorer,critic,None,None,batch,'S',{'entropy':.01})


if __name__ == '__main__':
    unittest.main()
