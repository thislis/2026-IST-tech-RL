import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from blackout_v11.critic import Critic
from blackout_v11.demonstrations import initialize_from_demonstrations
from blackout_v11.io import atomic_torch, cpu_state, file_hash
from blackout_v11.policy import MyPolicy, fixture
from blackout_v11.recurrent_ppo import optimizers, update
from blackout_v11.rewards import shape_episode, gae_episode
from blackout_v11.scheduler import DeficitScheduler
from blackout_v11.targets import commit_targets
from blackout_v11.teachers import label_choices
from blackout_v11.trajectory import Episode, EpisodeWriter, WaveBatch


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_variable_duration_returns(self):
        rewards = shape_episode([.1, -.2, .05], [2, 7, 3], 1.)
        self.assertAlmostEqual(float(rewards.sum()), .9, places=6)
        advantages, _ = gae_episode([.1, .8], [.2, .3], [2, 3], gamma=1., lam=.9)
        np.testing.assert_allclose(advantages, [.2+.9**2*.5, .5], atol=1e-6)
        discounted = shape_episode([.1, -.2], [2, 3], -1, gamma=.9)
        self.assertAlmostEqual(float(discounted[0]+.9**2*discounted[1]), -.9**4-.1, places=6)

    def test_step_deficit_uses_duration(self):
        catalog = {f: [{"kind": f}] for f in ("planner", "scripted", "self_play", "direct")}
        scheduler = DeficitScheduler(catalog)
        for _ in range(100):
            task = scheduler.next(11)
            duration = 1000 if task["family"] == "planner" else 10000
            scheduler.observe(task["family"], task["side"], duration)
        report = scheduler.report()
        self.assertGreater(report["families"]["planner"]["episodes"], report["families"]["scripted"]["episodes"])
        self.assertLess(abs(report["families"]["planner"]["actual_fraction"]-.4), .1)

    def test_complete_recurrent_wave_and_teacher_labels(self):
        torch.manual_seed(7)
        policy, critic = MyPolicy().eval(), Critic()
        policy.configure("F", "sample")
        v, g = fixture()
        g = g.clone()
        g[:, 0, 0, 0] = .3  # A lossy semantic argmax must not be used here.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            snapshot = base/"behavior.pt"
            atomic_torch(snapshot, {"policy_state": cpu_state(policy), "critic_state": cpu_state(critic), "critic_arm": "C1"})
            checksum = file_hash(snapshot)
            writer = EpisodeWriter(base/"trajectory", chunk_size=3)
            state = policy.initial_state(1)
            with torch.no_grad():
                for step in range(18):
                    current = v.clone()
                    current[..., 95] -= step/21000
                    out = policy.step(current, g, state)
                    state = out["state"]
                    if bool(out["event"][0]):
                        labels = label_choices(policy, current, g, out["decision_state"], out["action"], out["base_action"])
                        teacher = policy.evaluate_decision(current, g, out["decision_state"], labels)
                        self.assertTrue(bool(torch.isfinite(teacher["log_prob"]).all()))
                        self.assertGreater(float(teacher["log_prob"][0]), -1e6)
                        writer.add(step=step, vectors=current[0].numpy(), graphic=g[0].numpy(),
                            decision_state=out["decision_state"], choices=out["choices"],
                            log_prob=float(out["log_prob"][0]), potential=0.)
            writer.commit(1., 1., .9975, {"natural_terminal": True, "env_steps": 18,
                          "policy_sha256": checksum, "on_policy": True})
            episode = Episode(base/"trajectory", validated=True)
            self.assertEqual(float(episode.read([0])["graphic"][0, 0, 0, 0]), float(g[0, 0, 0, 0]))
            commit_targets([base/"trajectory"], snapshot, checksum, {"value_batch": 4}, "cpu")
            wave = WaveBatch([base/"trajectory"])
            before = {k: x.clone() for k, x in policy.state_dict().items()}
            ao, co = optimizers(policy, critic, {})
            result = update(policy, critic, ao, co, wave,
                {"epochs": 1, "burn_in": 2, "sequence_length": 3, "minibatch_sequences": 2, "hard_kl": 10},
                np.random.default_rng(19))
            self.assertLess(result["ratio_audit"]["max_log_prob_error"], 2e-3)
            self.assertTrue(any(not torch.equal(before[k], x) for k, x in policy.state_dict().items()))
            actor_after = {k: x.clone() for k, x in policy.state_dict().items()}
            co.zero_grad()
            critic(v, g).square().mean().backward()
            co.step()
            self.assertTrue(all(torch.equal(actor_after[k], x) for k, x in policy.state_dict().items()))

    def test_popart_preserves_values(self):
        critic = Critic()
        v, g = fixture()
        before = critic(v, g).detach().clone()
        critic.update_scale(torch.tensor([2., -3., 7.]))
        torch.testing.assert_close(critic(v, g), before)

    def test_common_teacher_encoder_initialization(self):
        v, g = fixture()
        source = MyPolicy()
        source.configure("F", "greedy")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"episode"
            writer = EpisodeWriter(path)
            with torch.no_grad():
                out = source.step(v, g, source.initial_state(1), deterministic=True)
                labels = label_choices(source, v, g, out["decision_state"], out["action"], out["base_action"])
                writer.add(step=0, vectors=v[0].numpy(), graphic=g[0].numpy(), decision_state=out["decision_state"],
                           choices=out["choices"], log_prob=float(out["log_prob"][0]), potential=0,
                           teacher_choices=labels, on_policy=False)
            writer.commit(1, 1, .9975, {"natural_terminal": True, "env_steps": 4, "policy_sha256": "fixture"})
            manifest = dict(complete=True, episodes=[dict(path=str(path), split="train")], env_steps=4, label_source="fixture")
            results = {}
            for arm in ("G", "F", "D"):
                torch.manual_seed(17)
                p = MyPolicy()
                p.configure(arm)
                initialize_from_demonstrations(p, manifest, {"teacher_epochs": 1, "teacher_batch": 1}, "cpu")
                self.assertEqual(p.arm, arm)
                results[arm] = cpu_state(p)
            for name in results["F"]:
                if name != "settings":
                    torch.testing.assert_close(results["G"][name], results["F"][name], rtol=0, atol=0)
                    if not name.startswith("direct_head."):
                        torch.testing.assert_close(results["D"][name], results["F"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
