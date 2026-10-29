"""Learned task control on public synthetic observations; never starts Unity."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from blackout_v12.contracts import observe
from blackout_v12.features import LEARNED_FEATURE_SCHEMA
from blackout_v12.export import export_policy, load_export
from blackout_v12.model import TaskScorer
from blackout_v12.navigation import Navigation, HUNTER_SHRINE, center
from blackout_v12.policy import MyPolicy
from blackout_v12.state import TeamState
from blackout_v12.tasks import all_tasks, generate_candidates
from test_policy import fixture


LEARNED = {'policy_mode':'learned', 'event_period':1}


def prefer_feature(policy, feature):
    """A real scorer whose learned weights prefer a supplied public feature."""
    with torch.no_grad():
        for parameter in policy.scorer.parameters():
            parameter.zero_()
        policy.scorer.layers[0].weight[0,feature] = 1.
        policy.scorer.layers[2].weight[0,0] = 1.
        policy.scorer.layers[4].weight[0,0] = 5.


class LearnedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_weights_load_and_prior_cannot_change_learned_scores(self):
        torch.manual_seed(72)
        original = TaskScorer()
        with torch.no_grad():
            original.layers[-1].weight.normal_(0., .1)
        learned = TaskScorer(LEARNED)
        learned.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in learned.parameters()),6273)
        features = torch.randn(5,7,64)
        torch.testing.assert_close(learned(features,torch.full((5,7),20.)),
                                   learned(features,torch.full((5,7),-20.)))
        torch.testing.assert_close(learned(features),original(features))

    def test_hold_and_evade_follow_network_without_enemy_override(self):
        vector,graphic = fixture()
        entities = vector[:,:90].reshape(5,10,9)
        entities[:,5,:2] = entities[:,0,:2] + np.array([.025,0],np.float32)
        policy = MyPolicy(LEARNED)
        prefer_feature(policy,20)  # task_hold
        held = policy.step(vector,graphic,deterministic=True)
        self.assertTrue(all(u.kind=='hold' for u in policy.runtime.units))
        np.testing.assert_array_equal(held['action'].numpy(),np.zeros((5,2)))
        self.assertEqual(held['decision']['schema'],LEARNED_FEATURE_SCHEMA['id'])
        prefer_feature(policy,39)  # is_evade
        vector[:,95] -= .02/420
        evaded = policy.step(vector,graphic,deterministic=True)
        self.assertTrue(all(u.kind=='evade' for u in policy.runtime.units))
        self.assertLess(float(evaded['action'][0,0]),0.)
        self.assertEqual(int(evaded['decision']['locked'].sum()),0)
        self.assertTrue(torch.isfinite(evaded['decision']['old_log_prob']))

    def test_transform_can_be_reconsidered_and_threat_does_not_veto_it(self):
        vector,graphic = fixture()
        policy = MyPolicy(LEARNED)
        prefer_feature(policy,18)  # task_transform-Hunter
        policy.step(vector,graphic,deterministic=True)
        self.assertTrue(all(u.kind=='transform-Hunter' for u in policy.runtime.units))
        for unit in policy.runtime.units:
            unit.commitment = 999
        entities = vector[:,:90].reshape(5,10,9)
        entities[:,5,:2] = entities[:,0,:2] + np.array([.025,0],np.float32)
        vector[:,95] -= .02/420
        still = policy.step(vector,graphic,deterministic=True)
        hunter = still['decision']['task_ids'].index('transform-Hunter')
        self.assertTrue(still['decision']['legal'][:,hunter].all())
        self.assertFalse(still['decision']['locked'].any())
        prefer_feature(policy,20)
        vector[:,95] -= .02/420
        policy.step(vector,graphic,deterministic=True)
        self.assertTrue(all(u.kind=='hold' for u in policy.runtime.units))

    def test_all_legal_support_and_only_physical_class_caps(self):
        vector,graphic = fixture()
        obs = observe(vector,graphic)
        state = TeamState()
        state.update(obs,LEARNED)
        state.item_age[:] = 0  # Public remembered resources, no prior shortlist.
        candidates = generate_candidates(obs,state,Navigation(),{**LEARNED,'max_tasks':10})
        self.assertGreater(len(candidates.tasks),24)
        self.assertEqual(candidates.diagnostics['candidate_support'],'all_legal')
        self.assertTrue(candidates.legal.any(0).all())
        np.testing.assert_array_equal(candidates.prior,np.zeros_like(candidates.prior))
        for kind in ('hold','evade'):
            for slot in range(5):
                task = next(j for j,t in enumerate(candidates.tasks)
                            if t.kind==kind and t.private_slot==slot)
                self.assertEqual(np.flatnonzero(candidates.legal[:,task]).tolist(),[slot])
        obs.classes[:2] = 1
        obs.classes[2] = 2
        tasks = {t.id:t for t in all_tasks(obs,state,LEARNED)}
        self.assertEqual(tasks['transform-Hunter'].capacity,3)
        self.assertEqual(tasks['transform-Carrier'].capacity,0)
        self.assertTrue(all(t.capacity==5 for t in tasks.values() if t.kind in ('intercept','defend')))
        obs.holding[3] = 1
        loaded = generate_candidates(obs,state,Navigation(),LEARNED)
        hunter = next(j for j,t in enumerate(loaded.tasks) if t.kind=='transform-Hunter')
        self.assertTrue(loaded.legal[3,hunter])
        walk = Navigation().walkability(state,3,0,1,'transform-Hunter',allow_loaded_transform=True)
        self.assertTrue(walk.reshape(-1)[list(HUNTER_SHRINE)].all())

    def test_long_no_progress_is_observed_without_timeout_action(self):
        vector,graphic = fixture()
        entities = vector[:,:90].reshape(5,10,9)
        entities[:,0,3] = 0
        entities[:,0,4] = 1
        entities[:,0,:2] = center(505)
        state = TeamState()
        state.update(observe(vector,graphic),LEARNED)
        unit = state.units[0]
        unit.kind,unit.phase,unit.task_id = 'deliver','return','deliver:505'
        unit.destination = 505
        unit.previous_action[:] = 1.
        for _ in range(130):
            vector[:,95] -= .02/420
            state.update(observe(vector,graphic),{**LEARNED,'delivery_wait_steps':1,'stuck_steps':1})
        self.assertEqual(unit.task_id,'deliver:505')
        self.assertGreater(unit.no_progress,100)
        self.assertFalse(unit.failed_stores)
        self.assertFalse(unit.blocked)
        self.assertNotIn('storage_failed',state.event_reasons)
        self.assertNotIn('stuck',state.event_reasons)
        first = generate_candidates(observe(vector,graphic),state,Navigation(),LEARNED)
        self.assertGreater(float(first.features[0,0,37]),0.)
        self.assertLess(float(first.features[0,0,11]),1.)
        unit.age = 10000
        later = generate_candidates(observe(vector,graphic),state,Navigation(),LEARNED)
        self.assertGreater(float(later.features[0,0,11]),float(first.features[0,0,11]))
        # Observed progress resets the history but leaves the chosen task alone.
        unit.destination = 510
        vector[:,95] -= .02/420
        state.update(observe(vector,graphic),LEARNED)
        vector[:,95] -= .02/420
        state.update(observe(vector,graphic),LEARNED)
        self.assertGreater(unit.no_progress,0)
        entities[:,0,0] += .001
        vector[:,95] -= .02/420
        state.update(observe(vector,graphic),LEARNED)
        self.assertEqual(unit.no_progress,0)
        self.assertEqual(unit.task_id,'deliver:505')

    def test_portable_export_preserves_learned_dynamic_support(self):
        vector,graphic = fixture()
        policy = MyPolicy(LEARNED)
        prefer_feature(policy,20)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'submission'
            export_policy(policy,directory)
            loaded = load_export(directory)
            counts = []
            for step in range(2):
                if step:
                    row,x = (23-14)*4,15*4
                    graphic[:,:,row:row+4,x:x+4] = 0.
                    graphic[:,6,row:row+4,x:x+4] = 1.
                    vector[:,95] -= .02/420
                expected = policy.step(vector,graphic,deterministic=True)
                observations = {f'unit_{i}':dict(vector=vector[i].copy(),
                    graphic=graphic[i].transpose(1,2,0).copy()) for i in range(5)}
                actual = loaded.act(observations)
                np.testing.assert_allclose(expected['action'].numpy(),np.stack(list(actual.values())))
                counts.append(expected['decision']['features'].shape[1])
                self.assertEqual(loaded._net.last_decision['features'].shape[1],counts[-1])
            self.assertNotEqual(*counts)

    def test_runtime_state_remains_explicit_and_accepts_legacy_fields(self):
        vector,graphic = fixture()
        legacy = MyPolicy()
        legacy.step(vector,graphic,deterministic=True)
        snapshot = legacy.runtime_state()
        for unit in snapshot['units']:
            for field in ('no_progress','no_progress_seconds','progress_goal','progress_distance'):
                unit.pop(field)
        restored = MyPolicy(LEARNED)
        restored.load_runtime_state(snapshot)
        self.assertEqual(restored.runtime.units[0].no_progress,0)
        restored.load_state_dict(legacy.state_dict(),strict=True)


if __name__ == '__main__':
    unittest.main()
