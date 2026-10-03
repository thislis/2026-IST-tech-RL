import copy
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from blackout_rl.v8.contracts import ContractError,OutcomeEvent,StepContext,digest
from blackout_rl.v8.reward import RewardLedger
from blackout_rl.v8.environment import OutcomeBroker
from blackout_rl.v8.distributions import gated,flat,exact_factorization
from blackout_rl.v8.decoders import decode
from blackout_rl.v8.returns import gae
from blackout_rl.v8.checkpoints import CheckpointStore
from blackout_rl.v8.statistics import compare
from blackout_rl.v8.export import verify_capabilities
from blackout_rl.v8.telemetry import Telemetry


def event(**kwargs):
    values=dict(run_id='r',episode_id='ep',event_id='end',decision_id=5,unity_tick=20,game_time=.4,
      final_score_points=(100,42),winner_team=0,winner_source='engine',termination_reason='target_score',
      terminated=True,truncated=False,build_sha256='a'*64,protocol_sha256='b'*64)
    return OutcomeEvent(**dict(values,**kwargs))

class OutcomeTests(unittest.TestCase):
    def test_last_delta_once_and_reset_is_not_score(self):
        ledger=RewardLedger(0);ledger.reset('r','ep',(97,42));self.assertAlmostEqual(ledger.finish(event())['reward'],1.03)
        self.assertEqual(ledger.finish(event())['reward'],0)
        with self.assertRaises(ContractError):ledger.finish(event(event_id='another'))
        ledger.reset('r','next');self.assertEqual(ledger.observe((0,0),0),0)
        with self.assertRaises(ContractError):ledger.finish(event())
    def test_theft_draw_and_points_units(self):
        ledger=RewardLedger(0);ledger.reset('r','ep',(20,10));self.assertAlmostEqual(ledger.observe((17,10),0),-.03)
        self.assertEqual(ledger.finish(event(final_score_points=(17,17),winner_team=-1,termination_reason='game_timeout'))['terminal_bonus'],0)
        for kwargs in [dict(final_score_points=None),dict(winner_team=None),dict(winner_source='reward_sign'),dict(final_score_points=(1.,.42))]:
            with self.assertRaises(ContractError):event(**kwargs)
    def test_broker_late_and_conflicting_event(self):
        broker=OutcomeBroker('a'*64,'b'*64);self.assertTrue(broker.receive(event()));self.assertFalse(broker.receive(event()))
        with self.assertRaises(ContractError):broker.join('r','next',5)
        with self.assertRaises(ContractError):broker.join('r','ep',6)
        with self.assertRaises(ContractError):broker.receive(event(final_score_points=(100,43)))
        with self.assertRaises(ContractError):broker.receive(event(build_sha256='c'*64))
        self.assertEqual(broker.join('r','ep',5).winner_team,0)
    def test_null_is_not_draw_or_valid_truncation(self):
        with self.assertRaises(ContractError):event(terminated=False,truncated=True,winner_team=None)
        e=event(terminated=False,truncated=True,winner_team=None,final_observation_id='final:5',termination_reason='external_stop')
        self.assertTrue(e.truncated)

class DistributionTests(unittest.TestCase):
    def test_factorization_preserves_masked_legacy_policy(self):
        torch.manual_seed(2);logits=torch.randn(11,41,dtype=torch.float64);mask=torch.rand(11,40)>.3
        a=flat(logits[:,0],logits[:,1:],mask);b=exact_factorization(logits,mask)
        torch.testing.assert_close(a.probs,b.probs)
        torch.testing.assert_close(a.entropy(),b.entropy())
        self.assertTrue(torch.equal(decode(a,'joint_argmax_v1'),decode(b,'joint_argmax_v1')))
        actions=a.sample();torch.testing.assert_close(a.log_prob(actions),b.log_prob(actions))
    def test_empty_single_and_gradient(self):
        gate=torch.tensor([0.,2.,-2.],requires_grad=True);logits=torch.zeros(3,40,requires_grad=True)
        mask=torch.zeros(3,40,dtype=torch.bool);mask[1,4]=True;mask[2,:]=True
        d=gated(gate,logits,mask)
        self.assertEqual(float(d.probs[0,0]),1);self.assertEqual(float(d.entropy()[0]),0)
        self.assertEqual(float(d.log_prob(torch.tensor([0,5,0]))[0]),0)
        self.assertAlmostEqual(float(d.probs[1,5]),float(torch.sigmoid(gate[1])),places=6)
        (-d.log_prob(torch.tensor([0,5,0])).mean()-.01*d.entropy().mean()).backward()
        self.assertTrue(torch.isfinite(gate.grad).all());self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[1,0]),0)
    def test_gate_mass_entropy_and_decoder_fragmentation(self):
        mask=torch.ones(40,dtype=torch.bool);q=.6
        d=gated(torch.tensor(math.log(q/(1-q))),torch.zeros(40),mask)
        self.assertEqual(int(decode(d,'joint_argmax_v1')),0)
        self.assertEqual(int(decode(d,'gate_then_conditional_argmax_v1')),1)
        expected=-q*math.log(q)-(1-q)*math.log(1-q)+q*math.log(40)
        self.assertAlmostEqual(float(d.entropy()),expected,places=5)
        mask[1:]=False
        self.assertAlmostEqual(float(gated(torch.tensor(0.),torch.zeros(40),mask).q),.5)
        self.assertEqual(int(decode(gated(torch.tensor(0.),torch.zeros(40),mask),'gate_then_conditional_argmax_v1')),0)
    def test_initial_flat_gate_equal_only_without_masks(self):
        logits=torch.full((40,),math.log(.1/(40*.9)));mask=torch.ones(40,dtype=torch.bool)
        a=flat(torch.tensor(0.),logits,mask);b=gated(torch.tensor(math.log(.1/.9)),torch.zeros(40),mask)
        torch.testing.assert_close(a.probs,b.probs)
        mask[1:]=False
        self.assertLess(float(flat(torch.tensor(0.),logits,mask).q),float(gated(torch.tensor(math.log(.1/.9)),torch.zeros(40),mask).q))
    def test_behavior_logprob_ratio_and_invalid(self):
        d=gated(torch.tensor(-2.),torch.zeros(40),torch.ones(40,dtype=torch.bool));action=d.sample()
        self.assertEqual(float((d.log_prob(action)-d.log_prob(action)).exp()),1)
        with self.assertRaises(ContractError):gated(torch.tensor(float('nan')),torch.zeros(40),torch.ones(40,dtype=torch.bool))

class ReturnTests(unittest.TestCase):
    def test_terminal_truncation_and_rollout(self):
        a,r=gae(torch.ones(3),torch.zeros(3),torch.tensor([0.,5.,100.]),torch.tensor([False,False,True]),torch.tensor([False,True,False]),.9,1.)
        torch.testing.assert_close(a,torch.tensor([5.95,5.5,1.]));torch.testing.assert_close(a,r)
        a,_=gae(torch.tensor([1.]),torch.tensor([2.]),torch.tensor([3.]),torch.tensor([False]),torch.tensor([False]),.9,1.)
        self.assertAlmostEqual(float(a),1.7,places=5)

class CheckpointTests(unittest.TestCase):
    def test_immutable_and_legitimate_latest_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            store=CheckpointStore(directory);old=store.save(dict(global_step=0,model={'x':torch.tensor(1.)}))
            new=store.save(dict(global_step=2,model={'x':torch.tensor(2.)}),old)
            self.assertNotEqual(old,new);self.assertEqual(store.load()[0]['parent_sha256'],old)
            self.assertEqual(store.load(old)[0]['global_step'],0)
            (Path(directory)/(old+'.pt')).write_bytes(b'corrupt')
            with self.assertRaises(ContractError):store.load(new)

class StatisticsTests(unittest.TestCase):
    def test_shared_map_cancels_fixed_map_effect(self):
        base=np.arange(8).reshape(1,4,2)/10
        result=compare(np.repeat(base+0.1,3,axis=0),base,reference_fixed=True,replicates=100,seed=42)
        np.testing.assert_allclose(result['intervals']['crossed'],[.1,.1],atol=1e-12)
    def test_run_rows_are_not_paired_by_seed_number(self):
        a=np.stack([np.zeros((8,2)),np.ones((8,2))]);b=a.copy()
        result=compare(a,b,replicates=1000,seed=2)
        self.assertEqual(result['delta'],0)
        self.assertLess(result['intervals']['crossed'][0],0);self.assertGreater(result['intervals']['crossed'][1],0)
    def test_no_fake_fixed_runs_or_missing_cells(self):
        with self.assertRaises(ContractError):compare(np.zeros((2,4,2)),np.zeros((2,4,2)),reference_fixed=True)
        with self.assertRaises(ContractError):compare(np.full((2,4,2),np.nan),np.zeros((1,4,2)))

class ExportAndTelemetryTests(unittest.TestCase):
    def test_official_unknown_capability_fails_closed(self):
        with self.assertRaises(ContractError):verify_capabilities({'status':'unresolved'})
    def test_random_and_trigger_windows_bound_memory(self):
        with tempfile.TemporaryDirectory() as d:
            t=Telemetry(d,window=2,control_probability=1,max_pending=1)
            for i in range(8):t.observe(dict(step=i),['outcome'] if i==4 else [])
            self.assertGreater(t.counters['dropped_windows'],0);self.assertLessEqual(len(t.before),2);t.close()
            self.assertTrue((Path(d)/'windows.jsonl').exists())
