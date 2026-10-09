"""Explicit synthetic fixtures test computation, never biological performance."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from blackout_rl.connectome.graph_artifact import GraphArtifact
from blackout_rl.connectome.sparse_ops import propagate
from blackout_rl.connectome.graph_controls import rewire
from blackout_rl.v7_1.model import GraphResidualHead,V7ResidualModel
from blackout_rl.mappo_v6 import V6Model,joint_distribution
from blackout_rl.v7_2.dynamics import WholeBrain,DynamicsConfig
from blackout_rl.v7_2.decoder import ReadoutModel,fixed_decode
from blackout_rl.v7_2.policy import DirectPolicy
from blackout_rl.v7_2.sensory import SemanticRetina
from blackout_rl.v7_2.training import update_readout
from blackout_rl.mappo_v6_training import update_v6
from blackout_rl.ppo import PPOConfig
from tests.test_ppo_training import small_model
from tests.test_mappo_v6 import all_observations,SyntheticEnv


def graph():
    ports=lambda name,indices:dict(name=name,indices=indices,pooling='mean',rationale='synthetic test only',confidence='test')
    return GraphArtifact(np.array(['9007199254740993','2','3','4']),np.array([0,1,2,0],np.int32),np.array([1,2,3,3],np.int32),np.array([2,3,4,1]),
        dict(synthetic=True,direction='presynaptic_to_postsynaptic',node_attributes=[dict(cell_type='test',side='left',transmitter='acetylcholine') for _ in range(4)],
        input_ports=[ports('input',[0])],output_ports=[ports('DNa02:L',[1]),ports('DNa02:R',[2]),ports('DNg100:L',[3])],
        retinal_mapping=[dict(index=0,source_id='9007199254740993',side='left',azimuth_degrees=0,elevation_degrees=0,confidence='test')]))


class GraphTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_roundtrip_and_source_refusal(self):
        g=graph()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'g.npz';g.save(p);loaded=GraphArtifact.load(p)
            self.assertEqual(loaded.node_ids[0],'9007199254740993')
            with self.assertRaises(ValueError):GraphArtifact.load(p,real=True)
            with p.open('ab') as f:f.write(b'x')
            with self.assertRaises(ValueError):GraphArtifact.load(p)

    def test_sparse_direction_and_dense_gradient(self):
        g=graph();s=torch.tensor(g.edge_src.astype('int64'));d=torch.tensor(g.edge_dst.astype('int64'))
        weight=torch.tensor([.2,.3,.4,.1],requires_grad=True);x=torch.randn(2,4,3,requires_grad=True)
        y=propagate(s,d,weight,x,4)
        dense=torch.zeros(4,4).index_put((d,s),weight)
        expected=torch.einsum('ds,bsc->bdc',dense,x)
        torch.testing.assert_close(y,expected)
        grads=torch.autograd.grad(y.square().sum(),(weight,x),retain_graph=True)
        refs=torch.autograd.grad(expected.square().sum(),(weight,x))
        for actual,reference in zip(grads,refs):torch.testing.assert_close(actual,reference)

    def test_head_determinism_gradients_keep(self):
        head=GraphResidualHead(12,graph());x=torch.randn(3,5,12)
        y=head(x);self.assertEqual(y.shape,(3,5,8))
        probs=torch.cat((torch.zeros(3,1),y.flatten(-2)),-1).softmax(-1)
        torch.testing.assert_close(probs[:,0],torch.full((3,),.9))
        torch.nn.init.normal_(head.output_head.weight,std=.1)
        torch.testing.assert_close(head(x),head(x))
        head(x).square().mean().backward()
        for p in (head.edge_weight,head.input_adapter.weight,head.output_head.weight):
            self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertEqual(head.edge_weight.numel(),4)
        self.assertFalse(torch.equal(head(x),head(x+1)))
        with self.assertRaises(ValueError):GraphResidualHead(12,graph(),rounds=1)

    def test_v6_mlp_equivalence_and_ppo_ratio(self):
        actor=small_model();torch.manual_seed(9);a=V6Model(copy.deepcopy(actor))
        torch.manual_seed(9);b=V7ResidualModel(copy.deepcopy(actor),variant='mlp')
        for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key])
        features=torch.randn(8,5,actor.model_config['hidden_dim']+49)
        mask=torch.ones(8,41,dtype=torch.bool);mask[:,4]=False
        da=joint_distribution(a.logits(features),mask,.03);db=joint_distribution(b.logits(features),mask,.03)
        action=da.sample();torch.testing.assert_close(da.log_prob(action),db.log_prob(action))
        torch.testing.assert_close((db.log_prob(action)-da.log_prob(action)).exp(),torch.ones(8))
        batch=dict(features=features,valid=mask,action=action,old_log_prob=da.log_prob(action).detach(),central=torch.randn(8,299),
                   old_value=torch.zeros(8),advantage=torch.randn(8),return_=torch.randn(8),exploration=.03)
        oa=torch.optim.Adam([p for p in a.parameters() if p.requires_grad]);ob=torch.optim.Adam([p for p in b.parameters() if p.requires_grad])
        torch.manual_seed(11);update_v6(a,oa,batch,PPOConfig(update_epochs=1,minibatch_size=8))
        torch.manual_seed(11);update_v6(b,ob,batch,PPOConfig(update_epochs=1,minibatch_size=8))
        for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key])


class DynamicsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_reference_tick_and_independent_slots(self):
        g=graph();cfg=replace(DynamicsConfig(),noise_hz=0.,tonic=0.,sensory_ema=0.,sensory_gain=1.)
        brain=WholeBrain(g,slots=2,config=cfg)
        brain.step(np.array([[1.],[0.]],np.float32))
        self.assertEqual(brain.spikes[0,0],1);self.assertFalse(brain.spikes[1].any())
        prev=brain.voltage.copy();spikes=brain.spikes.copy()
        brain.step(np.zeros((2,1),np.float32))
        expected=np.exp(-cfg.dt/cfg.tau)*prev+cfg.recurrent_gain*brain.matrix.toarray().dot(spikes.T).T
        # Temporal contrast produces additional receptor drive after the switch to black.
        expected[0,0]+=1
        expected[expected>=cfg.threshold]=0
        np.testing.assert_allclose(brain.voltage,expected,atol=1e-6)

    def test_checkpoint_replay_and_sensory_dependence(self):
        a=WholeBrain(graph(),config=replace(DynamicsConfig(),noise_hz=0.,tonic=0.))
        saved=a.state_dict();sequence=[]
        for _ in range(20):sequence.append(a.step(np.ones((1,1),np.float32)))
        a.load_state_dict(saved)
        for rates in sequence:np.testing.assert_array_equal(a.step(np.ones((1,1),np.float32)),rates)
        b=WholeBrain(graph(),config=replace(DynamicsConfig(),noise_hz=0.,tonic=0.))
        for _ in range(20):dark=b.step(np.zeros((1,1),np.float32))
        self.assertFalse(np.array_equal(sequence[-1],dark))

    def test_explicit_tick_cache_reset_and_state_replay(self):
        p=DirectPolicy(graph(),None,noise_seed=3);p.reset_episode_state('one',0)
        obs=all_observations()
        first=p.act(obs,episode_id='one',env_step_id=0,team_id=0)
        state=p.state_dict();second=p.act(obs,episode_id='one',env_step_id=0,team_id=0)
        self.assertEqual(p.brain.ticks,1);np.testing.assert_array_equal(first['unit_0'],second['unit_0'])
        next_action=p.act(obs,episode_id='one',env_step_id=1,team_id=0);self.assertEqual(p.brain.ticks,2)
        p.load_state_dict(state)
        np.testing.assert_array_equal(p.act(obs,episode_id='one',env_step_id=1,team_id=0)['unit_0'],next_action['unit_0'])
        with self.assertRaises(ValueError):p.features(obs,episode_id='two',env_step_id=2,team_id=0)
        with self.assertRaises(ValueError):p.features(obs,episode_id='one',env_step_id=3,team_id=0)
        p.reset_episode_state('two',1);self.assertEqual(p.brain.ticks,0)

    def test_readout_joint_ratio_and_update(self):
        m=ReadoutModel(3);z=torch.randn(8,5,3);d=m.distribution(z);action=d.sample()
        logp=torch.distributions.Categorical(logits=m.readout(z)).log_prob(action).sum(-1)
        torch.testing.assert_close(logp,d.log_prob(action))
        batch=dict(features=z,action=action,old_log_prob=logp.detach(),old_value=torch.zeros(8),central=torch.randn(8,299),
                   advantage=torch.randn(8),return_=torch.randn(8))
        before=m.readout.weight.detach().clone()
        update_readout(m,torch.optim.Adam(m.parameters(),lr=.01),batch,PPOConfig(update_epochs=1,minibatch_size=8))
        self.assertFalse(torch.equal(before,m.readout.weight))

    def test_sensor_wall_occlusion_and_world_y(self):
        sensor=SemanticRetina(width=3,height=8,fov_degrees=90)
        obs=all_observations()['unit_0'];obs=copy.deepcopy(obs)
        obs['graphic'][:]=0;obs['vector'][:2]=(.5,.5)
        # North wall hides a north battery. Image row decreases toward north.
        obs['graphic'][40,48,1]=1;obs['graphic'][32,48,6]=1
        a=sensor.render(obs,0,np.pi/2)
        obs['graphic'][32,48,6]=0
        np.testing.assert_array_equal(a,sensor.render(obs,0,np.pi/2))
        self.assertGreater(a.sum(),0)

if __name__=='__main__':unittest.main()

class RunnerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_collector_real_ppo_path_and_episode_boundary(self):
        from blackout_rl.v7_training import Collector
        from blackout_rl.v7_registry import load_config
        cfg=load_config('configs/v7/v7_1_flywire.yaml')
        cfg['training']['opponent_schedule']=[dict(until=100,probabilities=[1.,0.,0.])]
        model=V7ResidualModel(small_model(),graph())
        with tempfile.TemporaryDirectory() as d:
            collector=Collector(SyntheticEnv(),model,graph(),cfg,{'train':[1,2]},11,Path(d))
            batch,diagnostics=collector.collect(8)
            self.assertEqual(collector.global_step,8)
            self.assertEqual(batch['terminated'].sum().item(),2)
            self.assertEqual(collector.team,1)
            with torch.no_grad():
                dist=joint_distribution(model.logits(batch['features']),batch['valid'],batch['exploration'])
                torch.testing.assert_close((dist.log_prob(batch['action'])-batch['old_log_prob']).exp(),torch.ones(8))
            before=model.residual.output_head.weight.detach().clone()
            update_v6(model,torch.optim.Adam([p for p in model.parameters() if p.requires_grad]),batch,PPOConfig(update_epochs=1,minibatch_size=8))
            self.assertFalse(torch.equal(before,model.residual.output_head.weight))

    def test_resume_restores_checkpoint_and_discards_inflight_unity(self):
        from unittest.mock import patch
        from blackout_rl.v7_training import Collector,save_checkpoint,train
        from blackout_rl.v7_registry import load_config,source_fingerprint
        cfg=load_config('configs/v7/v7_1_flywire.yaml')
        cfg['training'].update(max_environment_steps=6,rollout_steps=2,save_every=2,opponent_schedule=[dict(until=100,probabilities=[1.,0.,0.])])
        factory=lambda *_:V7ResidualModel(small_model(),graph())
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'checkpoints').mkdir()
            model=factory();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['learning_rate'])
            collector=Collector(SyntheticEnv(),model,graph(),cfg,{'train':[1,2]},11,run);collector.collect(2)
            save_checkpoint(run/'checkpoints/latest.pt',model,opt,collector,cfg,11,source_fingerprint())
            with patch('blackout_rl.v7_training.ContractBlackOutEnv',SyntheticEnv),patch('blackout_rl.v7_training.make_model',factory):
                train(cfg,graph(),{'train':[1,2]},seed=11,run=run,resume=True)
            payload=torch.load(run/'checkpoints/latest.pt',weights_only=True)
            self.assertEqual(payload['global_step'],6)
            self.assertGreater(payload['episode_count'],collector.episode_count)
            with patch('blackout_rl.v7_training.make_model',factory),self.assertRaisesRegex(ValueError,'budget already complete'):
                train(cfg,graph(),{'train':[1,2]},seed=11,run=run,resume=True)

    def test_rewire_preserves_degrees(self):
        rng=np.random.default_rng(3);n=30
        keys=rng.choice(n*n,150,replace=False);src=keys//n;dst=keys%n
        original=GraphArtifact(np.arange(n).astype(str),src,dst,np.ones(150,dtype=int),
            dict(synthetic=True,direction='presynaptic_to_postsynaptic',node_attributes=[dict(side='unknown') for _ in range(n)],
                 input_ports=[dict(name='in',indices=list(range(n)),rationale='test',confidence='test')],
                 output_ports=[dict(name='out',indices=[0],rationale='test',confidence='test')]))
        control=rewire(original,4,swaps_per_edge=1)
        np.testing.assert_array_equal(np.bincount(original.edge_src,minlength=n),np.bincount(control.edge_src,minlength=n))
        np.testing.assert_array_equal(np.bincount(original.edge_dst,minlength=n),np.bincount(control.edge_dst,minlength=n))
        self.assertFalse(np.array_equal(original.edge_dst,control.edge_dst))
