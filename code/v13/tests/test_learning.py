import copy
import unittest
import torch
from blackout_v13.actor import Actor
from blackout_v13.policy import Policy
from blackout_v13.critic import Critic
from blackout_v13.export import synthetic_observation
from blackout_v13.rollout import prepare_chunk,advantages,manager_returns
from blackout_v13.ppo import Learner,worker_group,manager_group
from blackout_v13.assignment import joint_log_prob
from blackout_v13.io import ROOT,read_json


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        self.cfg=read_json(ROOT/'code/v13/configs/default.json')
        self.cfg.update(ppo_epochs=1,sequence_minibatch=1,event_period=2)

    def chunk(self,phase='worker',prefix_steps=0):
        p=Policy(self.cfg);critic=Critic();actor=Actor(p,critic,self.cfg)
        v,g=synthetic_observation()
        for j in range(prefix_steps):
            v[:,95]=1-j*.02/420;actor.step(v,g,phase=phase)
        snap=actor.prefix_snapshot();rows=[]
        for j in range(5):
            v[:,95]=1-(prefix_steps+j)*.02/420
            _,row=actor.step(v,g,phase=phase);row['reward']=.1*(j%2);rows.append(row)
        return p,critic,dict(**snap,rows=rows,bootstrap=.2,version=0,phase=phase)

    def test_frozen_worker_log_probability(self):
        p,critic,chunk=self.chunk(prefix_steps=4)
        seq=prepare_chunk(chunk,self.cfg['lags'],'cpu');hidden=seq['hidden']
        with torch.no_grad():
            encoded=p.worker.encode_lagged(seq['local'],seq['vector'],seq['lags'])
            for j in range(len(encoded)):
                hidden=hidden.masked_fill(seq['reset'][j,:,None],0)
                logits,hidden=p.worker.step_encoded(encoded[j],hidden)
                if j>=seq['burn']:
                    row=seq['rows'][j-seq['burn']]
                    lp=torch.distributions.Categorical(logits=logits).log_prob(row['action']).sum()
                    self.assertLess(abs(float(lp-row['old_log_prob'])),2e-4)

    def test_only_selected_actor_updates(self):
        for phase in ('worker','manager'):
            p,c,chunk=self.chunk(phase)
            other='manager' if phase=='worker' else 'worker'
            before=copy.deepcopy(getattr(p,other).state_dict())
            selected=copy.deepcopy(getattr(p,phase).state_dict())
            learner=Learner(p,{'worker':copy.deepcopy(c),'manager':copy.deepcopy(c)},self.cfg,'cpu')
            result=learner.update([chunk],phase)
            self.assertGreater(result['updates'],0)
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in getattr(p,other).state_dict().items()))
            self.assertTrue(any(not torch.equal(v,selected[k]) for k,v in getattr(p,phase).state_dict().items()))

    def test_gae_terminal_and_duration(self):
        adv,ret=advantages([1.,2.],[.5,.6],[False,True],100.,1.,1.,[4,2])
        torch.testing.assert_close(ret,torch.tensor([3.,2.]))
        adv,ret=advantages([1.],[.5],[False],2.,.9,.8,[3])
        self.assertAlmostEqual(ret.item(),1.+.9**3*2,places=6)

    def test_mixed_versions_rejected(self):
        p,c,chunk=self.chunk();chunk['version']=9
        learner=Learner(p,{'worker':c,'manager':Critic()},self.cfg,'cpu')
        with self.assertRaises(ValueError):learner.update([chunk],'worker')

    def test_gru_formula(self):
        p=Policy();x=torch.randn(5,128);h=torch.randn(5,64)
        _,actual=p.worker.step_encoded(x,h)
        torch.testing.assert_close(actual,p.worker.gru(x,h))

    def test_batched_sequences_with_unequal_burn(self):
        p,c,a=self.chunk(prefix_steps=4)
        _,_,b=self.chunk(prefix_steps=0)
        b['rows']=b['rows'][:3]
        sequences=[prepare_chunk(x,self.cfg['lags'],'cpu') for x in (a,b)]
        with torch.no_grad():
            batched=worker_group(p.worker,sequences)
            for seq,actual in zip(sequences,batched):
                expected=worker_group(p.worker,[seq])[0]
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

    def test_batched_manager_joint_probability(self):
        p,c,chunk=self.chunk('manager',prefix_steps=3)
        seq=prepare_chunk(chunk,self.cfg['lags'],'cpu')
        seq['indices']=manager_returns(chunk,self.cfg)[0]
        empty={**seq,'indices':[]}
        actual=manager_group(p,[seq,empty,seq],'cpu')
        expected=[]
        for j in seq['indices']:
            event=seq['rows'][j]['manager']
            score=p.manager.from_lags(event['features'],seq['lags'][seq['burn']+j])
            expected.append(joint_log_prob(score,event['assignment'],event['caps'],event['legal']))
        torch.testing.assert_close(actual[0],torch.stack(expected),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(actual[0],torch.stack([seq['rows'][j]['manager']['old_log_prob'] for j in seq['indices']]),
                                   atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(actual[0],actual[2]);self.assertIsNone(actual[1])

    def test_missing_worker_probability_replay(self):
        p=Policy(self.cfg);actor=Actor(p,Critic(),self.cfg);v,g=synthetic_observation()
        rows=[];snapshot=actor.prefix_snapshot()
        for step in range(4):
            v[:,95]=1-step*.02/420
            vv=v.clone();gg=g.clone()
            if step in (1,2):vv[2]=0;gg[2]=0
            _,row=actor.step(vv,gg);rows.append(row)
        chunk=dict(**snapshot,rows=rows,bootstrap=0.,version=0,phase='worker')
        seq=prepare_chunk(chunk,self.cfg['lags'],'cpu')
        with torch.no_grad():
            logits=worker_group(p.worker,[seq])[0]
            actions=torch.stack([r['action'] for r in rows])
            logp=(torch.distributions.Categorical(logits=logits).log_prob(actions)*seq['present']).sum(-1)
        torch.testing.assert_close(logp,torch.stack([r['old_log_prob'] for r in rows]),atol=2e-4,rtol=2e-5)


if __name__=='__main__':unittest.main()
