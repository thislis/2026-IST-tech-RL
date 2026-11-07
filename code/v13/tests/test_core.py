import itertools
import unittest
import numpy as np
import torch
from blackout_v13.history import History,advance
from blackout_v13.geometry import Geometry
from blackout_v13.assignment import Assignment,joint_log_prob,log_partition
from blackout_v13.policy import Policy
from blackout_v13.contracts import initial_state


def observation():
    v=torch.zeros(5,96)
    e=torch.zeros(10,9)
    e[:5,0]=.08;e[:5,1]=torch.linspace(.78,.94,5);e[:5,2]=1
    e[5:,0]=torch.linspace(.78,.94,5);e[5:,1]=.08;e[5:,2]=-1
    e[:,3]=1
    v[:,:90]=e.flatten();v[:,90]=1;v[:,95]=1
    semantic=torch.zeros(11,24,24)
    for y,x in [(21,1),(16,4),(12,5),(8,8),(4,12)]:semantic[2,y,x]=1
    for y,x in [(18,4),(15,8),(9,12),(3,16)]:semantic[6,y,x]=1
    g=semantic.flip(-2).repeat_interleave(4,-2).repeat_interleave(4,-1)[None].repeat(5,1,1,1)
    return v,g


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_history_exact_lags(self):
        hist=torch.zeros(5,64,128)
        for t in range(201):
            token=torch.full((5,128),float(t));token[:,121]=1
            result=History()(token,hist)
            for j,lag in enumerate((0,4,8,16,32,64)):
                if t>=lag:self.assertEqual(result[0,j,0].item(),t-lag)
                else:self.assertEqual(result[0,j,121].item(),0)
            hist=advance(token,hist)

    def test_grid_matches_dijkstra(self):
        from blackout_v12.navigation import distance_field
        rng=np.random.default_rng(31)
        walk=rng.random((24,24))>.23;walk[0,0]=True
        seeds=torch.zeros(1,576,dtype=torch.bool);seeds[0,0]=True
        actual,_=Geometry()(torch.from_numpy(walk.reshape(1,-1)),seeds)
        expected,_=distance_field(walk,[0]);expected=np.minimum(expected,10000.)
        np.testing.assert_allclose(actual[0].numpy(),expected,atol=2e-5,rtol=1e-6)

    def test_exact_capacity_distribution(self):
        torch.manual_seed(5)
        scores=torch.randn(5,4);legal=torch.ones_like(scores,dtype=torch.bool);caps=torch.tensor([2,1,1,1])
        valid=[a for a in itertools.product(range(4),repeat=5) if all(a.count(j)<=caps[j] for j in range(4))]
        values=torch.stack([sum(scores[i,j] for i,j in enumerate(a)) for a in valid])
        self.assertTrue(torch.allclose(log_partition(scores,caps,legal),values.logsumexp(0),atol=2e-6))
        chosen=Assignment()(scores,legal,caps)
        self.assertAlmostEqual(sum(scores[i,j].item() for i,j in enumerate(chosen)),values.max().item(),places=5)
        probabilities=torch.stack([joint_log_prob(scores,torch.tensor(a),caps,legal).exp() for a in valid])
        self.assertAlmostEqual(probabilities.sum().item(),1.,places=5)

    def test_script_and_state_parity(self):
        torch.manual_seed(4)
        model=Policy().eval();script=torch.jit.script(model)
        v,g=observation();state=initial_state();other=initial_state()
        with torch.no_grad():
            for step in range(3):
                v[:,95]=1-step*.02/420
                a=model(v,g,torch.arange(5),*state)
                b=script(v,g,torch.arange(5),*other)
                for x,y in zip(a,b):torch.testing.assert_close(x,y)
                state=a[1:];other=b[1:]
        self.assertEqual(state[0].shape,(5,64,128))
        self.assertTrue(torch.isfinite(a[0]).all())

    def test_missing_slots_preserve_state_in_actor_and_graph(self):
        from blackout_v13.actor import Actor
        from blackout_v13.critic import Critic
        torch.manual_seed(19);model=Policy().eval();actor=Actor(model,Critic(),{})
        v,g=observation();actor.step(v,g,deterministic=True)
        before=tuple(x.clone() for x in actor.state)
        v[2]=0;g[2]=0;v[[0,1,3,4],95]-=.02/420
        with torch.no_grad():expected=model(v,g,torch.arange(5),*before)
        actual,_=actor.step(v,g,deterministic=True)
        np.testing.assert_allclose(actual,expected[0].numpy())
        for state,old,reference in zip(actor.state,before,expected[1:]):
            torch.testing.assert_close(state[2],old[2]);torch.testing.assert_close(state,reference)

    def test_team_reflection(self):
        torch.manual_seed(19);model=Policy().eval();v,g=observation()
        reflected=v.clone();entities=v[0,:90].reshape(10,9).clone()
        entities=entities[torch.tensor([5,6,7,8,9,0,1,2,3,4])]
        entities[:,:2]=entities[:,:2].flip(-1)
        reflected[:,:90]=entities.flatten()
        mirror=g.transpose(-1,-2).flip((-1,-2))
        with torch.no_grad():
            a=model(v,g,torch.arange(5),*initial_state())
            b=model(reflected,mirror,torch.arange(5),*initial_state())
        torch.testing.assert_close(a[0],b[0].flip(-1))
        torch.testing.assert_close(a[4],b[4])

    def test_resource_capacity_and_missing_reservation(self):
        p=Policy();v,g=observation();state=initial_state()
        decoded=p.prepare(v,g,*state)
        _,sem,pos,enemies,cls,holding,present,side,h,hidden,t,token,clear,event=decoded
        f,legal,caps,goals,dest,kinds,_=p.candidates(sem,pos,enemies,cls,holding,t,token,present)
        resource=int(goals[0,torch.where(legal[0,:10])[0][0]])
        t[2,0]=resource;t[2,2]=0;t[2,11]=1;present[2]=False
        f,legal,caps,goals,dest,kinds,_=p.candidates(sem,pos,enemies,cls,holding,t,token,present)
        self.assertFalse(bool((legal[:,:10]&(goals[:,:10]==resource)).any()))
        choice=p.assignment(torch.zeros(5,32),legal,caps)
        self.assertTrue(bool(legal.gather(1,choice[:,None]).all()))
        usage=torch.nn.functional.one_hot(choice,32).sum(0)
        self.assertTrue(bool((usage<=caps).all()))


if __name__=='__main__':unittest.main()
