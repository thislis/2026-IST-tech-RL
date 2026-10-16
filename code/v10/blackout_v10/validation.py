"""Small synthetic algorithm gates: never starts Unity or a persistent learner."""
import copy
import torch
from .policy import MyPolicy
from .value import Critic
from .benchmark import fixture
from .rewards import shape_episode, gae_episode


def validate_algorithms():
    with torch.random.fork_rng():
        torch.manual_seed(919)
        actor=MyPolicy(); v,g=fixture(); before=copy.deepcopy(actor.state_dict())
        assert sum(p.numel() for p in actor.parameters())==120905
        for arm, count in (('C0',29057),('C1',128769)):
            critic=Critic(arm)
            assert sum(p.numel() for p in critic.parameters())==count
            opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                loss=(critic(v[None],g[:1])-.2).square().mean()
                loss.backward(); opt.step()
            grads=[p.grad for n,p in critic.named_parameters() if n.startswith(('body.','encoder.')) and p.grad is not None]
            assert any(x.abs().sum()>0 for x in grads)
            output=critic(v[None],g[:1]).detach()
            critic.update_scale(torch.tensor([-2.,.4,1.]))
            torch.testing.assert_close(output,critic(v[None],g[:1]),atol=1e-6,rtol=1e-5)
        for k in before: torch.testing.assert_close(before[k],actor.state_dict()[k],atol=0,rtol=0)
        r=shape_episode([.2,-.2,.1],1.)
        _, ret=gae_episode(r,torch.zeros(3).numpy(),lam=1.)
        assert abs(float(ret[0])-.8)<1e-6
    return dict(passed=True,independent_encoders=True,popart_output_preserved=True,
                bounded_return=True,unity_started=False,persisted_learning_updates=0)
