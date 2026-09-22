"""F1 PPO over frozen whole-brain features; joint log probability sums slots."""
import torch
from torch import nn


def update_readout(model, optimizer, batch, config):
    config.validate()
    advantage = batch['advantage']
    advantage = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
    totals = dict(policy_loss=0.,value_loss=0.,entropy=0.,approximate_kl=0.,gradient_norm=0.)
    updates, early = 0, False
    for _ in range(config.update_epochs):
        for ix in torch.randperm(len(advantage)).split(config.minibatch_size):
            distribution = model.distribution(batch['features'][ix])
            log_ratio = distribution.log_prob(batch['action'][ix])-batch['old_log_prob'][ix]
            ratio = log_ratio.exp()
            kl = ((ratio-1)-log_ratio).mean()
            if not torch.isfinite(kl):
                raise RuntimeError('non-finite readout PPO ratio')
            if config.target_kl is not None and kl.detach() > config.target_kl:
                early = True; break
            policy_loss = torch.maximum(-advantage[ix]*ratio,-advantage[ix]*ratio.clamp(1-config.clip_coef,1+config.clip_coef)).mean()
            value = model.critic(batch['central'][ix]).squeeze(-1)
            error = (value-batch['return_'][ix]).square()
            if config.value_clip_coef is not None:
                clipped = batch['old_value'][ix]+(value-batch['old_value'][ix]).clamp(-config.value_clip_coef,config.value_clip_coef)
                error = torch.maximum(error,(clipped-batch['return_'][ix]).square())
            value_loss = .5*error.mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss+config.value_loss_coef*value_loss-config.entropy_coef*entropy
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite readout PPO loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm,error_if_nonfinite=True)
            optimizer.step(); updates += 1
            for key,val in dict(policy_loss=policy_loss,value_loss=value_loss,entropy=entropy,approximate_kl=kl,gradient_norm=norm).items():
                totals[key] += float(val.detach())
        if early:
            break
    return {**{k:v/max(1,updates) for k,v in totals.items()}, 'minibatches':updates,'early_stopped':early}
