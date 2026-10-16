"""Agent-ratio MAPPO, independent encoders, frozen targets and post-step KL."""
import copy
import time
import numpy as np
import torch
from torch.nn import functional as F


def optimizers(policy, critic, cfg):
    if {id(p) for p in policy.parameters()} & {id(p) for p in critic.parameters()}:
        raise ValueError('actor and critic share parameters')
    return (torch.optim.Adam(policy.parameters(), lr=cfg['actor_lr']),
            torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr']))


@torch.no_grad()
def audit(policy, row):
    logs = policy.team_log_probs(row['vectors'], row['graphic'])
    old = row['old_distribution']
    exact = (old.exp()*(old-logs)).sum(-1).mean()
    lr = logs.gather(-1, row['actions'][..., None]).squeeze(-1)-row['log_probs']
    approximate = ((lr.exp()-1)-lr).mean()
    return torch.stack((exact, approximate)).cpu().tolist()


def update(policy, critic, actor_opt, critic_opt, batch, cfg, rng, arm, progress=lambda **_: None):
    start = time.monotonic(); device = next(policy.parameters()).device
    old_values = np.concatenate([e.arrays['values'][:e.length] for e in batch.episodes])
    variance = float(np.var(batch.returns))
    ev = None if variance < 1e-12 else float(1-np.var(batch.returns-old_values)/variance)
    wave_actor = copy.deepcopy(policy.state_dict()); wave_optimizer = copy.deepcopy(actor_opt.state_dict())
    policy.train(); critic.train()
    advantages = (batch.advantages-batch.advantages.mean())/max(float(batch.advantages.std()), 1e-8)
    old_scale = float(critic.scale.detach().cpu())
    critic.update_scale(torch.from_numpy(batch.returns).to(device))
    scale_ratio = old_scale/float(critic.scale.detach().cpu())
    for param in (critic.output.weight, critic.output.bias):
        for key, power in (('exp_avg', 1), ('exp_avg_sq', 2), ('max_exp_avg_sq', 2)):
            state = critic_opt.state.get(param, {})
            if key in state: state[key].mul_(scale_ratio**power)
    ids = np.unique(np.linspace(0, len(batch)-1, min(cfg['audit_scenes'], len(batch))).astype(int))
    probe = batch.minibatch(ids, device)
    initial = {k: p.detach().clone() for k, p in policy.named_parameters()
               if k in ('patch.weight', 'cross.q.weight', 'action_head.weight')}
    critic_initial = {k: p.detach().clone() for k, p in critic.named_parameters()
                      if k in ('encoder.patch.weight', 'body.0.weight', 'output.weight')}
    metrics = []; stopped = False; families = {}
    for epoch in range(cfg['epochs']):
        permutation = rng.permutation(len(batch))
        for begin in range(0, len(batch), cfg['minibatch']):
            ids = permutation[begin:begin+cfg['minibatch']]
            row = batch.minibatch(ids, device)
            logs, features, _ = policy.team_distribution(row['vectors'], row['graphic'])
            logratio = logs.gather(-1, row['actions'][..., None]).squeeze(-1)-row['log_probs']
            ratio = logratio.exp()
            kl = ((ratio-1)-logratio).mean()
            if not torch.isfinite(kl): raise ArithmeticError('nonfinite PPO KL')
            if float(kl.detach().cpu()) > cfg['target_kl']:
                stopped = True; break
            adv = torch.from_numpy(advantages[ids]).to(device)[:, None].detach()
            per_agent = torch.maximum(-adv*ratio, -adv*ratio.clamp(1-cfg['clip'], 1+cfg['clip']))
            actor_loss = per_agent.mean()
            entropy = -(logs.exp()*logs).sum(-1).mean()
            actor_total = actor_loss-cfg['entropy']*entropy
            target = ((row['returns']-critic.mean)/critic.scale).detach()
            value_loss = F.smooth_l1_loss(critic.normalized(row['vectors'], row['graphic']), target)
            if not torch.isfinite(actor_total+value_loss): raise ArithmeticError('nonfinite loss')
            actor_opt.zero_grad(set_to_none=True); critic_opt.zero_grad(set_to_none=True)
            actor_total.backward(); value_loss.backward()
            def gradnorm(module, prefixes):
                terms = [p.grad.detach().norm() for name,p in module.named_parameters()
                         if name.startswith(prefixes) and p.grad is not None]
                return torch.stack(terms).sum() if terms else actor_loss.new_zeros(())
            actor_grad = gradnorm(policy, ('patch.', 'entity.', 'cross.'))
            critic_grad = gradnorm(critic, ('encoder.', 'body.'))
            anorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
            cnorm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
            before = copy.deepcopy(policy.state_dict()); opt_before = copy.deepcopy(actor_opt.state_dict())
            actor_opt.step()
            exact, approximate = audit(policy, probe)
            if not np.isfinite([exact, approximate]).all() or max(exact, approximate) > cfg['hard_kl']:
                policy.load_state_dict(before); actor_opt.load_state_dict(opt_before)
                raise RuntimeError('hard post-step KL gate; actor/optimizer restored, run stopped')
            critic_opt.step()
            metrics.append(dict(kl=float(kl.detach().cpu()), post_exact_kl=exact, post_sampled_kl=approximate,
                actor_loss=float(actor_loss.detach().cpu()), entropy=float(entropy.detach().cpu()),
                value_loss=float(value_loss.detach().cpu()), encoder_gradient=float(actor_grad.cpu()),
                critic_encoder_gradient=float(critic_grad.cpu()), actor_grad_norm=float(anorm.cpu()),
                critic_grad_norm=float(cnorm.cpu())))
            contribution = per_agent.detach().mean(-1).cpu().numpy()
            # Exact local action-head gradient contribution, without seven encoder backward passes.
            with torch.no_grad():
                probability = logs.detach().exp()
                indicator = F.one_hot(row['actions'], 9).to(probability.dtype)
                active = ((adv >= 0) & (ratio.detach() <= 1+cfg['clip'])) | ((adv < 0) & (ratio.detach() >= 1-cfg['clip']))
                derivative = (-adv*ratio.detach()*active)[..., None]*(indicator-probability)
                derivative += cfg['entropy']*probability*(logs.detach()-(probability*logs.detach()).sum(-1, keepdim=True))
                derivative /= len(ids)*5

            which = np.searchsorted(batch.offsets[1:], ids, side='right')
            for i, ep in enumerate(batch.episodes):
                chosen = which == i
                if not chosen.any(): continue
                key = ep.meta['opponent']['kind']
                item = families.setdefault(key, dict(samples=0, loss_sum=0., advantage_sum=0., advantage_square_sum=0., action_head_gradient_norm_sum=0.))
                selected_rows = torch.from_numpy(np.flatnonzero(chosen)).to(device)
                local_derivative = derivative[selected_rows].flatten(0, 1)
                local_features = features.detach()[selected_rows].flatten(0, 1)
                weight_grad = local_derivative.T @ local_features
                bias_grad = local_derivative.sum(0)
                item['action_head_gradient_norm_sum'] += float(torch.sqrt(weight_grad.square().sum()+bias_grad.square().sum()).cpu())
                item['samples'] += int(chosen.sum())
                item['loss_sum'] += float(contribution[chosen].sum())
                item['advantage_sum'] += float(advantages[ids[chosen]].sum())
                item['advantage_square_sum'] += float((advantages[ids[chosen]]**2).sum())
            progress(phase='update', epoch=epoch, minibatches=len(metrics))
            if max(exact, approximate) > cfg['target_kl']:
                stopped = True; break
        if stopped: break
    if not metrics: raise RuntimeError('no PPO update: behavior mismatch')
    delta = {k: float((dict(policy.named_parameters())[k].detach()-v).norm().cpu()) for k,v in initial.items()}
    cdelta = {k: float((dict(critic.named_parameters())[k].detach()-v).norm().cpu()) for k,v in critic_initial.items()}
    if delta['patch.weight'] == 0 or max(m['encoder_gradient'] for m in metrics) == 0:
        raise RuntimeError('actor encoder disconnected')
    # Zero output initialization legitimately blocks critic-body gradient on the first step.
    if len(metrics) > 1 and max(m['critic_encoder_gradient'] for m in metrics) == 0:
        raise RuntimeError('critic encoder disconnected after multiple updates')
    final_kl = np.zeros(2); count = 0
    for begin in range(0, len(batch), cfg['minibatch']):
        ids = np.arange(begin, min(len(batch), begin+cfg['minibatch']))
        final_kl += np.asarray(audit(policy, batch.minibatch(ids, device)))*len(ids); count += len(ids)
    if not np.isfinite(final_kl).all() or max(final_kl/count) > cfg["hard_kl"]:
        policy.load_state_dict(wave_actor); actor_opt.load_state_dict(wave_optimizer)
        raise RuntimeError("hard full-wave KL gate; wave actor restored, last checkpoint retained")
    return dict(minibatches=len(metrics), early_kl_stop=stopped, seconds=time.monotonic()-start,
        mean={k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
        parameter_delta=delta, critic_parameter_delta=cdelta, final_wave_kl=(final_kl/count).tolist(),
        critic_explained_variance_before_update=ev, family_loss_contribution=families,
        gradient_attribution='per-family unclipped-by-norm action-head contributions; encoder gradient is shared; summed norms are not a full-network decomposition',
        value_mean=float(critic.mean.detach().cpu()), value_scale=float(critic.scale.detach().cpu()))
