"""PPO of the sampled complete assignment, on immutable event support."""
import math
import torch
from .assignment import joint_entropy, joint_log_prob, sequential_log_prob


def event_log_prob(scorer, batch, arm):
    scores = _scores(scorer, batch)
    return _log_prob(scores, batch, arm)


def _scores(scorer, batch):
    return scorer(batch['features'], batch['prior']).masked_fill(~batch['legal'], 0.)


def _log_prob(scores, batch, arm):
    fn = sequential_log_prob if arm == 'S' else joint_log_prob
    return fn(scores, batch['assignment'], batch['capacities'],
              batch['legal'], batch['temperature'])


def _chunk(batch, indices, device):
    # A wave can contain many full-support events. Keep its large feature tensor
    # on the host and transfer only the current learner minibatch to the GPU.
    return {key: value[indices.to(value.device) if torch.is_tensor(indices) else indices].to(device)
            for key, value in batch.items()}


def update(scorer, critic, actor_optimizer, critic_optimizer, batch, arm, cfg):
    """No task regeneration, teacher forcing, or cross-episode return here."""
    device = next(scorer.parameters()).device
    if arm not in ('S','J','F'):
        raise ValueError('PPO supports only learned S/J/F arms')
    if not batch['on_policy'].all():
        raise ValueError('teacher/foreign-policy events cannot enter PPO')
    count = len(batch['old_log_prob'])
    if not count:
        raise ValueError('empty PPO wave')
    if any(len(v)!=count for v in batch.values()):
        raise ValueError('inconsistent PPO event batch lengths')
    if any(not torch.isfinite(v).all() for v in batch.values() if v.is_floating_point()):
        raise FloatingPointError('nonfinite PPO replay input')
    configured_minibatch = int(cfg.get('minibatch_events', 128))
    if configured_minibatch < 1:
        raise ValueError('positive PPO event minibatch size required')
    tasks = batch['features'].shape[-2]
    if tasks < 1:
        raise ValueError('PPO events require at least one task')
    minibatch = min(count, configured_minibatch,
                    max(1, configured_minibatch * 24 // tasks))
    entropy_coefficient = float(cfg.get('entropy', cfg.get('entropy_coef', 0.)))
    if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
        raise ValueError('nonnegative finite PPO entropy coefficient required')
    if arm == 'S' and entropy_coefficient:
        raise ValueError('exact PPO entropy is supported only for joint J/F assignments')
    entropy_sample_events = int(cfg.get('entropy_sample_events', 16))
    if entropy_coefficient and entropy_sample_events < 1:
        raise ValueError('positive PPO entropy event sample size required')
    with torch.no_grad():
        discrepancy = 0.
        for start in range(0, count, minibatch):
            chunk = _chunk(batch, slice(start, start+minibatch), device)
            discrepancy = max(discrepancy,
                              float((event_log_prob(scorer, chunk, arm)
                                     - chunk['old_log_prob']).abs().max()))
    if discrepancy > float(cfg.get('replay_logprob_tolerance', 2e-4)):
        raise ValueError(f'frozen support ratio != 1 before update: logprob error {discrepancy}')
    advantages = batch['advantages'].to(device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    report = {'events': count, 'old_new_logprob_max_error': discrepancy,
              'updates': 0, 'early_kl_stop': False,
              'minibatch_events': minibatch, 'task_count': tasks,
              'entropy_coefficient': entropy_coefficient,
              'entropy_per_unit': 0., 'assignment_entropy': 0., 'entropy_events': 0}
    clip = float(cfg.get('ppo_clip', .2))
    for epoch in range(int(cfg.get('ppo_epochs', 4))):
        for indices in torch.randperm(count).split(minibatch):
            chunk = _chunk(batch, indices, device)
            scores = _scores(scorer, chunk)
            lp = _log_prob(scores, chunk, arm)
            log_ratio = lp - chunk['old_log_prob']
            ratio = torch.exp(log_ratio)
            kl = ((ratio - 1) - log_ratio).mean()
            if not torch.isfinite(kl):
                raise FloatingPointError('nonfinite assignment KL')
            if float(kl.detach()) > float(cfg.get('target_kl', .02)):
                report['early_kl_stop'] = True
                return report
            adv = advantages[indices.to(device)]
            actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip)*adv).mean()
            entropy = scores.new_zeros(())
            entropy_events = 0
            if entropy_coefficient:
                # Uniform event subsampling gives an unbiased minibatch entropy
                # gradient while bounding the second-derivative DP graph size.
                entropy_events = min(len(indices), entropy_sample_events)
                selected = torch.randperm(len(indices), device=device)[:entropy_events]
                entropy = joint_entropy(scores[selected], chunk['capacities'][selected],
                                        chunk['legal'][selected], chunk['temperature'][selected]).mean()
                actor_loss = actor_loss - entropy_coefficient * entropy / scores.shape[-2]
            values = critic(chunk['critic_features'])
            critic_loss = .5*(values-chunk['returns']).square().mean()
            if not torch.isfinite(actor_loss+critic_loss):
                raise FloatingPointError('nonfinite PPO loss')
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), float(cfg.get('max_grad_norm', .5)), error_if_nonfinite=True)
            actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(cfg.get('max_grad_norm', .5)), error_if_nonfinite=True)
            critic_optimizer.step()
            report.update(updates=report['updates']+1, epoch=epoch+1,
                          actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                          approximate_kl=float(kl.detach()), entropy_events=entropy_events,
                          assignment_entropy=float(entropy.detach()),
                          entropy_per_unit=float(entropy.detach())/scores.shape[-2])
    return report
