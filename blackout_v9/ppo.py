"""Agent-ratio MAPPO; raw observations are encoded again on every update."""
import time
import numpy as np
import torch
from torch.nn import functional as F


def optimizers(policy, critic, cfg):
    return (torch.optim.Adam(policy.parameters(), lr=cfg["actor_lr"]),
            torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"]))


def update(policy, critic, actor_opt, critic_opt, batch, cfg, rng, arm, progress=lambda **_: None):
    start = time.monotonic()
    device = next(policy.parameters()).device
    old_values = np.concatenate([e.arrays["values"][:e.length] for e in batch.episodes])
    return_variance = float(np.var(batch.returns))
    explained_variance = None if return_variance < 1e-12 else float(1-np.var(batch.returns-old_values)/return_variance)
    policy.train(); critic.train()
    advantages = (batch.advantages-batch.advantages.mean()) / max(float(batch.advantages.std()), 1e-8)
    old_scale = float(critic.scale.detach().cpu())
    critic.update_scale(torch.from_numpy(batch.returns).to(device))
    scale_ratio = old_scale / float(critic.scale.detach().cpu())
    for param in (critic.output.weight, critic.output.bias):
        state = critic_opt.state.get(param, {})
        for key, power in (("exp_avg", 1), ("exp_avg_sq", 2), ("max_exp_avg_sq", 2)):
            if key in state:
                state[key].mul_(scale_ratio**power)
    metrics = []
    initial = {k: p.detach().clone() for k, p in policy.named_parameters() if k in ("patch.weight", "cross.q.weight", "experts.weight")}
    stopped = False
    for epoch in range(cfg["epochs"]):
        for begin in range(0, len(batch), cfg["minibatch"]):
            if begin == 0:
                permutation = rng.permutation(len(batch))
            ids = permutation[begin:begin+cfg["minibatch"]]
            row = batch.minibatch(ids, device)
            logs, features, gates = policy.team_distribution(row["vectors"], row["graphic"])
            selected = logs.gather(-1, row["actions"][..., None]).squeeze(-1)
            logratio = selected - row["log_probs"]
            ratio = logratio.exp()
            kl = ((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):
                raise ArithmeticError("nonfinite PPO KL")
            if float(kl.detach().cpu()) > cfg["target_kl"]:
                stopped = True
                break
            adv = torch.from_numpy(advantages[ids]).to(device)[:, None]
            actor_loss = torch.maximum(-adv*ratio, -adv*ratio.clamp(1-cfg["clip"], 1+cfg["clip"])).mean()
            entropy = -(logs.exp()*logs).sum(-1).mean()
            if arm == "C":
                aux_loss = F.smooth_l1_loss(policy.auxiliary(features).mean(1), row["aux_targets"], reduction="none").mean(-1)
                auxiliary = (aux_loss*row["aux_valid"]).sum() / row["aux_valid"].sum().clamp_min(1)
            else:
                auxiliary = actor_loss.new_zeros(())
            actor_total = actor_loss - cfg["entropy"]*entropy + cfg["aux_coef"]*auxiliary
            targets = (row["returns"]-critic.mean) / critic.scale
            value_loss = F.smooth_l1_loss(critic.normalized(row["vectors"]), targets)
            if not torch.isfinite(actor_total+value_loss):
                raise ArithmeticError("nonfinite loss")
            actor_opt.zero_grad(set_to_none=True); critic_opt.zero_grad(set_to_none=True)
            actor_total.backward(); value_loss.backward()
            encoder_grad = torch.stack([p.grad.detach().norm() for name, p in policy.named_parameters()
                               if name.startswith(("patch.", "entity.", "cross.")) and p.grad is not None]).sum()
            actor_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
            actor_opt.step(); critic_opt.step()
            names = ("kl", "entropy", "actor_loss", "value_loss", "auxiliary_loss", "encoder_gradient",
                     "actor_grad_norm", "critic_grad_norm", "clip_fraction", "ratio_min", "ratio_max")
            packed = torch.cat((torch.stack((kl.detach(), entropy.detach(), actor_loss.detach(), value_loss.detach(),
                auxiliary.detach(), encoder_grad, actor_norm, critic_norm,
                ((ratio-1).abs()>cfg["clip"]).float().mean().detach(), ratio.min().detach(), ratio.max().detach())),
                gates.detach().mean((0, 1)))).cpu().tolist()
            metrics.append(dict(zip(names, packed[:len(names)]), expert_usage=packed[len(names):]))
            progress(phase="update", epoch=epoch, minibatches=len(metrics))
        if stopped:
            break
    if not metrics:
        raise RuntimeError("PPO performed no update; behavior policy mismatch")
    deltas = {k: float((dict(policy.named_parameters())[k].detach()-value).norm().cpu()) for k, value in initial.items()}
    if deltas["patch.weight"] == 0 or max(m["encoder_gradient"] for m in metrics) == 0:
        raise RuntimeError("encoder gradient/parameter update disconnected")
    keys = [k for k in metrics[0] if k != "expert_usage"]
    return {"minibatches": len(metrics), "early_kl_stop": stopped, "seconds": time.monotonic()-start,
            "mean": {k: float(np.mean([m[k] for m in metrics])) for k in keys},
            "critic_explained_variance_before_update": explained_variance,
            "expert_usage": np.mean([m["expert_usage"] for m in metrics], axis=0).tolist(),
            "parameter_delta": deltas, "value_mean": float(critic.mean.detach().cpu()),
            "value_scale": float(critic.scale.detach().cpu())}
