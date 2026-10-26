"""Recurrent event PPO with exact recorded choices and separate value gradients."""
import time
import ctypes
import os
import resource
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import torch
from .critic import Critic


class UpdateRejected(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("post-update KL exceeded registered hard bound: " + str(report))


@lru_cache(maxsize=1)
def _resident_reader():
    if sys.platform == "darwin":
        library = ctypes.CDLL(None)
        port = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
        task_info = library.task_info
        task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        task_info.restype = ctypes.c_int
        def read():
            # mach_task_basic_info_data_t: first three fields are 64-bit
            # virtual_size, resident_size, resident_size_max; twelve naturals.
            data = (ctypes.c_uint64*6)()
            count = ctypes.c_uint32(ctypes.sizeof(data)//4)
            if task_info(port, 20, ctypes.byref(data), ctypes.byref(count)) != 0:
                raise RuntimeError("cannot measure current learner RSS")
            return int(data[1])
        return read
    def read():
        with open("/proc/self/statm") as stream:
            return int(stream.read().split()[1])*os.sysconf("SC_PAGE_SIZE")
    return read


def _to_device(tree, device):
    return {k: _to_device(v, device) if isinstance(v, dict) else v.to(device) for k, v in tree.items()}


def _sequence_batches(group, batch_size, device, prefetch):
    groups = [group[i:i+batch_size] for i in range(0, len(group), batch_size)]
    if not prefetch:
        for descriptors in groups:
            yield sequence_batch(descriptors, device)
        return
    # Only one future CPU batch can exist. The worker never touches MPS or
    # policy parameters; recurrent order and frozen-wave boundaries are intact.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="v11-replay") as executor:
        pending = executor.submit(sequence_batch, groups[0], "cpu")
        for index in range(len(groups)):
            cpu_batch, burn = pending.result()
            pending = executor.submit(sequence_batch, groups[index+1], "cpu") if index+1 < len(groups) else None
            data = _to_device(cpu_batch, device)
            del cpu_batch
            yield data, burn


def optimizers(policy, critic, cfg):
    if set(map(id, policy.parameters())) & set(map(id, critic.parameters())):
        raise ValueError("actor and critic parameters overlap")
    return (torch.optim.Adam(policy.parameters(), lr=cfg.get("actor_lr", 1e-4)),
            torch.optim.Adam(critic.parameters(), lr=cfg.get("critic_lr", 3e-4)))


def _slice(tree, t):
    return {key: _slice(value, t) if isinstance(value, dict) else value[:, t] for key, value in tree.items()}


def _stack(rows, device):
    return {key: _stack([r[key] for r in rows], device) if isinstance(rows[0][key], dict)
            else torch.stack([r[key] for r in rows]).to(device) for key in rows[0]}


def _recurrent(state, hidden):
    state = dict(state)
    if hidden is not None:
        unit, team = hidden
        if "unit_reset" in state:
            unit = unit * (~state["unit_reset"].bool()).unsqueeze(-1)
        if "team_reset" in state:
            team = team * (~state["team_reset"].bool()).unsqueeze(-1)
        state["unit_hidden"], state["team_hidden"] = unit, team
    return state


def _hidden(result):
    return result["unit_hidden"], result["team_hidden"]


def sequence_batch(descriptors, device):
    """Pad only at the end; padding contributes no actor/critic loss."""
    burn = descriptors[0][2]-descriptors[0][1]
    if any(s-b != burn for _, b, s, _ in descriptors):
        raise ValueError("group sequences by burn-in length")
    length = max(end-begin for _, begin, _, end in descriptors)
    rows, masks = [], []
    for episode, begin, start, end in descriptors:
        ids = np.minimum(np.arange(begin, begin+length), end-1)
        rows.append(episode.read(ids))
        masks.append(np.arange(length) < end-begin)
    batch = _stack(rows, device)
    batch["valid"] = torch.as_tensor(np.stack(masks), device=device)
    return batch, burn


@torch.no_grad()
def ratio_audit(policy, batch, device, tolerance=2e-3, count=32):
    errors = []
    for episode in batch.episodes:
        n = min(count, episode.length)
        data = episode.read(np.arange(n), device)
        if not bool(data["on_policy"].all()):
            raise ValueError("teacher-forced data cannot enter PPO")
        out = policy.evaluate_decision(data["vectors"], data["graphic"], data["state"], data["choices"])
        error = (out["log_prob"]-data["log_probs"]).abs().max().item()
        errors.append(error)
    maximum = max(errors)
    if maximum > tolerance:
        raise ArithmeticError(f"behavior likelihood mismatch before update: max log error {maximum:g}")
    return {"max_log_prob_error": maximum, "max_ratio_error_bound": float(np.expm1(maximum)),
            "events_checked": sum(min(count, e.length) for e in batch.episodes)}


@torch.no_grad()
def post_update_kl(policy, batch, device, count=64):
    # Start at each real episode boundary so updated hidden is recomputed without
    # stale stored prefix. This fixed probe is selected before seeing updates.
    descriptors = [(ep, 0, 0, min(count, ep.length)) for ep in batch.episodes]
    data, _ = sequence_batch(descriptors, device)
    hidden, total, maximum, events = None, 0., 0., 0
    for t in range(data["valid"].shape[1]):
        frame = _slice(data, t)
        out = policy.evaluate_decision(frame["vectors"], frame["graphic"],
                                       _recurrent(frame["state"], hidden), frame["choices"])
        hidden = _hidden(out)
        ratio = out["log_prob"]-frame["log_probs"]
        kl = (ratio.clamp(-20, 20).exp()-1-ratio)[frame["valid"]]
        if not torch.isfinite(kl).all():
            raise ArithmeticError("nonfinite post-update KL")
        total += float(kl.sum().cpu())
        maximum = max(maximum, float(kl.max().cpu()))
        events += len(kl)
    return {"mean_kl": total/max(events, 1), "max_event_kl": maximum, "events": events,
            "hidden_recomputed_from_episode_start": True}


def update(policy, critic, actor_opt, critic_opt, batch, cfg, rng, arm=None, progress=lambda **_: None,
           stop=lambda: False):
    del arm
    device = next(policy.parameters()).device
    audit = ratio_audit(policy, batch, device, cfg.get("ratio_tolerance", 2e-3))
    advantages = batch.advantages.astype(np.float64)
    adv_mean, adv_std = float(advantages.mean()), max(float(advantages.std()), 1e-6)
    critic.update_scale(torch.as_tensor(batch.returns, device=device))
    burn_in, learn = cfg.get("burn_in", 32), cfg.get("sequence_length", 64)
    sequences = batch.sequences(burn_in, learn)
    batch_size = int(cfg.get("minibatch_sequences", cfg.get("minibatch", 16)))
    metrics, stopped_kl = [], False
    prefetch = device.type == "mps" and cfg.get("prefetch", True)
    started = time.monotonic()
    for epoch in range(int(cfg.get("epochs", 4))):
        order = rng.permutation(len(sequences))
        groups = {}
        for index in order:
            desc = sequences[int(index)]
            groups.setdefault(desc[2]-desc[1], []).append(desc)
        for group in groups.values():
            batches = _sequence_batches(group, batch_size, device, prefetch)
            for data, burn in batches:
                if stop():
                    raise InterruptedError("PPO interrupted before atomic wave commit")
                hidden = None
                for t in range(burn):
                    frame = _slice(data, t)
                    with torch.no_grad():
                        out = policy.evaluate_decision(frame["vectors"], frame["graphic"],
                                                       _recurrent(frame["state"], hidden), frame["choices"])
                    hidden = tuple(x.detach() for x in _hidden(out))
                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                policy_losses, value_losses, entropies, kls, clips, masks = [], [], [], [], [], []
                for t in range(burn, data["valid"].shape[1]):
                    frame = _slice(data, t)
                    state = _recurrent(frame["state"], hidden)
                    out = policy.evaluate_decision(frame["vectors"], frame["graphic"], state, frame["choices"])
                    hidden = _hidden(out)
                    mask = frame["valid"].float() * frame["on_policy"].float()
                    log_ratio = out["log_prob"]-frame["log_probs"]
                    ratio = log_ratio.clamp(-20, 20).exp()
                    advantage = (frame["advantages"]-adv_mean)/adv_std
                    unclipped = ratio*advantage
                    clipped = ratio.clamp(1-cfg.get("clip", .2), 1+cfg.get("clip", .2))*advantage
                    policy_losses.append(-torch.minimum(unclipped, clipped)*mask)
                    entropies.append(out["entropy"]*mask)
                    # Critic receives recorded PRE-decision state, no actor graph.
                    pred = critic.normalized(frame["vectors"], frame["graphic"], frame["state"])
                    target = (frame["returns"]-critic.mean)/critic.scale
                    value_losses.append(.5*(pred-target).square()*mask)
                    kls.append((ratio-1-log_ratio)*mask)
                    clips.append(((ratio-1).abs() > cfg.get("clip", .2)).float()*mask)
                    masks.append(mask)
                denom = torch.stack(masks).sum().clamp_min(1)
                entropy = torch.stack(entropies).sum()/denom
                actor_loss = torch.stack(policy_losses).sum()/denom-cfg.get("entropy", .003)*entropy
                value_loss = torch.stack(value_losses).sum()/denom
                if not torch.isfinite(actor_loss+value_loss):
                    raise ArithmeticError("nonfinite recurrent PPO objective")
                actor_loss.backward()
                value_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.get("max_grad_norm", .5))
                critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.get("max_grad_norm", .5))
                if not torch.isfinite(actor_norm+critic_norm):
                    raise ArithmeticError("nonfinite recurrent PPO gradient")
                actor_opt.step()
                critic_opt.step()
                kl = float((torch.stack(kls).sum()/denom).detach().cpu())
                row = {"epoch": epoch, "actor_loss": float(actor_loss.detach().cpu()),
                       "critic_loss": float(value_loss.detach().cpu()), "entropy": float(entropy.detach().cpu()),
                       "approx_kl": kl, "clip_fraction": float((torch.stack(clips).sum()/denom).detach().cpu()),
                       "actor_grad_norm": float(actor_norm.detach().cpu()), "critic_grad_norm": float(critic_norm.detach().cpu()),
                       "events": int(denom.detach().cpu())}
                metrics.append(row)
                progress(phase="recurrent_ppo", epoch=epoch, minibatches=len(metrics), **{k: v for k, v in row.items() if k != "epoch"})
                del data, hidden
                if kl > cfg.get("target_kl", .02):
                    stopped_kl = True
                    break
            batches.close()
            if stopped_kl:
                break
        if stopped_kl:
            break
    if not metrics:
        raise ValueError("no recurrent PPO minibatch")
    report = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0] if key != "epoch"}
    report.update(minibatches=len(metrics), seconds=time.monotonic()-started, stopped_kl=stopped_kl,
                  ratio_audit=audit, burn_in=burn_in, sequence_length=learn,
                  hidden_recomputed=True, actor_critic_separate=True, cpu_prefetch=prefetch)
    report["post_update_kl"] = post_update_kl(policy, batch, device, cfg.get("audit_events", 64))
    if report["post_update_kl"]["mean_kl"] > cfg.get("hard_kl", .04):
        raise UpdateRejected(report)
    return report


def benchmark_update(device, batch_sequences=4, steps=4):
    """Disposable recurrent actor/critic updates, including input transfer costs."""
    from .policy import MyPolicy
    from .benchmark import fixture
    device = torch.device(device)
    torch.manual_seed(719)
    policy, critic = MyPolicy().to(device).train(), Critic().to(device).train()
    policy.configure("F", "sample")
    policy.configure_devices(planner_device="cpu")
    sample = fixture()
    if isinstance(sample, dict):
        v, g = sample["vector"], sample["graphic"]
    else:
        v, g = sample
    if not isinstance(v, torch.Tensor):
        v, g = torch.as_tensor(v), torch.as_tensor(g)
    if v.ndim == 2:
        v = v[None]
    if g.ndim == 4 and g.shape[0] == 5:
        g = g[:1]
    if g.ndim == 3:
        g = g[None]
    v, g = v[:1].expand(batch_sequences, -1, -1).contiguous(), g[:1].expand(batch_sequences, -1, -1, -1).contiguous()
    with torch.no_grad():
        decision = policy.step(v.to(device), g.to(device), policy.initial_state(batch_sequences, device), deterministic=True)
    state = {k: x.detach() for k, x in decision["decision_state"].items()}
    choices = {k: x.detach() for k, x in decision["choices"].items()}
    actor_opt, critic_opt = optimizers(policy, critic, {})
    def synchronize():
        if device.type == "mps":
            torch.mps.synchronize()
    peak_rss, peak_mps = 0, 0
    def sample_memory():
        nonlocal peak_rss, peak_mps
        rss = _resident_reader()()
        peak_rss = max(peak_rss, rss)
        if device.type == "mps":
            peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
        if max(peak_rss, peak_mps) > 32*1024**3:
            raise RuntimeError("learner probe exceeded 32 GiB memory soft limit")
    def one_update(events=4, burn_in=0):
        # Materialize the sequence inputs, matching real replay storage/transfer
        # rather than sharing one image allocation over all recurrent events.
        total_events = events+burn_in
        cpu_v = v[:, None].expand(-1, total_events, -1, -1).contiguous()
        cpu_g = g[:, None].expand(-1, total_events, -1, -1, -1).contiguous()
        vt, gt = cpu_v.to(device), cpu_g.to(device)
        # Keep one CPU sequence batch resident, matching bounded next-batch
        # prefetch while the current MPS batch is being learned.
        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        hidden, losses, value_losses = None, [], []
        for t in range(burn_in):
            current_state = dict(state)
            if t:
                current_state["unit_reset"] = torch.zeros_like(state["unit_reset"])
                current_state["team_reset"] = torch.zeros_like(state["team_reset"])
            with torch.no_grad():
                out = policy.evaluate_decision(vt[:, t], gt[:, t], _recurrent(current_state, hidden), choices)
            hidden = tuple(x.detach() for x in _hidden(out))
        for t in range(burn_in, total_events):
            current_state = dict(state)
            if t:
                current_state["unit_reset"] = torch.zeros_like(state["unit_reset"])
                current_state["team_reset"] = torch.zeros_like(state["team_reset"])
            out = policy.evaluate_decision(vt[:, t], gt[:, t], _recurrent(current_state, hidden), choices)
            hidden = _hidden(out)
            losses.append(-out["log_prob"].mean()-.003*out["entropy"].mean())
            value_losses.append(.5*(critic.normalized(vt[:, t], gt[:, t], state)-.5).square().mean())
            if t % 8 == 0:
                sample_memory()
        sample_memory()
        (torch.stack(losses).mean()).backward()
        (torch.stack(value_losses).mean()).backward()
        sample_memory()
        finite = all(bool(torch.isfinite(p.grad).all()) for module in (policy, critic)
                     for p in module.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
        actor_opt.step()
        critic_opt.step()
        return finite
    cold = time.perf_counter()
    one_update(4)
    synchronize()
    warmup_seconds = time.perf_counter()-cold
    reference = next(policy.parameters()).detach().clone()
    synchronize()
    started = time.perf_counter()
    finite = True
    for _ in range(steps):
        finite = one_update(4) and finite
    synchronize()
    seconds = time.perf_counter()-started
    changed = not torch.equal(reference, next(policy.parameters()).detach())
    # Verify actual training sequence activation residency once per candidate.
    # This is excluded from short-kernel timing and explicitly reported.
    capacity_start = time.perf_counter()
    finite = one_update(64, burn_in=32) and finite
    synchronize()
    capacity_seconds = time.perf_counter()-capacity_start
    return {"seconds": seconds, "events_per_second": batch_sequences*64/max(capacity_seconds, 1e-9),
            "short_sequence_events_per_second": batch_sequences*steps*4/max(seconds, 1e-9),
            "finite_gradients": finite, "parameters_changed": changed, "batch_sequences": batch_sequences,
            "device": str(device), "recurrent_events": 4, "warmup_seconds": warmup_seconds,
            "capacity_sequence_length": 64, "capacity_burn_in": 32, "capacity_seconds": capacity_seconds, "memory_allocated":
                int(torch.mps.driver_allocated_memory()) if device.type == "mps" else None,
            "peak_rss_bytes": peak_rss, "peak_mps_driver_bytes": peak_mps,
            "rss_peak_scope": "current process RSS sampled per probe",
            "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == "darwin" else 1024)}
