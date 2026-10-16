"""Measured CPU/MPS choices; all probes use disposable models, never a live learner."""

from project_paths import project_root, project_path

import copy
import time
import numpy as np
import torch
from .io import ROOT
from .policy import MyPolicy


def fixture():
    vector = torch.zeros(5, 96)
    vector[:, :90].reshape(5, 10, 9)[:, :, 2] = torch.tensor([1]*5+[-1]*5)
    vector[:, 90] = 1
    graphic = torch.zeros(5, 11, 96, 96); graphic[:, 0] = 1
    return vector, graphic


def sync(device):
    if device == "mps": torch.mps.synchronize()


def benchmark_inference(repetitions=20):
    old_threads = torch.get_num_threads()
    vector, graphic = fixture()
    torch.manual_seed(901)
    policy = MyPolicy().eval()
    results = []
    try:
        for threads in (1, 2):
            torch.set_num_threads(threads)
            for device in ("cpu", "mps"):
                if device == "mps" and not torch.backends.mps.is_available(): continue
                candidate = copy.deepcopy(policy).to(device)
                v, g = vector, graphic
                samples = []
                try:
                    with torch.inference_mode():
                        for i in range(repetitions+3):
                            sync(device); start = time.perf_counter()
                            candidate(v.to(device), g.to(device)).cpu()
                            sync(device)
                            if i >= 3: samples.append(time.perf_counter()-start)
                    results.append({"device": device, "threads": threads,
                                    "p50": float(np.median(samples)), "p95": float(np.quantile(samples, .95)),
                                    "p99": float(np.quantile(samples, .99)), "batch": 5})
                except RuntimeError as exc:
                    results.append({"device": device, "threads": threads, "error": str(exc)})
                del candidate
        cpu = min((r for r in results if r["device"] == "cpu" and "error" not in r), key=lambda r: r["p95"])
        return {"measurements": results, "selected_actor_threads": cpu["threads"],
                "actor_device": "cpu", "unity_started": False, "gradient_updates": 0}
    finally:
        torch.set_num_threads(old_threads)


def choose_learner(requested="auto", minibatch=128):
    """CPU/MPS parity and end-to-end transfer + actor/critic Adam probe."""
    from .value import Critic
    from .ppo import optimizers
    from .contracts import config
    cfg = config()["ppo"]
    with torch.random.fork_rng():
        torch.manual_seed(902)
        reference = MyPolicy()
        critics = {arm: Critic(arm) for arm in ("C0", "C1")}
    v, g = fixture()
    v = v[None].expand(minibatch, -1, -1).contiguous()
    g = g[:1].expand(minibatch, -1, -1, -1).contiguous()
    reports = []; cpu_states = {}; cpu_outputs = {}
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            reports.append(dict(device=device, valid=False, error="MPS unavailable")); continue
        timings = {}; parity = {}; failed = None
        try:
            for arm in ("C0", "C1"):
                actor = copy.deepcopy(reference).to(device)
                critic = copy.deepcopy(critics[arm]).to(device)
                ao, co = optimizers(actor, critic, cfg)
                durations = []
                for repeat in range(3):
                    sync(device); start = time.perf_counter()
                    tv, tg = v.to(device), g.to(device)
                    with torch.no_grad(): old_value = critic(tv, tg)
                    ao.zero_grad(set_to_none=True); co.zero_grad(set_to_none=True)
                    logs = actor.team_log_probs(tv, tg)
                    loss = -logs[..., 1].mean()
                    value_loss = (critic.normalized(tv, tg)-.3).square().mean()
                    loss.backward(); value_loss.backward()
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in list(actor.parameters())+list(critic.parameters())):
                        raise ValueError("nonfinite device gradient")
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
                    ao.step(); co.step()
                    # Include device-to-host transfer required by frozen behavior snapshots.
                    state = {"actor."+k: x.detach().cpu().clone() for k,x in actor.named_parameters()}
                    state.update({"critic."+k: x.detach().cpu().clone() for k,x in critic.named_parameters()})
                    outputs = (logs.detach().cpu(), old_value.detach().cpu(), actor.patch.weight.grad.detach().cpu())
                    sync(device); durations.append(time.perf_counter()-start)
                if device == "cpu": cpu_states[arm], cpu_outputs[arm] = state, outputs
                else:
                    for actual, expected in zip(outputs, cpu_outputs[arm]):
                        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
                    for key in state:
                        torch.testing.assert_close(state[key], cpu_states[arm][key], atol=2e-4, rtol=2e-3)
                before = critic(tv, tg).detach()
                critic.update_scale(torch.tensor([-.5, .2, 1.], device=device))
                torch.testing.assert_close(critic(tv, tg), before, atol=2e-5, rtol=2e-3)
                timings[arm] = float(np.median(durations[1:]))
                parity[arm] = True
        except (RuntimeError, ValueError, AssertionError) as exc:
            failed = str(exc)
        finally:
            if device == "mps": torch.mps.empty_cache()
        reports.append(dict(device=device, valid=failed is None, error=failed, arm_seconds=timings,
                            seconds=sum(timings.values()), parity=parity))
    valid = [r for r in reports if r["valid"] and (requested == "auto" or requested == r["device"])]
    if not valid: raise RuntimeError("no validated learner device; retry --device cpu: " + str(reports))
    return dict(selected_device=min(valid,key=lambda r:r["seconds"])["device"], probes=reports,
                synthetic_only=True, persisted_learning_updates=0, minibatch=minibatch)


class WorkerTuner:
    """Fixed paired-arm worker count. Pressure stops instead of altering the comparison."""
    def __init__(self, initial=4, cap=8, state=None):
        if not 1 <= initial <= cap <= 8: raise ValueError("invalid worker budget")
        self.selected = initial
        self.samples = {} if state is None else state.get("samples", {})
        if state and state["selected"] != initial: raise ValueError("registered workers changed")

    def observe(self, workers, steps, seconds, memory_gib, limit=32):
        self.samples.setdefault(str(workers), []).append(steps/max(seconds, 1e-6))
        self.samples[str(workers)] = self.samples[str(workers)][-16:]
        if memory_gib > limit: raise RuntimeError("memory soft limit exceeded; retain last committed wave")
        return self.selected

    def state(self):
        return dict(selected=self.selected, samples=self.samples, frozen=True, method="fixed paired-arm workers")


def benchmark_values(device, candidates=(128,256,512)):
    """Frozen map-value throughput including host/device transfers."""
    from .value import Critic
    v,g=fixture();reports=[]
    with torch.random.fork_rng():
        torch.manual_seed(903);critic=Critic('C1').to(device).eval()
    with torch.inference_mode():
        for size in candidates:
            vector=v[None].expand(size,-1,-1).contiguous()
            graphic=g[:1].expand(size,-1,-1,-1).contiguous()
            try:
                samples=[]
                for repeat in range(3):
                    sync(device);start=time.perf_counter()
                    value=critic(vector.to(device),graphic.to(device)).cpu()
                    sync(device)
                    if not torch.isfinite(value).all():raise ValueError('nonfinite batched value')
                    if repeat:samples.append(time.perf_counter()-start)
                seconds=float(np.median(samples))
                reports.append(dict(batch=size,seconds=seconds,scenes_per_second=size/seconds,valid=True))
            except (RuntimeError,ValueError) as exc:
                reports.append(dict(batch=size,valid=False,error=str(exc)))
            finally:
                if device=='mps':torch.mps.empty_cache()
    valid=[r for r in reports if r['valid']]
    if not valid:raise RuntimeError('no valid value microbatch')
    return dict(selected_batch=max(valid,key=lambda r:r['scenes_per_second'])['batch'],probes=reports)
