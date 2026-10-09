"""Measured CPU/MPS choices; all probes use disposable models, never a live learner."""

from project_paths import project_root, project_path

import copy
import time
import numpy as np
import torch
from .io import ROOT
from .policy import MyPolicy


def fixture():
    paths = sorted((project_path('logs/v8/provided_competition_v1', root=ROOT)).glob("**/policy_inputs.pt"))
    if paths:
        row = torch.load(paths[0], map_location="cpu", weights_only=True)[0]
        return row["vector"].float(), row["graphic"].float()
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
                v, g = vector.to(device), graphic.to(device)
                samples = []
                try:
                    with torch.inference_mode():
                        for i in range(repetitions+3):
                            sync(device); start = time.perf_counter()
                            candidate(v, g).cpu()
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
    """Synthetic encoder-backward correctness/latency probe, not an experiment."""
    torch.manual_seed(902)
    reference = MyPolicy()
    v, g = fixture()
    v = v[None].expand(minibatch, -1, -1).contiguous()
    g = g[:1].expand(minibatch, -1, -1, -1).contiguous()
    reports = []
    cpu_gradient = None
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available(): continue
        candidate = copy.deepcopy(reference).to(device)
        tv, tg = v.to(device), g.to(device)
        try:
            durations = []
            for _ in range(3):
                candidate.zero_grad(set_to_none=True)
                sync(device); start = time.perf_counter()
                logs = candidate.team_distribution(tv, tg)[0]
                loss = -logs[..., 1].mean()
                loss.backward(); sync(device)
                durations.append(time.perf_counter()-start)
            gradient = candidate.patch.weight.grad.detach().cpu()
            if not torch.isfinite(gradient).all() or gradient.norm() == 0:
                raise ValueError("encoder backward failed")
            if device == "cpu": cpu_gradient = gradient
            elif not torch.allclose(gradient, cpu_gradient, atol=2e-5, rtol=2e-3):
                raise ValueError("MPS/CPU encoder gradient parity failed")
            reports.append({"device": device, "seconds": float(np.median(durations[1:])), "valid": True})
        except (RuntimeError, ValueError) as exc:
            reports.append({"device": device, "valid": False, "error": str(exc)})
        finally:
            del candidate, tv, tg
            if device == "mps": torch.mps.empty_cache()
    valid = [r for r in reports if r["valid"] and (requested == "auto" or r["device"] == requested)]
    if not valid: raise RuntimeError("no validated learner device: " + str(reports))
    selected = min(valid, key=lambda r: r["seconds"])["device"]
    return {"selected_device": selected, "probes": reports, "synthetic_only": True, "persisted_learning_updates": 0}


class WorkerTuner:
    """Online bounded throughput pilot, frozen after three waves per candidate.

    Unlike a controlled benchmark this is a heuristic under varying opponents.
    Memory pressure always takes precedence over throughput.
    """
    def __init__(self, initial=4, cap=8, state=None):
        self.candidates = [n for n in (2, 4, 6, 8) if n <= cap] or [1]
        self.selected = initial
        self.samples = {} if state is None else state.get("samples", {})
        self.frozen = False if state is None else state.get("frozen", False)
        if state: self.selected = state["selected"]

    def observe(self, workers, steps, seconds, memory_gib, limit=32):
        self.samples.setdefault(str(workers), []).append(steps/max(seconds, 1e-6))
        if memory_gib > limit:
            self.selected = max(1, workers-2)
            self.frozen = True
        elif not self.frozen:
            untested = [n for n in self.candidates if len(self.samples.get(str(n), [])) < 3]
            if untested: self.selected = untested[0]
            else:
                self.selected = max(self.candidates, key=lambda n: np.median(self.samples[str(n)]))
                self.frozen = True
        return self.selected

    def state(self):
        return {"selected": self.selected, "samples": self.samples, "frozen": self.frozen,
                "method": "three-wave collection throughput heuristic; not a controlled speedup claim"}
