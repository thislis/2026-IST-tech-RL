"""Determinism, latency, clean-room, device, and promotion validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import shutil
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Mapping, Sequence

import torch


def assert_deterministic_inference(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, repeats: int = 3) -> None:
    if repeats < 2: raise ValueError("determinism check needs at least two repeats")
    model.eval()
    with torch.inference_mode(): outputs=[model(*inputs).detach().cpu() for _ in range(repeats)]
    if any(not torch.equal(outputs[0], output) for output in outputs[1:]):
        raise AssertionError("inference is not deterministic")


@dataclass(frozen=True)
class LatencyResult:
    device: str
    parameters: int
    median_ms: float
    p95_ms: float
    runs: int


def benchmark_inference(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, runs: int = 20) -> LatencyResult:
    if runs <= 0: raise ValueError("runs must be positive")
    device=str(next(model.parameters()).device); model.eval(); timings=[]
    with torch.inference_mode():
        model(*inputs)
        for _ in range(runs):
            started=time.perf_counter(); model(*inputs)
            if device.startswith("mps"): torch.mps.synchronize()
            timings.append((time.perf_counter()-started)*1000)
    ordered=sorted(timings)
    return LatencyResult(device,sum(p.numel() for p in model.parameters()),statistics.median(ordered),ordered[min(len(ordered)-1,int(.95*len(ordered)))],runs)


def select_lightweight_candidate(results: Sequence[Mapping[str,float]], *, max_win_rate_drop: float=.01) -> str:
    if not results: raise ValueError("no lightweight candidates")
    baseline=next((row for row in results if row["candidate"]=="baseline"),None)
    if baseline is None: raise ValueError("lightweight comparison requires baseline")
    eligible=[row for row in results if row["win_rate"] >= baseline["win_rate"]-max_win_rate_drop]
    return str(min(eligible,key=lambda row:(row["latency_ms"],row["parameters"]))["candidate"])


def clean_room_load(policy_file: str | Path, checkpoint: str | Path) -> dict[str, object]:
    """Copy exactly two artifacts and load/run them in an isolated Python process."""
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary); shutil.copy2(policy_file,root/"policy.py"); shutil.copy2(checkpoint,root/"checkpoint.pt")
        code=("import torch; from policy import load_policy; "
              "m=load_policy('checkpoint.pt'); y=m(torch.zeros(5,96),torch.zeros(5,11,96,96)); "
              "assert y.shape==(5,2); print('clean-room-ok')")
        import sys
        completed=subprocess.run([sys.executable,"-c",code],cwd=root,capture_output=True,text=True)
        if completed.returncode != 0: raise RuntimeError(completed.stderr)
        files=sorted(path.name for path in root.iterdir() if path.name != "__pycache__")
        return {"passed":True,"files":files,"stdout":completed.stdout.strip()}


@dataclass(frozen=True)
class PromotionEvidence:
    candidate_id: str
    win_rate: float
    mean_score_diff: float
    side_gap: float
    regressions: tuple[str,...]
    deterministic: bool
    clean_room: bool
    devices_passed: tuple[str,...]


def promote_final_model(evidence: PromotionEvidence, *, min_win_rate: float=.5, max_side_gap: float=.15) -> bool:
    return (evidence.win_rate >= min_win_rate and evidence.mean_score_diff >= 0
            and abs(evidence.side_gap) <= max_side_gap and not evidence.regressions
            and evidence.deterministic and evidence.clean_room
            and "cpu" in evidence.devices_passed and any(d in evidence.devices_passed for d in ("cuda","mps")))
