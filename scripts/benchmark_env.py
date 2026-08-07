#!/usr/bin/env python3
"""Benchmark one or more live BlackOut Unity environments (PREP-11)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from blackout_rl import ContractBlackOutEnv


GPU_KEYS = (
    "Device Utilization %",
    "Renderer Utilization %",
    "Tiler Utilization %",
    "In use system memory",
)


def executable_in(build: Path) -> Path:
    if build.suffix != ".app":
        return build
    candidates = [path for path in (build / "Contents" / "MacOS").iterdir() if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one executable in app bundle, got {candidates}")
    return candidates[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_apple_gpu() -> dict[str, float] | None:
    """Read whole-device Apple GPU counters exposed by AGXAccelerator.

    These are host-wide counters, so the report includes an idle baseline and
    does not claim process-level attribution.
    """
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    result: dict[str, float] = {}
    for key in GPU_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*=\s*([0-9]+)', output)
        if match:
            value = float(match.group(1))
            if key == "In use system memory":
                result["in_use_system_memory_mib"] = value / (1024 * 1024)
            else:
                result[key.lower().replace(" ", "_").replace("%", "pct")] = value
    return result or None


def sample_nvidia_gpu() -> dict[str, float] | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [row.strip().split(",") for row in output.splitlines() if row.strip()]
    if not rows:
        return None
    return {
        "device_utilization_pct": mean(float(row[0]) for row in rows),
        "memory_used_mib": sum(float(row[1]) for row in rows),
    }


def sample_gpu() -> dict[str, float] | None:
    return sample_apple_gpu() if platform.system() == "Darwin" else sample_nvidia_gpu()


def sample_processes(pids: list[int]) -> dict[str, float] | None:
    if not pids:
        return None
    try:
        output = subprocess.run(
            ["ps", "-o", "pid=,%cpu=,rss=", "-p", ",".join(str(pid) for pid in pids)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [row.split() for row in output.splitlines() if row.strip()]
    if not rows:
        return None
    return {
        "cpu_pct_sum": sum(float(row[1]) for row in rows),
        "rss_mib_sum": sum(float(row[2]) for row in rows) / 1024.0,
    }


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for sample in samples for key in sample})
    return {
        key: {
            "mean": round(mean(sample[key] for sample in samples if key in sample), 3),
            "max": round(max(sample[key] for sample in samples if key in sample), 3),
        }
        for key in keys
    }


def full_episode_cost(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    episodes = payload.get("episodes", [])
    if not episodes:
        return None
    steps = [int(episode["episode_length"]["steps"]) for episode in episodes]
    seconds = [float(episode["episode_length"]["wall_seconds"]) for episode in episodes]
    return {
        "source": str(path),
        "episodes": len(episodes),
        "mean_episode_steps": round(mean(steps), 3),
        "mean_wall_seconds_per_episode": round(mean(seconds), 3),
        "mean_environment_steps_per_second": round(sum(steps) / sum(seconds), 3),
        "wall_seconds": seconds,
    }


def benchmark_count(
    *,
    build: Path,
    count: int,
    seed: int,
    time_scale: float,
    warmup_steps: int,
    measure_steps: int,
    sample_every: int,
) -> dict[str, Any]:
    envs: list[ContractBlackOutEnv] = []
    process_samples: list[dict[str, float]] = []
    gpu_samples: list[dict[str, float]] = []
    rngs = [np.random.default_rng(seed + index) for index in range(count)]
    try:
        for index in range(count):
            env = ContractBlackOutEnv(
                env_path=str(build),
                no_graphics=False,
                time_scale=time_scale,
            )
            env.reset(seed=seed + index)
            envs.append(env)

        for _ in range(warmup_steps):
            for index, env in enumerate(envs):
                actions = {
                    agent: rngs[index].uniform(-1.0, 1.0, 2).astype(np.float32)
                    for agent in env.agents
                }
                env.step(actions)

        unity_pids = [int(env._unity_env._process.pid) for env in envs]
        all_pids = [os.getpid(), *unity_pids]
        started = time.perf_counter()
        for outer_step in range(measure_steps):
            for index, env in enumerate(envs):
                if not env.agents:
                    env.reset(seed=seed + index + outer_step + 1)
                actions = {
                    agent: rngs[index].uniform(-1.0, 1.0, 2).astype(np.float32)
                    for agent in env.agents
                }
                env.step(actions)
            if outer_step % sample_every == 0 or outer_step == measure_steps - 1:
                process_sample = sample_processes(all_pids)
                if process_sample:
                    process_samples.append(process_sample)
                gpu_sample = sample_gpu()
                if gpu_sample:
                    gpu_samples.append(gpu_sample)
        wall_seconds = time.perf_counter() - started
    finally:
        for env in envs:
            env.close()

    environment_steps = count * measure_steps
    return {
        "environment_count": count,
        "warmup_steps_per_environment": warmup_steps,
        "measured_steps_per_environment": measure_steps,
        "aggregate_environment_steps": environment_steps,
        "agent_decisions": environment_steps * 10,
        "wall_seconds": round(wall_seconds, 6),
        "aggregate_environment_steps_per_second": round(environment_steps / wall_seconds, 3),
        "per_environment_steps_per_second": round(measure_steps / wall_seconds, 3),
        "agent_decisions_per_second": round(environment_steps * 10 / wall_seconds, 3),
        "unity_process_ids": unity_pids,
        "process_metrics": summarize_samples(process_samples),
        "gpu_metrics": summarize_samples(gpu_samples),
        "metric_samples": {"process": len(process_samples), "gpu": len(gpu_samples)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--env-counts", default="1,2")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--measure-steps", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument(
        "--episode-log",
        type=Path,
        default=REPOSITORY_ROOT / "logs" / "prep08_10_paired_seed_810.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    build = args.build.expanduser().resolve()
    executable = executable_in(build)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    counts = [int(value) for value in args.env_counts.split(",")]
    if not counts or any(value < 1 for value in counts):
        raise ValueError("--env-counts must contain positive integers")

    idle_gpu = sample_gpu()
    results = [
        benchmark_count(
            build=build,
            count=count,
            seed=args.seed + 1000 * index,
            time_scale=args.time_scale,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            sample_every=args.sample_every,
        )
        for index, count in enumerate(counts)
    ]
    mps = getattr(torch.backends, "mps", None)
    output = {
        "schema_version": "blackout.prep11_benchmark.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mps_built": bool(mps and mps.is_built()),
            "mps_available": bool(mps and mps.is_available()),
        },
        "environment": {
            "build_path": str(build),
            "executable_path": str(executable),
            "executable_sha256": sha256_file(executable),
            "adapter": "blackout_rl.env.ContractBlackOutEnv",
            "time_scale": args.time_scale,
            "action_policy": "seeded uniform random float32[-1,1]",
        },
        "measurement": {
            "definition": "one environment step is one PettingZoo parallel step for all 10 agents",
            "synchronous_multi_env": True,
            "idle_gpu_baseline": idle_gpu,
            "gpu_scope": "whole-device; not attributable to only Unity/Python",
        },
        "runs": results,
        "full_episode_cost": full_episode_cost(args.episode_log.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
