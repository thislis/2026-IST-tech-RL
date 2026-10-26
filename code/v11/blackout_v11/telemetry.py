"""Owned-process resource measurements and a conservative experiment ETA."""
import json
import os
import platform
from pathlib import Path
import subprocess
import statistics
import torch
from .io import ROOT, read_json


def hardware_identity():
    result = dict(platform=platform.platform(), machine=platform.machine(),
                  cpu_count=os.cpu_count(), torch=str(torch.__version__),
                  mps_built=torch.backends.mps.is_built(),
                  mps_available=torch.backends.mps.is_available())
    try:
        probe = subprocess.run(["system_profiler", "SPHardwareDataType", "-json"],
                               capture_output=True, text=True, timeout=15, check=True)
        row = json.loads(probe.stdout)["SPHardwareDataType"][0]
        result["hardware"] = {k: row.get(k) for k in
                              ("chip_type", "machine_model", "number_processors", "physical_memory")}
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        result["hardware_probe_unavailable"] = True
    return result


def memory_pressure():
    try:
        result = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                capture_output=True, text=True, timeout=2, check=True)
        level = int(result.stdout.strip())
        return dict(verified=True, level=level, critical=level >= 4)
    except (OSError, subprocess.SubprocessError, ValueError):
        return dict(verified=False, level=None, critical=False)


def mps_memory():
    if not torch.backends.mps.is_available():
        return dict(available=False, allocated_bytes=0, driver_bytes=0)
    return dict(available=True, allocated_bytes=torch.mps.current_allocated_memory(),
                driver_bytes=torch.mps.driver_allocated_memory(),
                recommended_bytes=torch.mps.recommended_max_memory())


def estimate(directory, cfg):
    directory = Path(directory)
    target = (cfg.get("teacher_steps", 262144) + len(cfg["arms"]) * cfg["pilot_steps"] +
              (len(cfg["arms"]) * len(cfg["seeds"]) + 2 * len(cfg["confirmation_seeds"])) * cfg["training_steps"])
    if cfg.get("ablation"):
        settings = read_json(ROOT / "code/v11/configs/ablations.json")
        target = cfg.get("teacher_steps", 262144) + 2 * len(settings["seeds"]) * settings["training_steps"]
    completed, rates = 0, []
    for path in directory.glob("**/progress.json"):
        row = read_json(path)
        completed += int(row.get("step", row.get("env_steps", 0)))
    for path in directory.glob("**/training.jsonl"):
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 128 * 1024))
            lines = stream.read().splitlines()
        if size > 128 * 1024:
            lines = lines[1:]
        for line in lines[-8:]:
            try:
                value = json.loads(line).get("end_to_end_steps_per_second")
                if value and value > 0:
                    rates.append(float(value))
            except ValueError:
                pass
    rate = statistics.median(rates[-16:]) if rates else None
    remaining = max(0, target - completed)
    return dict(target_steps=target, completed_target_steps=completed,
                remaining_target_steps=remaining, training_step_rate=rate,
                remaining_training_seconds=None if rate is None else remaining / rate,
                includes_evaluation=False,
                caveat="Measured complete-wave rate; evaluation and episode overrun are additional.")
