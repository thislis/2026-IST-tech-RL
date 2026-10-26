"""Explicit evaluation-only continuation of an already registered study.

The original registration remains the training identity. A separately pinned
source revision authorizes the evaluation schedule and interruption accounting
changes without rewriting checkpoints, completed evaluations, or measurements.
"""
from pathlib import Path
import shutil
import time
from .contracts import registration
from .io import atomic_json, digest, file_hash, read_json

SCHEMA = "blackout.v11.evaluation_revision.1"
ALLOWED_SOURCES = {
    "code/v11/blackout_v11/study.py",
    "code/v11/blackout_v11/evaluation.py",
    "code/v11/blackout_v11/evaluation_schedule.py",
    "code/v11/blackout_v11/training.py",
    "code/v11/blackout_v11/runner.py",
    "code/v11/blackout_v11/revision.py",
}


def _changes(original, current):
    if {k: v for k, v in original.items() if k != "sources"} != {
        k: v for k, v in current.items() if k != "sources"
    }:
        raise ValueError("evaluation revision requires identical training config, packages and contracts")
    old, new = original["sources"], current["sources"]
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if not set(changed) <= ALLOWED_SOURCES:
        raise ValueError("non-evaluation source changed: " + str(sorted(set(changed) - ALLOWED_SOURCES)))
    return changed


def validated_revision(directory, cfg, current=None):
    directory = Path(directory)
    path = directory / "evaluation_revision.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    original = read_json(directory / "registration.json")
    current = registration(cfg) if current is None else current
    changed = _changes(original, current)
    if (receipt.get("schema") != SCHEMA
            or receipt.get("training_registration_sha256") != digest(original)
            or receipt.get("source_registration_sha256") != digest(current)
            or receipt.get("changed_sources") != changed):
        raise ValueError("evaluation revision provenance mismatch; source/config changed after approval")
    if receipt.get("runtime_profile_sha256") != file_hash(directory / "runtime_profile.json"):
        raise ValueError("frozen runtime measurements changed after evaluation revision")
    if receipt.get("task_plan_sha256") != file_hash(directory / "task_plan.json"):
        raise ValueError("original training study plan changed after evaluation revision")
    from .evaluation_schedule import build_evaluation_schedule
    schedule = read_json(directory / "evaluation_schedule.json")
    if (receipt.get("evaluation_schedule_sha256") != digest(schedule)
            or schedule != build_evaluation_schedule(cfg)):
        raise ValueError("approved evaluation schedule changed")
    return receipt


def resumed_runtime_profile(directory, cfg):
    """Reuse measured training settings only for the exact approved revision."""
    receipt = validated_revision(directory, cfg)
    if receipt is None:
        return None
    from .telemetry import hardware_identity
    profile = read_json(Path(directory) / "runtime_profile.json")
    if profile["hardware"] != hardware_identity():
        raise ValueError("hardware changed since the frozen training profile; cannot silently retune a resumed study")
    return profile


def create_evaluation_revision(directory, cfg):
    """Prepare continuation only; never launch workers or modify a checkpoint."""
    from .runner import is_active
    directory = Path(directory)
    if is_active(directory):
        raise ValueError("stop the study before preparing an evaluation revision")
    if (directory / "evaluation_revision.json").exists():
        return validated_revision(directory, cfg)
    status = read_json(directory / "status.json")
    if status.get("state") != "stopped" or status.get("operation") != "study":
        raise ValueError("evaluation continuation requires a normally stopped study")
    original = read_json(directory / "registration.json")
    current = registration(cfg)
    changed = _changes(original, current)
    # This migration preserves completed baseline/pilots and only revises
    # evaluations that have not begun. Never reinterpret existing main scores.
    if any(directory.glob("runs/*/evaluation/**/plan.json")):
        raise ValueError("main evaluation already started; a separate evaluation revision directory is required")
    if any((directory / "evaluation" / split).exists() for split in ("confirmation", "test")):
        raise ValueError("confirmation/test already started; cannot change their schedule in place")
    from .evaluation_schedule import record_evaluation_schedule
    schedule, schedule_hash = record_evaluation_schedule(directory, cfg)
    archive = directory / "evaluation_revisions" / "reduced-v1"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("registration.json", "runtime_profile.json", "resolved_config.json",
                 "task_plan.json", "status.json", "stage.json"):
        source = directory / name
        if source.exists():
            shutil.copy2(source, archive / name)
    atomic_json(archive / "source_registration.json", current)
    checkpoints = []
    for pointer in sorted(directory.glob("**/checkpoints/latest.json")):
        if archive in pointer.parents:
            continue
        value = read_json(pointer)
        checkpoint = pointer.parent / (value["sha256"] + ".pt")
        if file_hash(checkpoint) != value["sha256"]:
            raise ValueError("checkpoint checksum mismatch before evaluation revision")
        relative = pointer.relative_to(directory)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pointer, target)
        # The training runner keeps only recent commits, so retain this precise
        # continuation point independently of future checkpoint cleanup.
        shutil.copy2(checkpoint, target.parent / checkpoint.name)
        checkpoints.append(dict(pointer=str(relative), checkpoint_sha256=value["sha256"], step=value["step"]))
    receipt = dict(schema=SCHEMA, created_at=time.time(),
        training_registration_sha256=digest(original), source_registration_sha256=digest(current),
        changed_sources=changed, runtime_profile_sha256=file_hash(directory / "runtime_profile.json"),
        task_plan_sha256=file_hash(directory / "task_plan.json"), archive=str(archive),
        evaluation_schedule_sha256=schedule_hash,
        remaining_evaluation_games=schedule["total_remaining_evaluation_games"],
        checkpoints=checkpoints, training_config_unchanged=True, training_budget_unchanged=True,
        training_seeds_unchanged=True, checkpoint_contents_unchanged=True,
        original_registration_preserved=True, completed_evaluations_preserved=True,
        interrupted_attempt_accounting="ledger-evidenced user interruptions are separated from technical failures on resume",
        evaluation_policy="reduced-v1: main dev 5 maps; all five seeds on five core opponents at 10 maps; selected submission only on full league, 10 maps / v4+v6 30 maps",
        resume_command=f"bash code/v11/run_v11_fast.sh --name {directory.name} --resume")
    atomic_json(directory / "evaluation_revision.json", receipt)
    return validated_revision(directory, cfg, current)
