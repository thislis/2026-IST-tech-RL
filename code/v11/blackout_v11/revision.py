"""Separately pinned continuations of an already registered study.

The original registration remains the training identity. A separately pinned
source revision authorizes each declared change without rewriting checkpoints,
completed evaluations, prior receipts, or frozen runtime measurements.
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

OBSERVABILITY_SOURCES = {
    "code/v11/blackout_v11/runner.py",
    "code/v11/blackout_v11/revision.py",
    "code/v11/blackout_v11/stop_events.py",
}

THROUGHPUT_SCHEMA = "blackout.v11.throughput_revision.1"
THROUGHPUT_ARCHIVE = "throughput_revisions/queue16-v1"
THROUGHPUT_OVERRIDES = {"wave_episodes": 16}
THROUGHPUT_SOURCES = {
    "code/v11/blackout_v11/training.py",
    "code/v11/blackout_v11/runner.py",
    "code/v11/blackout_v11/revision.py",
    "code/v11/blackout_v11/benchmark.py",
    "code/v11/blackout_v11/behavior.py",
    "code/v11/blackout_v11/behavior_reports.py",
}
THROUGHPUT_PROTECTED = (
    "registration.json", "evaluation_revision.json", "observability_revision.json",
    "runtime_profile.json", "task_plan.json", "evaluation_schedule.json",
)
THROUGHPUT_FIX_SCHEMA = "blackout.v11.throughput_fix_revision.1"
THROUGHPUT_FIX_ARCHIVE = "throughput_revisions/evaluation-config-fix-v1"
THROUGHPUT_FIX_SOURCES = {
    "code/v11/blackout_v11/evaluation.py",
    "code/v11/blackout_v11/revision.py",
}


def _throughput_fix_changes(before, current):
    if {k: v for k, v in before.items() if k != "sources"} != {
        k: v for k, v in current.items() if k != "sources"
    }:
        raise ValueError("throughput fix changed config/packages/contracts")
    old, new = before["sources"], current["sources"]
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if not set(changed) <= THROUGHPUT_FIX_SOURCES:
        raise ValueError("non-throughput-fix source changed: " + str(sorted(set(changed) - THROUGHPUT_FIX_SOURCES)))
    return changed


def _throughput_fix_base(directory, current):
    """Pin evaluator config normalization while preserving the queue16 receipt."""
    path = directory / "throughput_fix_revision.json"
    if not path.exists():
        return current
    receipt = read_json(path)
    archive = directory / THROUGHPUT_FIX_ARCHIVE
    before = read_json(archive / "before_source_registration.json")
    changed = _throughput_fix_changes(before, current)
    previous_hash = file_hash(directory / "throughput_revision.json")
    if (receipt.get("schema") != THROUGHPUT_FIX_SCHEMA
            or receipt.get("revision") != "evaluation-config-fix-v1"
            or receipt.get("before_registration_sha256") != digest(before)
            or receipt.get("source_registration_sha256") != digest(current)
            or receipt.get("throughput_revision_sha256") != previous_hash
            or file_hash(archive / "throughput_revision.json") != previous_hash
            or receipt.get("changed_sources") != changed
            or receipt.get("evaluation_config_ignored_keys") != ["wave_episodes"]
            or receipt.get("training_procedure_unchanged") is not True):
        raise ValueError("throughput fix provenance mismatch")
    return before


def create_throughput_fix_revision(directory, cfg):
    """Pin only evaluator config normalization for an inactive failed/stopped study."""
    from .runner import is_active
    directory = Path(directory)
    status = read_json(directory / "status.json")
    if (is_active(directory) or status.get("state") not in ("failed", "stopped")
            or status.get("operation") != "study"):
        raise ValueError("throughput fix requires an inactive failed or stopped study")
    current = registration(cfg)
    if (directory / "throughput_fix_revision.json").exists():
        validated_revision(directory, cfg, current)
        return read_json(directory / "throughput_fix_revision.json")
    archive = directory / THROUGHPUT_FIX_ARCHIVE
    before = read_json(archive / "before_source_registration.json")
    if not (directory / "throughput_revision.json").exists() or validated_revision(directory, cfg, before) is None:
        raise ValueError("throughput fix requires the prior approved queue revision")
    changed = _throughput_fix_changes(before, current)
    _copy_unchanged(directory / "throughput_revision.json", archive / "throughput_revision.json")
    receipt = dict(schema=THROUGHPUT_FIX_SCHEMA, revision="evaluation-config-fix-v1", created_at=time.time(),
        before_registration_sha256=digest(before), source_registration_sha256=digest(current),
        throughput_revision_sha256=file_hash(directory / "throughput_revision.json"), changed_sources=changed,
        evaluation_config_ignored_keys=["wave_episodes"], training_procedure_unchanged=True,
        previous_revision_receipts_preserved=True, completed_evaluations_preserved=True,
        evaluation_schedule_unchanged=True, checkpoint_contents_unchanged=True,
        purpose="Exclude only the training wave_episodes runtime override from evaluator config; retain exact evaluation plan checks",
        resume_command=f"bash code/v11/run_v11_fast.sh --name {directory.name} --resume")
    atomic_json(directory / "throughput_fix_revision.json", receipt)
    validated_revision(directory, cfg, current)
    return receipt


def _throughput_changes(before, current):
    if {k: v for k, v in before.items() if k != "sources"} != {
        k: v for k, v in current.items() if k != "sources"
    }:
        raise ValueError("throughput revision changed config/packages/contracts")
    old, new = before["sources"], current["sources"]
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if not set(changed) <= THROUGHPUT_SOURCES:
        raise ValueError("non-throughput source changed: " + str(sorted(set(changed) - THROUGHPUT_SOURCES)))
    return changed


def _copy_unchanged(source, target):
    """Never replace evidence already captured before the source edits."""
    if target.exists():
        if file_hash(source) != file_hash(target):
            raise ValueError("throughput archive differs from continuation boundary: " + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _throughput_boundary(directory, archive, *, capture=False):
    boundary = read_json(archive / "continuation_boundary.json")
    run = boundary.get("run")
    if not isinstance(run, str) or run in ("", ".", "..") or Path(run).name != run:
        raise ValueError("invalid throughput continuation run")
    pointer = Path("runs") / run / "checkpoints/latest.json"
    archive_pointer = archive / (run + "_latest.json")
    if capture:
        source_pointer = directory / pointer
        value = read_json(source_pointer)
        if value.get("sha256") != boundary.get("checkpoint_sha256") or value.get("step") != boundary.get("step"):
            raise ValueError("throughput checkpoint moved after the captured boundary")
        checkpoint = source_pointer.parent / (value["sha256"] + ".pt")
        if file_hash(checkpoint) != value["sha256"]:
            raise ValueError("throughput checkpoint checksum mismatch")
        _copy_unchanged(source_pointer, archive_pointer)
        _copy_unchanged(checkpoint, archive / checkpoint.name)
    value = read_json(archive_pointer)
    if (value.get("sha256") != boundary.get("checkpoint_sha256")
            or value.get("step") != boundary.get("step")
            or file_hash(archive / (value["sha256"] + ".pt")) != value["sha256"]):
        raise ValueError("throughput archived checkpoint provenance mismatch")
    return dict(pointer=str(pointer), checkpoint_sha256=value["sha256"], step=value["step"],
                archived_pointer_sha256=file_hash(archive_pointer),
                continuation_boundary_sha256=file_hash(archive / "continuation_boundary.json"))


def _throughput_base(directory, current):
    """Peel the queue revision before validating the older logging/evaluation layers."""
    path = directory / "throughput_revision.json"
    if not path.exists():
        return current
    receipt = read_json(path)
    archive = directory / THROUGHPUT_ARCHIVE
    before = read_json(archive / "before_source_registration.json")
    changed = _throughput_changes(before, current)
    protected = {name: file_hash(directory / name) for name in THROUGHPUT_PROTECTED}
    if (receipt.get("schema") != THROUGHPUT_SCHEMA
            or receipt.get("revision") != "queue16-v1"
            or receipt.get("before_registration_sha256") != digest(before)
            or receipt.get("source_registration_sha256") != digest(current)
            or receipt.get("training_registration_sha256") != digest(read_json(directory / "registration.json"))
            or receipt.get("changed_sources") != changed
            or digest(receipt.get("runtime_overrides")) != digest(THROUGHPUT_OVERRIDES)
            or receipt.get("protected_sha256") != protected
            or receipt.get("training_procedure_changed") is not True
            or receipt.get("checkpoint_boundary") != _throughput_boundary(directory, archive)):
        raise ValueError("throughput revision provenance mismatch")
    for name, checksum in protected.items():
        if file_hash(archive / name) != checksum:
            raise ValueError("throughput archived receipt/config provenance mismatch: " + name)
    return before


def throughput_overrides(directory, cfg):
    """Apply only the pinned override; older approved studies retain eight episodes."""
    directory = Path(directory)
    if (directory / "throughput_revision.json").exists():
        if validated_revision(directory, cfg) is None:
            raise ValueError("throughput revision requires its earlier evaluation revision")
        return dict(THROUGHPUT_OVERRIDES)
    if (directory / "evaluation_revision.json").exists():
        validated_revision(directory, cfg)
        return {"wave_episodes": 8}
    # A newly registered source closure uses training.py's default (16).
    # prepare_registration rejects an old, unapproved closure before this call.
    return {}


def create_throughput_revision(directory, cfg):
    """Pin a stopped study's 16-episode continuation without launching training."""
    from .runner import is_active
    directory = Path(directory)
    status = read_json(directory / "status.json")
    if is_active(directory) or status.get("state") != "stopped" or status.get("operation") != "study":
        raise ValueError("throughput continuation requires an inactive stopped study")
    current = registration(cfg)
    if (directory / "throughput_revision.json").exists():
        validated_revision(directory, cfg, current)
        return read_json(directory / "throughput_revision.json")
    archive = directory / THROUGHPUT_ARCHIVE
    before = read_json(archive / "before_source_registration.json")
    if not (directory / "observability_revision.json").exists() or validated_revision(directory, cfg, before) is None:
        raise ValueError("throughput continuation requires the prior evaluation and observability approvals")
    changed = _throughput_changes(before, current)
    boundary = _throughput_boundary(directory, archive, capture=True)
    for name in THROUGHPUT_PROTECTED:
        _copy_unchanged(directory / name, archive / name)
    receipt = dict(schema=THROUGHPUT_SCHEMA, revision="queue16-v1", created_at=time.time(),
        before_registration_sha256=digest(before), source_registration_sha256=digest(current),
        training_registration_sha256=digest(read_json(directory / "registration.json")),
        changed_sources=changed, runtime_overrides=dict(THROUGHPUT_OVERRIDES),
        protected_sha256={name: file_hash(directory / name) for name in THROUGHPUT_PROTECTED},
        checkpoint_boundary=boundary, training_procedure_changed=True,
        procedure_change="16 complete episodes per frozen FIFO wave instead of 8; optimizer batch composition, update frequency, and history cadence change",
        retention_change="incremental behavior retention; existing retention byte limit preserved",
        training_config_unchanged=True, training_budget_unchanged=True, training_seeds_unchanged=True,
        evaluation_schedule_unchanged=True, frozen_runtime_measurements_preserved=True,
        checkpoint_contents_unchanged=True, original_registration_preserved=True,
        previous_revision_receipts_preserved=True, completed_evaluations_preserved=True,
        purpose="Bounded FIFO collection with eight concurrent workers and incremental behavior retention",
        resume_command=f"bash code/v11/run_v11_fast.sh --name {directory.name} --resume")
    atomic_json(directory / "throughput_revision.json", receipt)
    validated_revision(directory, cfg, current)
    return receipt


def _observability_base(directory, current):
    """Validate a separately pinned logging-only revision, retaining prior approval."""
    path = directory / "observability_revision.json"
    if not path.exists():
        return current
    receipt = read_json(path)
    before = read_json(directory / "observability_revisions/stop-events-v1/before_source_registration.json")
    if {k: v for k, v in before.items() if k != "sources"} != {
        k: v for k, v in current.items() if k != "sources"
    }:
        raise ValueError("observability revision changed config/packages/contracts")
    old, new = before["sources"], current["sources"]
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if (receipt.get("schema") != "blackout.v11.observability_revision.1"
            or receipt.get("before_registration_sha256") != digest(before)
            or receipt.get("source_registration_sha256") != digest(current)
            or receipt.get("evaluation_revision_sha256") != file_hash(directory / "evaluation_revision.json")
            or receipt.get("changed_sources") != changed
            or not set(changed) <= OBSERVABILITY_SOURCES):
        raise ValueError("observability revision provenance mismatch")
    return before


def create_observability_revision(directory, cfg):
    """Pin logging-only edits for a stopped study; never rewrite training assets."""
    from .runner import is_active
    directory = Path(directory)
    if is_active(directory) or read_json(directory / "status.json").get("state") != "stopped":
        raise ValueError("observability continuation requires a stopped study")
    current = registration(cfg)
    if (directory / "observability_revision.json").exists():
        validated_revision(directory, cfg, current)
        return read_json(directory / "observability_revision.json")
    before = read_json(directory / "observability_revisions/stop-events-v1/before_source_registration.json")
    validated_revision(directory, cfg, before)
    if {k: v for k, v in before.items() if k != "sources"} != {
        k: v for k, v in current.items() if k != "sources"
    }:
        raise ValueError("observability revision changed config/packages/contracts")
    old, new = before["sources"], current["sources"]
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if not set(changed) <= OBSERVABILITY_SOURCES:
        raise ValueError("non-observability source changed: " + str(changed))
    receipt = dict(schema="blackout.v11.observability_revision.1", created_at=time.time(),
                   before_registration_sha256=digest(before), source_registration_sha256=digest(current),
                   evaluation_revision_sha256=file_hash(directory / "evaluation_revision.json"),
                   changed_sources=changed, training_and_evaluation_unchanged=True,
                   purpose="Durable stop/signal/resource records; existing limits and training preserved")
    atomic_json(directory / "observability_revision.json", receipt)
    validated_revision(directory, cfg, current)
    return receipt


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
    approved = _observability_base(directory, _throughput_base(directory, _throughput_fix_base(directory, current)))
    changed = _changes(original, approved)
    if (receipt.get("schema") != SCHEMA
            or receipt.get("training_registration_sha256") != digest(original)
            or receipt.get("source_registration_sha256") != digest(approved)
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
