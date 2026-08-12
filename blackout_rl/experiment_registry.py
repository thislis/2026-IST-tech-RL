"""Checkpoint-bound experiment identity and append-only registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .logging_schema import sha256_file, validate_sha256
from .model_contract import save_checkpoint, validate_checkpoint


EXPERIMENT_REGISTRY_SCHEMA_VERSION = "blackout.experiment_registry.v1"
EXPERIMENT_ENTRY_SCHEMA_VERSION = "blackout.experiment_entry.v1"


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentIdentity:
    run_id: str
    config: dict[str, Any]
    seed: int
    git_sha: str
    opponent_id: str

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if len(self.git_sha) != 40:
            raise ValueError("git_sha must be a full 40-character commit")
        try:
            int(self.git_sha, 16)
        except ValueError as exc:
            raise ValueError("git_sha must be hexadecimal") from exc
        if not self.opponent_id:
            raise ValueError("opponent_id must not be empty")
        canonical_config_sha256(self.config)

    def checkpoint_metadata(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def register_checkpoint(
    checkpoint_path: str | Path,
    payload: Mapping[str, Any],
    *,
    registry_path: str | Path,
) -> dict[str, Any]:
    """Save a provenance-complete checkpoint and append its immutable registry entry."""
    validate_checkpoint(payload)
    if "experiment" not in payload:
        raise ValueError("registered checkpoint must contain experiment metadata")
    identity = ExperimentIdentity(**dict(payload["experiment"]))
    identity.validate()
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint, payload)
    checkpoint_sha = sha256_file(checkpoint)
    entry = {
        "schema_version": EXPERIMENT_ENTRY_SCHEMA_VERSION,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": identity.run_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": canonical_config_sha256(identity.config),
        "config": identity.config,
        "seed": identity.seed,
        "git_sha": identity.git_sha,
        "opponent_id": identity.opponent_id,
        "global_step": int(payload["training"]["global_step"]),
    }
    validate_registry_entry(entry)
    registry = Path(registry_path)
    registry.parent.mkdir(parents=True, exist_ok=True)
    existing = read_registry(registry) if registry.exists() else []
    if any(item["run_id"] == identity.run_id for item in existing):
        raise ValueError(f"duplicate experiment run_id: {identity.run_id}")
    with registry.open("a", encoding="utf-8") as output:
        output.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    return entry


def validate_registry_entry(entry: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "registered_at_utc", "run_id", "checkpoint_path",
        "checkpoint_sha256", "config_sha256", "config", "seed", "git_sha",
        "opponent_id", "global_step",
    }
    missing = required - set(entry)
    if missing:
        raise ValueError(f"registry entry missing keys: {sorted(missing)}")
    if entry["schema_version"] != EXPERIMENT_ENTRY_SCHEMA_VERSION:
        raise ValueError("unsupported experiment entry schema")
    validate_sha256(entry["checkpoint_sha256"], "checkpoint_sha256")
    validate_sha256(entry["config_sha256"], "config_sha256")
    if canonical_config_sha256(entry["config"]) != entry["config_sha256"]:
        raise ValueError("registry config hash mismatch")
    ExperimentIdentity(
        run_id=str(entry["run_id"]), config=dict(entry["config"]),
        seed=int(entry["seed"]), git_sha=str(entry["git_sha"]),
        opponent_id=str(entry["opponent_id"]),
    ).validate()


def read_registry(path: str | Path) -> list[dict[str, Any]]:
    registry = Path(path)
    entries = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    for entry in entries:
        validate_registry_entry(entry)
    run_ids = [entry["run_id"] for entry in entries]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("experiment registry contains duplicate run IDs")
    return entries
