"""Frozen-opponent snapshot self-play and cross-generation evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .logging_schema import sha256_file
from .policy import DeterministicCheckpointPolicy


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    generation: int
    global_step: int

    def validate_artifact(self) -> None:
        if self.generation < 0 or self.global_step < 0 or not self.snapshot_id:
            raise ValueError("invalid snapshot metadata")
        if sha256_file(Path(self.checkpoint_path)) != self.checkpoint_sha256:
            raise ValueError(f"snapshot artifact changed: {self.snapshot_id}")


class FrozenCheckpointOpponent:
    """Inference-only checkpoint adapter with no optimizer or mutable training state."""

    def __init__(self, snapshot: Snapshot, *, team: int, learner_sha256: str | None = None,
                 device: str = "cpu") -> None:
        snapshot.validate_artifact()
        if learner_sha256 is not None and snapshot.checkpoint_sha256 == learner_sha256:
            raise ValueError("opponent must be a frozen snapshot, not the live learner")
        self.snapshot = snapshot
        self.team = team
        self._policy = DeterministicCheckpointPolicy(snapshot.checkpoint_path, team=team, device=device)

    @property
    def trainable_parameters(self) -> tuple[()]: return ()

    @property
    def optimizer_state(self) -> None: return None

    def reset(self) -> None: pass

    def act(self, observations: Mapping[str, Mapping[str, Any]], agents: Sequence[str]) -> dict:
        before = self.snapshot.checkpoint_sha256
        result = self._policy.act(observations, agents)
        if sha256_file(Path(self.snapshot.checkpoint_path)) != before:
            raise RuntimeError("frozen opponent checkpoint changed while acting")
        return result


class SnapshotPool:
    def __init__(self, *, capacity: int = 10, latest_probability: float = .5, seed: int = 0) -> None:
        if capacity <= 0 or not 0 <= latest_probability <= 1:
            raise ValueError("invalid snapshot pool configuration")
        self.capacity = capacity; self.latest_probability = latest_probability
        self._rng = random.Random(seed); self._snapshots: list[Snapshot] = []

    @property
    def snapshots(self) -> tuple[Snapshot, ...]: return tuple(self._snapshots)

    def add(self, snapshot: Snapshot) -> None:
        snapshot.validate_artifact()
        if any(row.snapshot_id == snapshot.snapshot_id for row in self._snapshots):
            raise ValueError(f"duplicate snapshot: {snapshot.snapshot_id}")
        if self._snapshots and snapshot.generation <= self._snapshots[-1].generation:
            raise ValueError("snapshot generations must increase")
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.capacity: self._snapshots.pop(0)

    def sample(self) -> Snapshot:
        if not self._snapshots: raise ValueError("cannot sample an empty snapshot pool")
        if self._rng.random() < self.latest_probability: return self._snapshots[-1]
        return self._rng.choice(self._snapshots)


class SideBalancedSampler:
    """Alternate learner side so exposure differs by at most one at all times."""

    def __init__(self, *, first_side: int = 0) -> None:
        if first_side not in (0,1): raise ValueError("first_side must be 0 or 1")
        self.next_side = first_side; self.counts = [0,0]

    def sample(self) -> int:
        side = self.next_side; self.counts[side] += 1; self.next_side = 1-side
        return side

    @property
    def balanced(self) -> bool: return abs(self.counts[0]-self.counts[1]) <= 1


@dataclass(frozen=True)
class StabilityThresholds:
    minimum_entropy: float = .25
    maximum_kl: float = .05
    maximum_value_error: float = 10.0


def self_play_stable(metrics: Sequence[Mapping[str, float]], thresholds: StabilityThresholds) -> bool:
    if not metrics: raise ValueError("stability check requires update metrics")
    return all(row["entropy"] >= thresholds.minimum_entropy
               and row["approximate_kl"] <= thresholds.maximum_kl
               and row["value_error"] <= thresholds.maximum_value_error for row in metrics)


@dataclass(frozen=True)
class MatchupResult:
    model_id: str
    opponent_id: str
    pair_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    win_rate: float
    mean_score_diff: float


def checkpoint_evaluation_matrix(
    checkpoint_ids: Sequence[str], results: Sequence[MatchupResult]
) -> dict[str, Any]:
    ids = tuple(checkpoint_ids)
    if len(ids) != len(set(ids)) or not ids: raise ValueError("checkpoint IDs must be unique")
    lookup = {(row.model_id,row.opponent_id): row for row in results}
    matrix: dict[str, dict[str, Any]] = {}
    for model in ids:
        matrix[model] = {}
        for opponent in ids:
            if model == opponent: continue
            if (model,opponent) not in lookup: raise ValueError(f"missing matchup {model} vs {opponent}")
            row = lookup[(model,opponent)]
            if len(row.seeds) != len(row.pair_ids) or len(set(row.seeds)) != len(row.seeds):
                raise ValueError("each matchup must contain unique paired seeds")
            matrix[model][opponent] = asdict(row)
    return matrix


def past_opponent_regressions(
    candidate_results: Mapping[str, float], reference_results: Mapping[str, float],
    *, tolerance: float = .05,
) -> tuple[str, ...]:
    if set(candidate_results) != set(reference_results):
        raise ValueError("candidate and reference must use the same past-opponent suite")
    return tuple(sorted(opponent for opponent in candidate_results
                        if candidate_results[opponent] + tolerance < reference_results[opponent]))


@dataclass(frozen=True)
class OpponentSuite:
    random: str
    scripted: str
    ippo: str
    historical_mappo: tuple[str, ...]
    latest_candidate: str

    def all_ids(self) -> tuple[str, ...]:
        ids = (self.random, self.scripted, self.ippo, *self.historical_mappo, self.latest_candidate)
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("opponent suite IDs must be non-empty and unique")
        return ids


def should_expand_to_psro(
    *, generations: int, plateau_generations: int, cyclic_regressions: int,
    min_generations: int = 8, min_plateau: int = 4,
) -> bool:
    """Decision gate; no population expansion before snapshot self-play saturates."""
    return (generations >= min_generations and plateau_generations >= min_plateau
            and cyclic_regressions >= 2)


def snapshot_from_checkpoint(path: str | Path, *, generation: int, global_step: int) -> Snapshot:
    checkpoint = Path(path)
    digest = sha256_file(checkpoint)
    return Snapshot(f"gen-{generation}-{digest[:10]}", str(checkpoint), digest, generation, global_step)
