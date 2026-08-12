"""Leakage-safe seed splits and paired-seed uncertainty estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SEED_SPLIT_SCHEMA_VERSION = "blackout.seed_splits.v1"


@dataclass(frozen=True)
class SeedSplits:
    train: tuple[int, ...]
    dev: tuple[int, ...]
    test: tuple[int, ...]

    def validate(self) -> None:
        groups = {"train": self.train, "dev": self.dev, "test": self.test}
        for name, values in groups.items():
            if not values or any(not isinstance(seed, int) or seed < 0 for seed in values):
                raise ValueError(f"{name} seeds must be non-empty non-negative integers")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} seeds must be unique")
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                raise ValueError(f"seed leakage between {left} and {right}: {sorted(overlap)}")

    def seeds_for(self, split: str, *, purpose: str) -> tuple[int, ...]:
        """Return a split while preventing test-driven model selection."""
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if purpose not in {"training", "model_selection", "final_report"}:
            raise ValueError("unknown seed-use purpose")
        if split == "test" and purpose != "final_report":
            raise PermissionError("test seeds may only be used for the final report")
        if split == "dev" and purpose == "training":
            raise PermissionError("dev seeds must not be used for training updates")
        return getattr(self, split)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": SEED_SPLIT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedSplits":
        if payload.get("schema_version") != SEED_SPLIT_SCHEMA_VERSION:
            raise ValueError("unsupported seed split schema")
        result = cls(*(tuple(int(seed) for seed in payload[name]) for name in ("train", "dev", "test")))
        result.validate()
        return result


def write_seed_splits(path: str | Path, splits: SeedSplits) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PairedBootstrapCI:
    metric: str
    pairs: int
    estimate: float
    confidence: float
    lower: float
    upper: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pair_values(
    episodes: Sequence[Mapping[str, Any]], metric: str
) -> np.ndarray:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        pair_id = str(episode["pair_id"])
        grouped.setdefault(pair_id, []).append(episode)
    values: list[float] = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or {int(row["side_assignment"]["model_team"]) for row in pair} != {0, 1}:
            raise ValueError(f"pair {pair_id} must contain one game on each physical side")
        if len({int(row["seed"]) for row in pair}) != 1:
            raise ValueError(f"pair {pair_id} contains different environment seeds")
        if metric == "win_rate":
            values.append(sum(row["model_result"] == "win" for row in pair) / 2.0)
        elif metric == "score_diff":
            values.append(float(np.mean([row["score"]["model_minus_opponent"] for row in pair])))
        else:
            raise ValueError("metric must be win_rate or score_diff")
    if not values:
        raise ValueError("paired bootstrap requires at least one seed pair")
    return np.asarray(values, dtype=np.float64)


def paired_seed_bootstrap_ci(
    episodes: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapCI:
    """Percentile bootstrap that samples complete side-swapped seed pairs."""
    if not 0.0 < confidence < 1.0 or resamples <= 0 or seed < 0:
        raise ValueError("invalid bootstrap configuration")
    values = _pair_values(episodes, metric)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    draws = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return PairedBootstrapCI(
        metric=metric,
        pairs=len(values),
        estimate=float(values.mean()),
        confidence=confidence,
        lower=float(lower),
        upper=float(upper),
        resamples=resamples,
        seed=seed,
    )
