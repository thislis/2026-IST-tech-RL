"""Minimal deterministic policy interfaces used by preparation evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .logging_schema import sha256_file


@dataclass(frozen=True)
class PolicyArtifact:
    policy_id: str
    kind: str
    checkpoint_path: str
    checkpoint_sha256: str

    @classmethod
    def from_file(cls, policy_id: str, kind: str, path: str | Path) -> "PolicyArtifact":
        resolved = Path(path).expanduser().resolve()
        return cls(
            policy_id=policy_id,
            kind=kind,
            checkpoint_path=str(resolved),
            checkpoint_sha256=sha256_file(resolved),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "policy_id": self.policy_id,
            "kind": self.kind,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


class RandomPolicy:
    """Uniform random `float32[2]` actions with an isolated RNG stream."""

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return {
            agent: self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            for agent in agents
        }
