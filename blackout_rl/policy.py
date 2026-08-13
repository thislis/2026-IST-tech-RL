"""Small policy primitives shared by smoke tests and scripted evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .logging_schema import sha256_file


@runtime_checkable
class ActionPolicy(Protocol):
    """Team action interface used by evaluators and scripted controllers."""

    seed: int

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]: ...


def _require_agents(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    agents: Sequence[str],
) -> tuple[str, ...]:
    ordered = tuple(agents)
    missing = [agent for agent in ordered if agent not in observations]
    if missing:
        raise KeyError(f"policy observations missing agents: {missing}")
    return ordered


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
        ordered = _require_agents(observations, agents)
        return {
            agent: self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            for agent in ordered
        }


class NoOpPolicy:
    """Deterministic policy that stops every requested unit."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return {
            agent: np.zeros(2, dtype=np.float32)
            for agent in _require_agents(observations, agents)
        }


class FixedDirectionPolicy:
    """Move every requested unit in one normalized, fixed direction."""

    def __init__(self, direction: Sequence[float], seed: int = 0):
        action = np.asarray(direction, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("direction must be a finite 2-vector")
        norm = float(np.linalg.norm(action))
        if norm == 0.0:
            raise ValueError("FixedDirectionPolicy direction must be non-zero; use NoOpPolicy")
        self.direction = (action / norm).astype(np.float32)
        self.seed = seed

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return {
            agent: self.direction.copy()
            for agent in _require_agents(observations, agents)
        }


class DeterministicCheckpointPolicy:
    """Canonical-team adapter for a registered IPPO checkpoint.

    A checkpoint may explicitly declare a deterministic safety controller. This
    is intentionally checkpoint metadata (and therefore auditable), rather than
    a hidden evaluator option. The neural policy remains packaged in the same
    artifact for subsequent residual/PPO fine-tuning.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        team: int,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        from .model_contract import CanonicalTeamModel, load_checkpoint

        model, payload = load_checkpoint(checkpoint, device=device)
        self._adapter = CanonicalTeamModel(model, team=team, device=device)
        self._safety_controller = None
        guardrail = payload.get("inference_guardrail")
        if guardrail is not None:
            if guardrail.get("version") != "scripted_counter_v1":
                raise ValueError("unsupported checkpoint inference guardrail")
            if guardrail.get("mode") != "planner_override":
                raise ValueError("unsupported checkpoint guardrail mode")
            from .scripted_fsm import ScriptedTeamController
            from .team_state import Role

            roles = tuple(Role(value) for value in guardrail["roles"])
            self._safety_controller = ScriptedTeamController(
                team,
                seed=seed,
                enable_special_items=False,
                roles=roles,
                chase_radius_cells=int(guardrail["chase_radius_cells"]),
            )
        self.checkpoint_payload = payload
        self.team = team
        self.seed = seed

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]:
        from .batching import team_agents

        expected = team_agents(self.team)
        if tuple(agents) != expected:
            raise ValueError(f"checkpoint policy expected canonical agents {expected}")
        if self._safety_controller is not None:
            return self._safety_controller.act(observations, expected)
        selected = {agent: observations[agent] for agent in expected}
        return self._adapter.act(selected)

    def reset(self) -> None:
        self._adapter._policy.reset_inference_state()
        if self._safety_controller is not None:
            self._safety_controller.reset()
