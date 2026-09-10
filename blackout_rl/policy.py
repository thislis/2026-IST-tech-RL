"""Small policy primitives shared by smoke tests and scripted evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

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

    _v6_policy = None

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
        self._v6_policy = None
        if payload.get("v6_schema") is not None:
            from .mappo_v6 import V6Policy, model_from_payload

            self._v6_policy = V6Policy(model_from_payload(payload, device), team=team, seed=seed)
            self.checkpoint_payload = payload
            self.team, self.seed = team, seed
            return
        self._adapter = CanonicalTeamModel(model, team=team, device=device)
        self._safety_controller = None
        self._guardrail_mode = None
        self._planner_residual_v2_head = None
        self._device = torch.device(device)
        guardrail = payload.get("inference_guardrail")
        if guardrail is not None:
            if guardrail.get("version") != "scripted_counter_v1":
                raise ValueError("unsupported checkpoint inference guardrail")
            if guardrail.get("mode") not in {
                "planner_override",
                "planner_residual_v1",
                "planner_residual_v2",
            }:
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
            self._guardrail_mode = str(guardrail["mode"])
            if self._guardrail_mode == "planner_residual_v2":
                from .mappo import PlannerConditionedResidualHead

                state = payload.get("planner_residual_v2_state")
                if not isinstance(state, Mapping):
                    raise ValueError("planner residual v2 checkpoint is missing its head state")
                head = PlannerConditionedResidualHead(
                    int(model.actor_critic.model_config["hidden_dim"]),
                    fallback_logit_bias=float(guardrail["fallback_logit_bias"]),
                ).to(self._device)
                head.load_state_dict(state, strict=True)
                self._planner_residual_v2_head = head.eval()
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
        if self._v6_policy is not None:
            return self._v6_policy.act(observations, agents)
        if tuple(agents) != expected:
            raise ValueError(f"checkpoint policy expected canonical agents {expected}")
        if self._guardrail_mode == "planner_override":
            assert self._safety_controller is not None
            return self._safety_controller.act(observations, expected)
        if self._guardrail_mode == "planner_residual_v2":
            from .behavior_cloning import action_vector_to_index
            from .mappo import planner_relative_residual_action, planner_residual_context
            from .model_contract import team_model_input

            assert self._safety_controller is not None
            assert self._planner_residual_v2_head is not None
            planner_actions = self._safety_controller.act(observations, expected)
            selected = {agent: observations[agent] for agent in expected}
            batch = team_model_input(selected, self.team, device=self._device)
            planner_index = torch.tensor(
                [action_vector_to_index(planner_actions[agent]) for agent in expected],
                dtype=torch.int64,
                device=self._device,
            )
            with torch.inference_mode():
                latent = self._adapter._policy.actor_critic.encode(
                    batch.vector, batch.graphic, batch.slot_id
                )
                context = planner_residual_context(
                    batch.vector,
                    batch.graphic,
                    batch.slot_id,
                    planner_index,
                )
                logits = self._planner_residual_v2_head(latent, context)
                residual_index = torch.argmax(logits, dim=-1)
                override_rows = torch.nonzero(residual_index != 0, as_tuple=False).flatten()
                max_overrides = int(
                    self.checkpoint_payload["inference_guardrail"].get(
                        "max_overrides_per_step", 1
                    )
                )
                if len(override_rows) > max_overrides:
                    selected_logits = logits[override_rows, residual_index[override_rows]]
                    fallback_logits = logits[override_rows, 0]
                    keep = override_rows[
                        torch.topk(selected_logits - fallback_logits, max_overrides).indices
                    ]
                    allowed = torch.zeros_like(residual_index, dtype=torch.bool)
                    allowed[keep] = True
                    residual_index = torch.where(
                        allowed, residual_index, torch.zeros_like(residual_index)
                    )
                residual_actions = planner_relative_residual_action(
                    residual_index, planner_index
                ).cpu().numpy()
            return {
                agent: (
                    planner_actions[agent]
                    if int(residual_index[row]) == 0
                    else residual_actions[row].astype(np.float32, copy=False)
                )
                for row, agent in enumerate(expected)
            }

        selected = {agent: observations[agent] for agent in expected}
        neural_actions = self._adapter.act(selected)
        if self._guardrail_mode != "planner_residual_v1":
            return neural_actions
        assert self._safety_controller is not None
        planner_actions = self._safety_controller.act(observations, expected)
        return {
            agent: (
                planner_actions[agent]
                if float(np.linalg.norm(neural_actions[agent])) <= 1e-6
                else neural_actions[agent]
            )
            for agent in expected
        }

    def reset(self) -> None:
        if self._v6_policy is not None:
            self._v6_policy.reset()
            return
        self._adapter._policy.reset_inference_state()
        if self._safety_controller is not None:
            self._safety_controller.reset()
