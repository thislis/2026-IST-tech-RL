"""Explicitly frozen scripted-opponent adapter for one-sided IPPO training."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .scripted_fsm import ScriptedTeamController


class FrozenScriptedOpponent:
    """Own a deterministic scripted controller with immutable training config.

    FSM runtime state changes while acting, but there are no trainable parameters,
    optimizer, or checkpoint updates on this object.
    """

    def __init__(
        self,
        team: int,
        *,
        seed: int = 0,
        enable_special_items: bool = False,
        policy_id: str = "scripted-battery-v1",
    ) -> None:
        self.team = team
        self.seed = seed
        self.policy_id = policy_id
        self.enable_special_items = enable_special_items
        self._frozen_config = {
            "team": team,
            "seed": seed,
            "enable_special_items": enable_special_items,
            "policy_id": policy_id,
        }
        self._controller = self._new_controller()

    def _new_controller(self) -> ScriptedTeamController:
        return ScriptedTeamController(
            self.team,
            seed=self.seed,
            enable_special_items=self.enable_special_items,
        )

    @property
    def trainable_parameters(self) -> tuple[()]:
        return ()

    @property
    def frozen_fingerprint(self) -> str:
        payload = json.dumps(self._frozen_config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        self._controller = self._new_controller()

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]:
        requested = tuple(agents)
        if set(requested) != set(self._controller.agents):
            raise ValueError("frozen scripted opponent must control exactly its configured team")
        fingerprint = self.frozen_fingerprint
        actions = self._controller.act(observations, requested)
        if self.frozen_fingerprint != fingerprint:
            raise RuntimeError("frozen opponent configuration changed while acting")
        return actions
