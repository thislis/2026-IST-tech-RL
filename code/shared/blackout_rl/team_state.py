"""Team-local unit state and stable scripted-role tracking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

import numpy as np

from .batching import N_TEAM_AGENTS, team_agents
from .observation import parse_vector
from .semantic_map import GridCell, normalized_to_cell


class Role(str, Enum):
    WORKER = "worker"
    GUARD = "guard"
    CARRIER = "carrier"


DEFAULT_ROLES = (
    Role.WORKER,
    Role.WORKER,
    Role.WORKER,
    Role.GUARD,
    Role.CARRIER,
)


def default_roles(team: int) -> dict[str, Role]:
    return dict(zip(team_agents(team), DEFAULT_ROLES))


@dataclass(frozen=True)
class UnitState:
    agent: str
    team: int
    slot_id: int
    unit_index: int
    position_normalized: tuple[float, float]
    cell: GridCell
    holding_item_id: int
    class_id: int
    role: Role
    target: GridCell | None
    step: int

    @property
    def is_carrying(self) -> bool:
        return self.holding_item_id != 0


@dataclass(frozen=True)
class TeamSnapshot:
    team: int
    step: int
    units: tuple[UnitState, ...]
    own_score: float
    opponent_score: float
    time_left: float

    def by_agent(self) -> dict[str, UnitState]:
        return {unit.agent: unit for unit in self.units}


class TeamStateTracker:
    """Build team-local snapshots without relying on observation dict order."""

    def __init__(
        self,
        team: int,
        *,
        width_cells: int = 24,
        height_cells: int = 24,
        roles: Mapping[str, Role] | None = None,
    ) -> None:
        self.team = team
        self.agents = team_agents(team)
        self.width_cells = width_cells
        self.height_cells = height_cells
        self.roles = dict(default_roles(team) if roles is None else roles)
        if set(self.roles) != set(self.agents):
            raise ValueError("roles must define exactly the team's five canonical agents")
        if any(not isinstance(role, Role) for role in self.roles.values()):
            raise TypeError("all role values must be Role members")
        self._targets: dict[str, GridCell | None] = {
            agent: None for agent in self.agents
        }
        self._latest: TeamSnapshot | None = None

    @property
    def latest(self) -> TeamSnapshot | None:
        return self._latest

    def set_target(self, agent: str, target: GridCell | None) -> None:
        if agent not in self._targets:
            raise KeyError(f"agent is not on team {self.team}: {agent}")
        self._targets[agent] = target
        if self._latest is not None:
            self._latest = replace(
                self._latest,
                units=tuple(
                    replace(unit, target=target) if unit.agent == agent else unit
                    for unit in self._latest.units
                ),
            )

    def set_targets(self, targets: Mapping[str, GridCell | None]) -> None:
        unknown = set(targets) - set(self.agents)
        if unknown:
            raise KeyError(f"target assignment contains unknown agents: {sorted(unknown)}")
        for agent, target in targets.items():
            self.set_target(agent, target)

    def clear_targets(self) -> None:
        for agent in self.agents:
            self.set_target(agent, None)

    def update(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        *,
        step: int,
    ) -> TeamSnapshot:
        missing = [agent for agent in self.agents if agent not in observations]
        if missing:
            raise KeyError(f"team state observations missing agents: {missing}")
        units = []
        parsed_first = None
        for slot_id, agent in enumerate(self.agents):
            parsed = parse_vector(observations[agent]["vector"])
            if parsed_first is None:
                parsed_first = parsed
            unit_index = self.team * N_TEAM_AGENTS + slot_id
            position_array = parsed.positions[unit_index]
            position = (float(position_array[0]), float(position_array[1]))
            units.append(
                UnitState(
                    agent=agent,
                    team=self.team,
                    slot_id=slot_id,
                    unit_index=unit_index,
                    position_normalized=position,
                    cell=normalized_to_cell(
                        position,
                        width_cells=self.width_cells,
                        height_cells=self.height_cells,
                    ),
                    holding_item_id=int(parsed.holding_item_ids[unit_index]),
                    class_id=parsed.self_class_id,
                    role=self.roles[agent],
                    target=self._targets[agent],
                    step=step,
                )
            )
        assert parsed_first is not None
        snapshot = TeamSnapshot(
            team=self.team,
            step=step,
            units=tuple(units),
            own_score=parsed_first.own_score,
            opponent_score=parsed_first.opponent_score,
            time_left=parsed_first.time_left,
        )
        self._latest = snapshot
        return snapshot
