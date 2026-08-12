"""Stuck detection and path-distance task assignment for scripted teams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .navigation import AStarPlanner, PathNotFound, path_cost
from .semantic_map import GridCell
from .team_state import Role, UnitState


@dataclass(frozen=True)
class StuckEvent:
    agent: str
    step: int
    target: GridCell
    displacement_normalized: float
    commanded_action_magnitude: float
    consecutive_events: int


@dataclass
class _MotionWindow:
    start_step: int
    start_position: np.ndarray
    target: GridCell
    consecutive_events: int = 0


class StuckDetector:
    """Detect commanded motion with insufficient displacement over a step window."""

    def __init__(
        self,
        *,
        window_steps: int = 30,
        min_displacement_normalized: float = 0.01,
        min_action_magnitude: float = 0.1,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be positive")
        if min_displacement_normalized <= 0.0:
            raise ValueError("min_displacement_normalized must be positive")
        self.window_steps = window_steps
        self.min_displacement_normalized = min_displacement_normalized
        self.min_action_magnitude = min_action_magnitude
        self._windows: dict[str, _MotionWindow] = {}

    def reset(self, agent: str | None = None) -> None:
        if agent is None:
            self._windows.clear()
        else:
            self._windows.pop(agent, None)

    def observe(
        self,
        state: UnitState,
        action: np.ndarray,
    ) -> StuckEvent | None:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (2,) or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be a finite float32[2]")
        action_magnitude = float(np.linalg.norm(action_array))
        if state.target is None or action_magnitude < self.min_action_magnitude:
            self.reset(state.agent)
            return None
        position = np.asarray(state.position_normalized, dtype=np.float32)
        window = self._windows.get(state.agent)
        if window is None or window.target != state.target or state.step < window.start_step:
            self._windows[state.agent] = _MotionWindow(
                start_step=state.step,
                start_position=position.copy(),
                target=state.target,
            )
            return None
        if state.step - window.start_step < self.window_steps:
            return None

        displacement = float(np.linalg.norm(position - window.start_position))
        if displacement < self.min_displacement_normalized:
            event_count = window.consecutive_events + 1
            self._windows[state.agent] = _MotionWindow(
                start_step=state.step,
                start_position=position.copy(),
                target=state.target,
                consecutive_events=event_count,
            )
            return StuckEvent(
                agent=state.agent,
                step=state.step,
                target=state.target,
                displacement_normalized=displacement,
                commanded_action_magnitude=action_magnitude,
                consecutive_events=event_count,
            )
        self._windows[state.agent] = _MotionWindow(
            start_step=state.step,
            start_position=position.copy(),
            target=state.target,
            consecutive_events=0,
        )
        return None


@dataclass(frozen=True)
class TaskAssignment:
    agent: str
    slot_id: int
    role: Role
    target: GridCell
    path: tuple[GridCell, ...]
    path_cost: float


def greedy_path_assignment(
    units: Iterable[UnitState],
    targets: Iterable[GridCell],
    planner: AStarPlanner,
    *,
    eligible_roles: set[Role] | None = None,
    reserved_targets: Iterable[GridCell] = (),
) -> tuple[TaskAssignment, ...]:
    """Assign unique reachable targets using deterministic greedy A* cost."""
    selected_units = [
        unit for unit in units if eligible_roles is None or unit.role in eligible_roles
    ]
    selected_units.sort(key=lambda unit: unit.slot_id)
    available_targets = sorted(set(targets) - set(reserved_targets))
    candidates = []
    for unit in selected_units:
        for target in available_targets:
            try:
                path = planner.plan(unit.cell, target)
            except PathNotFound:
                continue
            candidates.append(
                (
                    path_cost(path),
                    unit.slot_id,
                    target.x,
                    target.y,
                    unit,
                    target,
                    path,
                )
            )
    candidates.sort(key=lambda candidate: candidate[:4])

    used_agents: set[str] = set()
    used_targets: set[GridCell] = set()
    assignments = []
    for cost, _, _, _, unit, target, path in candidates:
        if unit.agent in used_agents or target in used_targets:
            continue
        used_agents.add(unit.agent)
        used_targets.add(target)
        assignments.append(
            TaskAssignment(
                agent=unit.agent,
                slot_id=unit.slot_id,
                role=unit.role,
                target=target,
                path=path,
                path_cost=cost,
            )
        )
    return tuple(sorted(assignments, key=lambda assignment: assignment.slot_id))


def assignment_targets(
    assignments: Iterable[TaskAssignment],
) -> Mapping[str, GridCell]:
    return {assignment.agent: assignment.target for assignment in assignments}
