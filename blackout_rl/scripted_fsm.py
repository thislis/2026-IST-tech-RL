"""Role-specific finite-state controllers for the first scripted BlackOut team."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

from .batching import N_TEAM_AGENTS, team_agents
from .coordination import StuckDetector, greedy_path_assignment
from .navigation import AStarPlanner, PathNotFound, WaypointFollower, path_cost
from .observation import parse_vector
from .semantic_map import (
    DecodedSemanticMap,
    GridCell,
    SemanticMapDecoder,
    normalized_to_cell,
)
from .team_state import Role, TeamSnapshot, TeamStateTracker, UnitState
from .strategy import (
    AbsorptionState,
    DangerMap,
    StrategyMode,
    WeightedAStarPlanner,
    absorption_state,
    build_danger_map,
    choose_strategy_mode,
)


COLLECTOR_CLASS_ID = 0
HUNTER_CLASS_ID = 1
CARRIER_CLASS_ID = 2
BATTERY_ITEM_ID = 1
SPECIAL_ITEM_CHANNELS = (
    ("buff_speed", 2),
    ("debuff_speed", 3),
    ("buff_size", 4),
    ("debuff_size", 5),
)

HUNTER_SHRINE_CELLS = (
    GridCell(11, 11),
    GridCell(11, 12),
    GridCell(12, 11),
    GridCell(12, 12),
)
CARRIER_SHRINE_BY_TEAM = {
    0: GridCell(1, 20),
    1: GridCell(20, 1),
}


class WorkerPhase(str, Enum):
    SEEK_BATTERY = "seek_battery"
    PICKUP = "pickup"
    DELIVER = "deliver"
    RETARGET = "retarget"


class GuardPhase(str, Enum):
    SEEK_SHRINE = "seek_shrine"
    TRANSFORM = "transform"
    PATROL = "patrol"
    CHASE = "chase"


class CarrierPhase(str, Enum):
    SEEK_SHRINE = "seek_shrine"
    TRANSFORM = "transform"
    SEEK_FAR_BATTERY = "seek_far_battery"
    PICKUP = "pickup"
    DELIVER = "deliver"
    EVADE = "evade"
    RETARGET = "retarget"


FSMPhase = WorkerPhase | GuardPhase | CarrierPhase


@dataclass(frozen=True)
class FSMDecision:
    agent: str
    role: Role
    phase: str
    target: GridCell | None
    action: np.ndarray


@dataclass
class _Runtime:
    role: Role
    phase: FSMPhase
    follower: WaypointFollower
    target: GridCell | None = None
    temporarily_blocked: set[GridCell] = field(default_factory=set)
    patrol_index: int = 0
    resume_phase: CarrierPhase | None = None
    delivery_wait_steps: int = 0
    failed_storage_cells: set[GridCell] = field(default_factory=set)
    expected_item_id: int = BATTERY_ITEM_ID


def carrier_shrine_cell(team: int) -> GridCell:
    try:
        return CARRIER_SHRINE_BY_TEAM[team]
    except KeyError as exc:
        raise ValueError(f"team must be 0 or 1, got {team}") from exc


def neutral_battery_cells(decoded: DecodedSemanticMap) -> tuple[GridCell, ...]:
    """Exclude batteries already lying in either team's storage or a shrine."""
    excluded = (
        set(decoded.cells("ally_storage"))
        | set(decoded.cells("enemy_storage"))
        | set(HUNTER_SHRINE_CELLS)
        | set(CARRIER_SHRINE_BY_TEAM.values())
    )
    return tuple(sorted(set(decoded.cells("battery")) - excluded))


def item_cells_from_graphic(
    graphic: np.ndarray,
    *,
    channel: int = 6,
    resolution_scale: int = 4,
) -> tuple[GridCell, ...]:
    """Vectorized extraction of one item channel from a top-down HWC map."""
    array = np.asarray(graphic)
    if array.ndim != 3 or not 0 <= channel < array.shape[-1]:
        raise ValueError(f"invalid graphic/channel: shape={array.shape}, channel={channel}")
    height, width = array.shape[:2]
    if height % resolution_scale or width % resolution_scale:
        raise ValueError("graphic size must be divisible by resolution_scale")
    height_cells = height // resolution_scale
    width_cells = width // resolution_scale
    mask = array[..., channel] == 1.0
    active = mask.reshape(
        height_cells,
        resolution_scale,
        width_cells,
        resolution_scale,
    ).any(axis=(1, 3))
    return tuple(
        sorted(
            GridCell(int(column), height_cells - 1 - int(row))
            for row, column in np.argwhere(active)
        )
    )


def battery_cells_from_graphic(
    graphic: np.ndarray,
    *,
    resolution_scale: int = 4,
) -> tuple[GridCell, ...]:
    return item_cells_from_graphic(
        graphic,
        channel=6,
        resolution_scale=resolution_scale,
    )


def nearest_reachable_cell(
    start: GridCell,
    targets: Iterable[GridCell],
    planner: AStarPlanner,
) -> tuple[GridCell, tuple[GridCell, ...], float]:
    candidates = []
    for target in sorted(set(targets)):
        try:
            path = planner.plan(start, target)
        except PathNotFound:
            continue
        cost = planner.path_cost(path) if hasattr(planner, "path_cost") else path_cost(path)
        candidates.append((cost, target.x, target.y, target, path))
    if not candidates:
        raise PathNotFound(f"no reachable target from {start}")
    cost, _, _, target, path = min(candidates)
    return target, path, cost


def farthest_reachable_cell(
    start: GridCell,
    targets: Iterable[GridCell],
    planner: AStarPlanner,
) -> tuple[GridCell, tuple[GridCell, ...], float]:
    candidates = []
    for target in sorted(set(targets)):
        try:
            path = planner.plan(start, target)
        except PathNotFound:
            continue
        cost = planner.path_cost(path) if hasattr(planner, "path_cost") else path_cost(path)
        candidates.append((cost, -target.x, -target.y, target, path))
    if not candidates:
        raise PathNotFound(f"no reachable target from {start}")
    cost, _, _, target, path = max(candidates)
    return target, path, cost


def select_evasion_target(
    start: GridCell,
    enemies: Iterable[GridCell],
    planner: AStarPlanner,
    *,
    search_radius: int = 4,
) -> GridCell:
    """Choose a reachable nearby cell maximizing distance from the closest enemy."""
    enemy_cells = tuple(enemies)
    if not enemy_cells:
        return start
    candidates = []
    for y in range(max(0, start.y - search_radius), min(planner.walkable_grid.shape[0], start.y + search_radius + 1)):
        for x in range(max(0, start.x - search_radius), min(planner.walkable_grid.shape[1], start.x + search_radius + 1)):
            target = GridCell(x, y)
            if not planner.walkable_grid[y, x]:
                continue
            try:
                route = planner.plan(start, target)
            except PathNotFound:
                continue
            cost = path_cost(route)
            if cost > search_radius + 0.5:
                continue
            safety = min(max(abs(x - enemy.x), abs(y - enemy.y)) for enemy in enemy_cells)
            # Maximize safety, then prefer a shorter deterministic escape route.
            candidates.append((safety, -cost, -x, -y, target))
    if not candidates:
        return start
    return max(candidates)[-1]


class ScriptedTeamController:
    """Stateful worker/guard/carrier controller with deterministic navigation.

    The controller owns one team. If ``act`` receives agents from both teams,
    agents outside the owned team receive NoOp actions, which makes the class
    directly usable in smoke runners and evaluator adapters.
    """

    def __init__(
        self,
        team: int,
        *,
        seed: int = 0,
        chase_radius_cells: int = 5,
        evade_radius_cells: int = 3,
        evade_release_radius_cells: int = 5,
        enable_absorption_strategy: bool = True,
        enable_danger_map: bool = True,
        enable_special_items: bool = False,
    ) -> None:
        self.team = team
        self.agents = team_agents(team)
        self.seed = seed
        self.chase_radius_cells = chase_radius_cells
        self.evade_radius_cells = evade_radius_cells
        self.evade_release_radius_cells = evade_release_radius_cells
        self.enable_absorption_strategy = enable_absorption_strategy
        self.enable_danger_map = enable_danger_map
        self.enable_special_items = enable_special_items
        self._decoder = SemanticMapDecoder()
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._static_map: DecodedSemanticMap | None = None
        self._dynamic_map: DecodedSemanticMap | None = None
        self._planner: AStarPlanner | None = None
        self._tracker: TeamStateTracker | None = None
        self._runtimes: dict[str, _Runtime] = {}
        self._patrol_cells: tuple[GridCell, ...] = ()
        self._last_time_left: float | None = None
        self._last_strategy_mode: StrategyMode | None = None
        self.last_absorption_state: AbsorptionState | None = None
        self.last_strategy_mode: StrategyMode | None = None
        self.last_danger_maps: dict[Role, DangerMap] = {}
        self.last_events: list[dict] = []
        self.last_decisions: dict[str, FSMDecision] = {}
        self.last_snapshot: TeamSnapshot | None = None

    @property
    def phases(self) -> dict[str, str]:
        return {agent: runtime.phase.value for agent, runtime in self._runtimes.items()}

    @property
    def targets(self) -> dict[str, GridCell | None]:
        return {agent: runtime.target for agent, runtime in self._runtimes.items()}

    @property
    def paths(self) -> dict[str, tuple[GridCell, ...]]:
        return {
            agent: runtime.follower.path
            for agent, runtime in self._runtimes.items()
        }

    def _initialize(self, observations: Mapping[str, Mapping[str, np.ndarray]]) -> None:
        missing = [agent for agent in self.agents if agent not in observations]
        if missing:
            raise KeyError(f"scripted controller observations missing agents: {missing}")
        decoded = self._decoder.decode(observations[self.agents[0]]["graphic"])
        if not decoded.cells("ally_storage"):
            raise RuntimeError("semantic map has no allied storage cells")
        if not neutral_battery_cells(decoded):
            raise RuntimeError("semantic map has no neutral battery cells")
        self._static_map = decoded
        self._dynamic_map = decoded
        self._planner = AStarPlanner(decoded.walkable_grid)
        self._tracker = TeamStateTracker(
            self.team,
            width_cells=decoded.width_cells,
            height_cells=decoded.height_cells,
        )
        roles = self._tracker.roles
        for agent in self.agents:
            role = roles[agent]
            if role == Role.WORKER:
                phase: FSMPhase = WorkerPhase.SEEK_BATTERY
            elif role == Role.GUARD:
                phase = GuardPhase.SEEK_SHRINE
            else:
                phase = CarrierPhase.SEEK_SHRINE
            self._runtimes[agent] = _Runtime(
                role=role,
                phase=phase,
                follower=WaypointFollower(
                    decoded.width_cells,
                    decoded.height_cells,
                    tolerance=0.004,
                    stall_limit=30,
                ),
            )
        components = decoded.connected_cell_components("ally_storage")
        self._patrol_cells = tuple(
            min(
                component,
                key=lambda cell: (
                    abs(cell.x - decoded.width_cells / 2)
                    + abs(cell.y - decoded.height_cells / 2),
                    cell.x,
                    cell.y,
                ),
            )
            for component in components
        )
        if not self._patrol_cells:
            raise RuntimeError("could not derive guard patrol cells")

    def _planner_avoiding(
        self,
        forbidden: Iterable[GridCell],
        *,
        role: Role | None = None,
    ) -> AStarPlanner | WeightedAStarPlanner:
        assert self._planner is not None
        grid = self._planner.walkable_grid.copy()
        for cell in forbidden:
            if 0 <= cell.x < grid.shape[1] and 0 <= cell.y < grid.shape[0]:
                grid[cell.y, cell.x] = False
        if self.enable_danger_map and role is not None and role in self.last_danger_maps:
            return WeightedAStarPlanner(
                grid,
                self.last_danger_maps[role].traversal_multiplier,
                allow_diagonal=self._planner.allow_diagonal,
            )
        return AStarPlanner(grid, allow_diagonal=self._planner.allow_diagonal)

    def _forbidden_cells(self, runtime: _Runtime, state: UnitState) -> set[GridCell]:
        if runtime.role == Role.WORKER:
            return set(HUNTER_SHRINE_CELLS) | {carrier_shrine_cell(self.team)}
        if runtime.role == Role.GUARD and state.class_id == COLLECTOR_CLASS_ID:
            return {carrier_shrine_cell(self.team)}
        if runtime.role == Role.CARRIER and state.class_id == COLLECTOR_CLASS_ID:
            return set(HUNTER_SHRINE_CELLS)
        return set()

    def _emit(self, agent: str, kind: str, **payload) -> None:
        runtime = self._runtimes[agent]
        self.last_events.append(
            {
                "kind": kind,
                "step": self._step,
                "agent": agent,
                "role": runtime.role.value,
                **payload,
            }
        )

    def _transition(self, agent: str, phase: FSMPhase, reason: str) -> None:
        runtime = self._runtimes[agent]
        if runtime.phase == phase:
            return
        previous = runtime.phase
        runtime.phase = phase
        self._emit(
            agent,
            "fsm_transition",
            reason=reason,
            from_phase=previous.value,
            to_phase=phase.value,
        )

    def _clear_target(self, agent: str) -> None:
        runtime = self._runtimes[agent]
        runtime.target = None
        runtime.temporarily_blocked.clear()
        assert self._tracker is not None
        self._tracker.set_target(agent, None)

    def _set_target(
        self,
        state: UnitState,
        target: GridCell,
        *,
        route: tuple[GridCell, ...] | None = None,
    ) -> None:
        runtime = self._runtimes[state.agent]
        if runtime.target == target and runtime.follower.path:
            return
        if route is None:
            planner = self._planner_avoiding(
                self._forbidden_cells(runtime, state) | runtime.temporarily_blocked,
                role=runtime.role,
            )
            route = planner.plan(state.cell, target)
        runtime.target = target
        runtime.follower.set_path(route)
        runtime.temporarily_blocked.clear()
        assert self._tracker is not None
        self._tracker.set_target(state.agent, target)

    def _observed_transitions(
        self,
        snapshot: TeamSnapshot,
        battery_cells: set[GridCell],
    ) -> None:
        for state in snapshot.units:
            runtime = self._runtimes[state.agent]
            if runtime.role == Role.WORKER:
                if runtime.phase == WorkerPhase.PICKUP and state.is_carrying:
                    is_special = state.holding_item_id != BATTERY_ITEM_ID
                    self._emit(
                        state.agent,
                        "special_item_pickup" if is_special else "battery_pickup",
                        item_id=state.holding_item_id,
                        expected_item_id=runtime.expected_item_id,
                        cell=[state.cell.x, state.cell.y],
                    )
                    self._transition(state.agent, WorkerPhase.DELIVER, "item_acquired")
                    self._clear_target(state.agent)
                elif (
                    runtime.phase == WorkerPhase.PICKUP
                    and runtime.target not in battery_cells
                ):
                    self._transition(state.agent, WorkerPhase.RETARGET, "item_unavailable")
                    runtime.expected_item_id = BATTERY_ITEM_ID
                    self._clear_target(state.agent)
                elif runtime.phase == WorkerPhase.DELIVER and not state.is_carrying:
                    was_special = runtime.expected_item_id != BATTERY_ITEM_ID
                    self._emit(
                        state.agent,
                        "special_item_deposit" if was_special else "battery_deposit",
                        item_id=runtime.expected_item_id,
                        cell=[state.cell.x, state.cell.y],
                    )
                    self._transition(state.agent, WorkerPhase.RETARGET, "item_deposited")
                    runtime.delivery_wait_steps = 0
                    runtime.failed_storage_cells.clear()
                    runtime.expected_item_id = BATTERY_ITEM_ID
                    self._clear_target(state.agent)
                elif runtime.phase == WorkerPhase.DELIVER:
                    self._update_blocked_delivery(state)
                elif runtime.phase == WorkerPhase.RETARGET:
                    self._transition(state.agent, WorkerPhase.SEEK_BATTERY, "retarget_ready")

            elif runtime.role == Role.GUARD:
                if state.class_id == HUNTER_CLASS_ID and runtime.phase in (
                    GuardPhase.SEEK_SHRINE,
                    GuardPhase.TRANSFORM,
                ):
                    self._emit(
                        state.agent,
                        "class_transform",
                        class_id=HUNTER_CLASS_ID,
                        cell=[state.cell.x, state.cell.y],
                    )
                    self._transition(state.agent, GuardPhase.PATROL, "hunter_confirmed")
                    self._clear_target(state.agent)
                elif state.class_id == COLLECTOR_CLASS_ID and runtime.phase in (
                    GuardPhase.PATROL,
                    GuardPhase.CHASE,
                ):
                    self._transition(state.agent, GuardPhase.SEEK_SHRINE, "respawned_collector")
                    self._clear_target(state.agent)

            else:
                if state.class_id == CARRIER_CLASS_ID and runtime.phase in (
                    CarrierPhase.SEEK_SHRINE,
                    CarrierPhase.TRANSFORM,
                ):
                    self._emit(
                        state.agent,
                        "class_transform",
                        class_id=CARRIER_CLASS_ID,
                        cell=[state.cell.x, state.cell.y],
                    )
                    self._transition(
                        state.agent,
                        CarrierPhase.SEEK_FAR_BATTERY,
                        "carrier_confirmed",
                    )
                    self._clear_target(state.agent)
                elif state.class_id == COLLECTOR_CLASS_ID and runtime.phase not in (
                    CarrierPhase.SEEK_SHRINE,
                    CarrierPhase.TRANSFORM,
                ):
                    runtime.resume_phase = None
                    self._transition(
                        state.agent,
                        CarrierPhase.SEEK_SHRINE,
                        "respawned_collector",
                    )
                    self._clear_target(state.agent)
                elif runtime.phase == CarrierPhase.PICKUP and state.is_carrying:
                    self._emit(
                        state.agent,
                        "battery_pickup",
                        item_id=state.holding_item_id,
                        cell=[state.cell.x, state.cell.y],
                    )
                    self._transition(state.agent, CarrierPhase.DELIVER, "item_acquired")
                    self._clear_target(state.agent)
                elif (
                    runtime.phase == CarrierPhase.PICKUP
                    and runtime.target not in battery_cells
                ):
                    self._transition(
                        state.agent,
                        CarrierPhase.RETARGET,
                        "battery_unavailable",
                    )
                    self._clear_target(state.agent)
                elif runtime.phase == CarrierPhase.DELIVER and not state.is_carrying:
                    self._emit(
                        state.agent,
                        "battery_deposit",
                        cell=[state.cell.x, state.cell.y],
                    )
                    self._transition(state.agent, CarrierPhase.RETARGET, "item_deposited")
                    runtime.delivery_wait_steps = 0
                    runtime.failed_storage_cells.clear()
                    self._clear_target(state.agent)
                elif runtime.phase == CarrierPhase.DELIVER:
                    self._update_blocked_delivery(state)
                elif runtime.phase == CarrierPhase.RETARGET:
                    self._transition(
                        state.agent,
                        CarrierPhase.SEEK_FAR_BATTERY,
                        "retarget_ready",
                    )

    def _update_blocked_delivery(self, state: UnitState) -> None:
        """Leave a storage region that did not accept an item and try another one."""
        runtime = self._runtimes[state.agent]
        if runtime.target is None or (
            state.cell != runtime.target and not runtime.follower.complete
        ):
            runtime.delivery_wait_steps = 0
            return
        runtime.delivery_wait_steps += 1
        if runtime.delivery_wait_steps < 12:
            return
        assert self._static_map is not None
        failed_component = next(
            (
                component
                for component in self._static_map.connected_cell_components("ally_storage")
                if runtime.target in component
            ),
            (runtime.target,),
        )
        runtime.failed_storage_cells.update(failed_component)
        self._emit(
            state.agent,
            "storage_retarget",
            failed_target=[runtime.target.x, runtime.target.y],
            reason="deposit_not_observed",
        )
        runtime.delivery_wait_steps = 0
        self._clear_target(state.agent)

    def _assign_workers(
        self,
        snapshot: TeamSnapshot,
        battery_cells: tuple[GridCell, ...],
        *,
        selection: str = "nearest_greedy",
    ) -> None:
        seeking = tuple(
            state
            for state in snapshot.units
            if self._runtimes[state.agent].role == Role.WORKER
            and self._runtimes[state.agent].phase == WorkerPhase.SEEK_BATTERY
        )
        if not seeking:
            return
        reserved = {
            runtime.target
            for runtime in self._runtimes.values()
            if runtime.target is not None
        }
        worker_forbidden = set(HUNTER_SHRINE_CELLS) | {carrier_shrine_cell(self.team)}
        planner = self._planner_avoiding(worker_forbidden, role=Role.WORKER)
        assignments = greedy_path_assignment(
            seeking,
            battery_cells,
            planner,
            eligible_roles={Role.WORKER},
            reserved_targets=reserved,
        )
        for assignment in assignments:
            state = next(state for state in seeking if state.agent == assignment.agent)
            self._set_target(state, assignment.target, route=assignment.path)
            self._emit(
                state.agent,
                "battery_assigned",
                target=[assignment.target.x, assignment.target.y],
                path_cost=assignment.path_cost,
                selection=selection,
            )
            self._transition(state.agent, WorkerPhase.PICKUP, "battery_assigned")

    def _assign_special_item(
        self,
        snapshot: TeamSnapshot,
        special_items: tuple[tuple[str, int, GridCell], ...],
    ) -> None:
        """Give at most one idle worker a currently active special item."""
        if not self.enable_special_items or not special_items:
            return
        seeking = tuple(
            state
            for state in snapshot.units
            if self._runtimes[state.agent].role == Role.WORKER
            and self._runtimes[state.agent].phase == WorkerPhase.SEEK_BATTERY
        )
        if not seeking:
            return
        reserved = {
            runtime.target
            for runtime in self._runtimes.values()
            if runtime.target is not None
        }
        planner = self._planner_avoiding(
            set(HUNTER_SHRINE_CELLS) | {carrier_shrine_cell(self.team)},
            role=Role.WORKER,
        )
        candidates = []
        for state in seeking:
            for name, item_id, target in special_items:
                if target in reserved:
                    continue
                try:
                    route = planner.plan(state.cell, target)
                except PathNotFound:
                    continue
                cost = planner.path_cost(route) if hasattr(planner, "path_cost") else path_cost(route)
                candidates.append(
                    (cost, state.slot_id, item_id, target.x, target.y, state, name, target, route)
                )
        if not candidates:
            return
        cost, _, item_id, _, _, state, name, target, route = min(candidates)
        runtime = self._runtimes[state.agent]
        runtime.expected_item_id = item_id
        self._set_target(state, target, route=route)
        self._emit(
            state.agent,
            "special_item_assigned",
            item=name,
            item_id=item_id,
            target=[target.x, target.y],
            path_cost=cost,
        )
        self._transition(state.agent, WorkerPhase.PICKUP, "special_item_assigned")

    def _assign_delivery(self, state: UnitState) -> None:
        runtime = self._runtimes[state.agent]
        if runtime.target is not None:
            return
        assert self._static_map is not None
        planner = self._planner_avoiding(
            self._forbidden_cells(runtime, state),
            role=runtime.role,
        )
        candidates = set(self._static_map.cells("ally_storage")) - runtime.failed_storage_cells
        if not candidates:
            runtime.failed_storage_cells.clear()
            candidates = set(self._static_map.cells("ally_storage"))
        target, route, cost = nearest_reachable_cell(state.cell, candidates, planner)
        self._set_target(state, target, route=route)
        self._emit(
            state.agent,
            "storage_assigned",
            target=[target.x, target.y],
            path_cost=cost,
        )

    def _enemy_cells(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        width: int,
        height: int,
    ) -> tuple[GridCell, ...]:
        parsed = parse_vector(observations[self.agents[0]]["vector"])
        start = (1 - self.team) * N_TEAM_AGENTS
        return tuple(
            normalized_to_cell(
                parsed.positions[index],
                width_cells=width,
                height_cells=height,
            )
            for index in range(start, start + N_TEAM_AGENTS)
        )

    def _update_strategy(
        self,
        snapshot: TeamSnapshot,
        enemies: tuple[GridCell, ...],
        enemy_storage_batteries: int,
    ) -> None:
        assert self._planner is not None
        if self.enable_absorption_strategy:
            clock = absorption_state(snapshot.time_left)
            mode = choose_strategy_mode(
                clock,
                own_score=snapshot.own_score,
                opponent_score=snapshot.opponent_score,
                enemy_storage_batteries=enemy_storage_batteries,
            )
        else:
            clock = absorption_state(snapshot.time_left)
            mode = StrategyMode.COLLECTION
        self.last_absorption_state = clock
        self.last_strategy_mode = mode
        self.last_danger_maps = {
            role: build_danger_map(self._planner.walkable_grid, enemies, role)
            for role in Role
        } if self.enable_danger_map else {}
        if mode != self._last_strategy_mode:
            for agent in self.agents:
                self._emit(
                    agent,
                    "strategy_mode",
                    mode=mode.value,
                    absorption_phase=clock.phase.value,
                    seconds_until_absorption=clock.seconds_until_absorption,
                    cycle_index=clock.cycle_index,
                )
            self._last_strategy_mode = mode

    def _apply_secure_mode(self, snapshot: TeamSnapshot) -> None:
        """Late-cycle safety: carrying collectors immediately target allied storage."""
        if self.last_strategy_mode != StrategyMode.SECURE:
            return
        for state in snapshot.units:
            runtime = self._runtimes[state.agent]
            if runtime.role == Role.GUARD or not state.is_carrying:
                continue
            delivery_phase: FSMPhase = (
                WorkerPhase.DELIVER
                if runtime.role == Role.WORKER
                else CarrierPhase.DELIVER
            )
            if runtime.phase != delivery_phase:
                self._transition(state.agent, delivery_phase, "secure_before_absorption")
                self._clear_target(state.agent)
                self._emit(
                    state.agent,
                    "secure_delivery",
                    seconds_until_absorption=(
                        self.last_absorption_state.seconds_until_absorption
                        if self.last_absorption_state is not None
                        else None
                    ),
                )

    @staticmethod
    def _chebyshev(first: GridCell, second: GridCell) -> int:
        return max(abs(first.x - second.x), abs(first.y - second.y))

    def _update_guard(self, state: UnitState, enemies: tuple[GridCell, ...]) -> None:
        runtime = self._runtimes[state.agent]
        if runtime.phase == GuardPhase.SEEK_SHRINE:
            planner = self._planner_avoiding(
                {carrier_shrine_cell(self.team)},
                role=Role.GUARD,
            )
            target, route, cost = nearest_reachable_cell(
                state.cell,
                HUNTER_SHRINE_CELLS,
                planner,
            )
            self._set_target(state, target, route=route)
            self._emit(
                state.agent,
                "shrine_assigned",
                shrine="hunter",
                target=[target.x, target.y],
                path_cost=cost,
            )
            self._transition(state.agent, GuardPhase.TRANSFORM, "hunter_shrine_assigned")
            return
        if runtime.phase == GuardPhase.TRANSFORM:
            return

        closest_enemy = min(
            enemies,
            key=lambda enemy: (self._chebyshev(state.cell, enemy), enemy.x, enemy.y),
        )
        enemy_distance = self._chebyshev(state.cell, closest_enemy)
        if enemy_distance <= self.chase_radius_cells:
            if runtime.phase != GuardPhase.CHASE:
                self._transition(state.agent, GuardPhase.CHASE, "enemy_in_range")
                self._emit(
                    state.agent,
                    "enemy_chase",
                    enemy_cell=[closest_enemy.x, closest_enemy.y],
                    distance_cells=enemy_distance,
                )
            try:
                self._set_target(state, closest_enemy)
            except PathNotFound:
                self._transition(state.agent, GuardPhase.PATROL, "enemy_unreachable")
                self._clear_target(state.agent)
            return

        if runtime.phase == GuardPhase.CHASE:
            self._transition(state.agent, GuardPhase.PATROL, "enemy_out_of_range")
            self._clear_target(state.agent)
        if runtime.phase != GuardPhase.PATROL:
            return
        if runtime.target is not None and (
            state.cell == runtime.target or runtime.follower.complete
        ):
            self._emit(
                state.agent,
                "patrol_waypoint_reached",
                cell=[state.cell.x, state.cell.y],
            )
            runtime.patrol_index = (runtime.patrol_index + 1) % len(self._patrol_cells)
            self._clear_target(state.agent)
        if runtime.target is None:
            target = self._patrol_cells[runtime.patrol_index]
            self._set_target(state, target)
            self._emit(
                state.agent,
                "patrol_assigned",
                target=[target.x, target.y],
                patrol_index=runtime.patrol_index,
            )

    def _update_carrier(
        self,
        state: UnitState,
        enemies: tuple[GridCell, ...],
        batteries: tuple[GridCell, ...],
    ) -> None:
        runtime = self._runtimes[state.agent]
        if runtime.phase == CarrierPhase.SEEK_SHRINE:
            target = carrier_shrine_cell(self.team)
            planner = self._planner_avoiding(
                HUNTER_SHRINE_CELLS,
                role=Role.CARRIER,
            )
            route = planner.plan(state.cell, target)
            self._set_target(state, target, route=route)
            self._emit(
                state.agent,
                "shrine_assigned",
                shrine="carrier",
                target=[target.x, target.y],
                path_cost=path_cost(route),
            )
            self._transition(state.agent, CarrierPhase.TRANSFORM, "carrier_shrine_assigned")
            return
        if runtime.phase == CarrierPhase.TRANSFORM:
            return

        enemy_distance = min(self._chebyshev(state.cell, enemy) for enemy in enemies)
        if runtime.phase != CarrierPhase.EVADE and enemy_distance <= self.evade_radius_cells:
            runtime.resume_phase = runtime.phase
            self._transition(state.agent, CarrierPhase.EVADE, "enemy_too_close")
            self._clear_target(state.agent)
            self._emit(state.agent, "evade_started", distance_cells=enemy_distance)
        elif runtime.phase == CarrierPhase.EVADE and enemy_distance > self.evade_release_radius_cells:
            resumed = runtime.resume_phase or CarrierPhase.SEEK_FAR_BATTERY
            runtime.resume_phase = None
            self._transition(state.agent, resumed, "enemy_distance_restored")
            self._clear_target(state.agent)
            self._emit(state.agent, "evade_finished", distance_cells=enemy_distance)

        if runtime.phase == CarrierPhase.EVADE:
            assert self._planner is not None
            target = select_evasion_target(state.cell, enemies, self._planner)
            if target != runtime.target:
                self._set_target(state, target)
                self._emit(
                    state.agent,
                    "evade_target",
                    target=[target.x, target.y],
                    distance_cells=enemy_distance,
                )
            return

        if runtime.phase == CarrierPhase.SEEK_FAR_BATTERY:
            reserved = {
                other.target
                for other in self._runtimes.values()
                if other is not runtime and other.target is not None
            }
            available = set(batteries) - reserved
            if not available:
                return
            assert self._planner is not None
            planner = self._planner_avoiding((), role=Role.CARRIER)
            target, route, cost = farthest_reachable_cell(
                state.cell,
                available,
                planner,
            )
            self._set_target(state, target, route=route)
            self._emit(
                state.agent,
                "battery_assigned",
                target=[target.x, target.y],
                path_cost=cost,
                selection="farthest_reachable",
            )
            self._transition(state.agent, CarrierPhase.PICKUP, "far_battery_assigned")

    def _navigate(self, state: UnitState) -> np.ndarray:
        runtime = self._runtimes[state.agent]
        if runtime.target is None:
            return np.zeros(2, dtype=np.float32)
        action = runtime.follower.action(state.position_normalized)
        detector = self._stuck_detector
        event = detector.observe(state, action)
        if event is None:
            return action
        waypoint = runtime.follower.current_waypoint
        if waypoint is not None and waypoint not in (state.cell, runtime.target):
            runtime.temporarily_blocked.add(waypoint)
        planner = self._planner_avoiding(
            self._forbidden_cells(runtime, state) | runtime.temporarily_blocked,
            role=runtime.role,
        )
        try:
            route = planner.plan(state.cell, runtime.target)
        except PathNotFound:
            self._emit(
                state.agent,
                "replan_failed",
                target=[runtime.target.x, runtime.target.y],
            )
            self._clear_target(state.agent)
            return np.zeros(2, dtype=np.float32)
        runtime.follower.set_path(route)
        self._emit(
            state.agent,
            "stuck_replan",
            target=[runtime.target.x, runtime.target.y],
            path_cells=len(route),
            blocked=[[cell.x, cell.y] for cell in sorted(runtime.temporarily_blocked)],
        )
        return runtime.follower.action(state.position_normalized)

    def act(
        self,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        agents: Sequence[str],
    ) -> dict[str, np.ndarray]:
        requested = tuple(agents)
        missing = [agent for agent in requested if agent not in observations]
        if missing:
            raise KeyError(f"scripted controller observations missing agents: {missing}")
        if self._tracker is None:
            self._initialize(observations)
            self._stuck_detector = StuckDetector(
                window_steps=30,
                min_displacement_normalized=0.01,
            )
        assert self._tracker is not None
        assert self._static_map is not None

        snapshot = self._tracker.update(observations, step=self._step)
        if (
            self._last_time_left is not None
            and snapshot.time_left > self._last_time_left + 0.05
        ):
            self.reset()
            self._initialize(observations)
            self._stuck_detector = StuckDetector(
                window_steps=30,
                min_displacement_normalized=0.01,
            )
            assert self._tracker is not None
            snapshot = self._tracker.update(observations, step=0)
        self._last_time_left = snapshot.time_left
        dynamic_batteries = battery_cells_from_graphic(
            observations[self.agents[0]]["graphic"],
            resolution_scale=self._static_map.resolution_scale,
        )
        dynamic_special_items = tuple(
            (name, item_id, cell)
            for channel, (name, item_id) in enumerate(SPECIAL_ITEM_CHANNELS, start=7)
            for cell in item_cells_from_graphic(
                observations[self.agents[0]]["graphic"],
                channel=channel,
                resolution_scale=self._static_map.resolution_scale,
            )
        )
        enemy_storage_battery_cells = tuple(
            sorted(set(dynamic_batteries) & set(self._static_map.cells("enemy_storage")))
        )
        excluded = (
            set(self._static_map.cells("ally_storage"))
            | set(self._static_map.cells("enemy_storage"))
            | set(HUNTER_SHRINE_CELLS)
            | set(CARRIER_SHRINE_BY_TEAM.values())
        )
        batteries = tuple(sorted(set(dynamic_batteries) - excluded))
        # A target in enemy storage remains a valid Battery target while RAID is
        # active; only assignment eligibility is strategy-gated below.
        battery_set = (
            set(batteries)
            | set(enemy_storage_battery_cells)
            | {cell for _, _, cell in dynamic_special_items}
        )
        self.last_events = []

        self._observed_transitions(snapshot, battery_set)
        enemies = self._enemy_cells(
            observations,
            self._static_map.width_cells,
            self._static_map.height_cells,
        )
        self._update_strategy(snapshot, enemies, len(enemy_storage_battery_cells))
        self._apply_secure_mode(snapshot)
        assignment_batteries = batteries
        assignment_selection = "nearest_greedy"
        if self.last_strategy_mode == StrategyMode.RAID:
            assignment_batteries = tuple(
                sorted(set(batteries) | set(enemy_storage_battery_cells))
            )
            battery_set.update(enemy_storage_battery_cells)
            assignment_selection = "raid_or_neutral_greedy"
        # During the final four seconds, do not start another collection trip;
        # carriers already holding an item still follow their delivery route.
        if self.last_strategy_mode != StrategyMode.SECURE:
            self._assign_special_item(snapshot, dynamic_special_items)
            self._assign_workers(
                snapshot,
                assignment_batteries,
                selection=assignment_selection,
            )

        state_by_agent = snapshot.by_agent()
        for agent in self.agents:
            state = state_by_agent[agent]
            runtime = self._runtimes[agent]
            if runtime.role == Role.WORKER and runtime.phase == WorkerPhase.DELIVER:
                self._assign_delivery(state)
            elif runtime.role == Role.GUARD:
                self._update_guard(state, enemies)
            elif runtime.role == Role.CARRIER:
                self._update_carrier(state, enemies, batteries)
                if runtime.phase == CarrierPhase.DELIVER:
                    self._assign_delivery(state)

        actions = {
            agent: np.zeros(2, dtype=np.float32)
            for agent in requested
        }
        self.last_decisions = {}
        for agent in self.agents:
            if agent not in actions:
                continue
            state = state_by_agent[agent]
            runtime = self._runtimes[agent]
            action = self._navigate(state)
            actions[agent] = action
            self.last_decisions[agent] = FSMDecision(
                agent=agent,
                role=runtime.role,
                phase=runtime.phase.value,
                target=runtime.target,
                action=action.copy(),
            )

        self.last_snapshot = self._tracker.latest
        self._step += 1
        return actions
