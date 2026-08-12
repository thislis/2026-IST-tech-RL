"""Absorption-cycle strategy and role-aware danger-cost path planning."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from itertools import count
from typing import Iterable

import numpy as np

from .navigation import CARDINAL_COST, DIAGONAL_COST, NEIGHBORS, PathNotFound
from .semantic_map import GridCell
from .team_state import Role


DEFAULT_EPISODE_SECONDS = 420.0
DEFAULT_ABSORPTION_INTERVAL_SECONDS = 20.0


class AbsorptionPhase(str, Enum):
    POST_ABSORPTION = "post_absorption"
    COLLECT = "collect"
    SECURE = "secure"


class StrategyMode(str, Enum):
    FRESH_COLLECTION = "fresh_collection"
    COLLECTION = "collection"
    SECURE = "secure"
    RAID = "raid"


@dataclass(frozen=True)
class AbsorptionState:
    elapsed_seconds: float
    cycle_index: int
    seconds_since_absorption: float
    seconds_until_absorption: float
    phase: AbsorptionPhase


def absorption_state(
    time_left_normalized: float,
    *,
    episode_seconds: float = DEFAULT_EPISODE_SECONDS,
    interval_seconds: float = DEFAULT_ABSORPTION_INTERVAL_SECONDS,
    secure_window_seconds: float = 4.0,
    post_window_seconds: float = 3.0,
) -> AbsorptionState:
    """Recover the looping absorption clock from the normalized episode timer."""
    if not math.isfinite(time_left_normalized) or not 0.0 <= time_left_normalized <= 1.0:
        raise ValueError("time_left_normalized must be finite and in [0,1]")
    if episode_seconds <= 0.0 or interval_seconds <= 0.0:
        raise ValueError("episode and interval seconds must be positive")
    if not 0.0 <= secure_window_seconds < interval_seconds:
        raise ValueError("secure window must be in [0, interval)")
    if not 0.0 <= post_window_seconds < interval_seconds:
        raise ValueError("post window must be in [0, interval)")

    elapsed = min(max((1.0 - time_left_normalized) * episode_seconds, 0.0), episode_seconds)
    nearest_boundary = round(elapsed)
    if abs(elapsed - nearest_boundary) < 1e-7:
        elapsed = float(nearest_boundary)
    cycle_index = int(math.floor((elapsed + 1e-9) / interval_seconds))
    since = elapsed - cycle_index * interval_seconds
    if since < 0.0:
        since = 0.0
    if since >= interval_seconds - 1e-7:
        cycle_index += 1
        since = 0.0
    until = interval_seconds - since
    if since < post_window_seconds:
        phase = AbsorptionPhase.POST_ABSORPTION
    elif until <= secure_window_seconds:
        phase = AbsorptionPhase.SECURE
    else:
        phase = AbsorptionPhase.COLLECT
    return AbsorptionState(
        elapsed_seconds=elapsed,
        cycle_index=cycle_index,
        seconds_since_absorption=since,
        seconds_until_absorption=until,
        phase=phase,
    )


def choose_strategy_mode(
    clock: AbsorptionState,
    *,
    own_score: float,
    opponent_score: float,
    enemy_storage_batteries: int,
    raid_score_deficit: float = 0.10,
    minimum_raid_seconds: float = 8.0,
) -> StrategyMode:
    """Choose a conservative team mode; raids are gated by score and time."""
    if clock.phase == AbsorptionPhase.SECURE:
        return StrategyMode.SECURE
    if (
        enemy_storage_batteries > 0
        and opponent_score - own_score >= raid_score_deficit
        and clock.seconds_until_absorption >= minimum_raid_seconds
    ):
        return StrategyMode.RAID
    if clock.phase == AbsorptionPhase.POST_ABSORPTION:
        return StrategyMode.FRESH_COLLECTION
    return StrategyMode.COLLECTION


@dataclass(frozen=True)
class DangerMap:
    role: Role
    enemy_cells: tuple[GridCell, ...]
    traversal_multiplier: np.ndarray

    def at(self, cell: GridCell) -> float:
        return float(self.traversal_multiplier[cell.y, cell.x])


def build_danger_map(
    walkable_grid: np.ndarray,
    enemy_cells: Iterable[GridCell],
    role: Role,
    *,
    worker_radius: int = 4,
    worker_peak: float = 7.0,
    carrier_radius: int = 5,
    carrier_peak: float = 11.0,
    guard_attraction_radius: int = 5,
    guard_minimum_multiplier: float = 0.40,
) -> DangerMap:
    """Build traversal multipliers: collectors avoid enemies, guards approach."""
    grid = np.asarray(walkable_grid)
    if grid.ndim != 2 or grid.dtype != np.bool_:
        raise TypeError("walkable_grid must be a 2-D bool array")
    enemies = tuple(sorted(set(enemy_cells)))
    multipliers = np.ones(grid.shape, dtype=np.float64)
    height, width = grid.shape
    for enemy in enemies:
        if not (0 <= enemy.x < width and 0 <= enemy.y < height):
            continue
        if role == Role.GUARD:
            radius = guard_attraction_radius
            for y in range(max(0, enemy.y - radius), min(height, enemy.y + radius + 1)):
                for x in range(max(0, enemy.x - radius), min(width, enemy.x + radius + 1)):
                    distance = max(abs(x - enemy.x), abs(y - enemy.y))
                    if distance > radius:
                        continue
                    strength = 1.0 - distance / (radius + 1.0)
                    candidate = 1.0 - (1.0 - guard_minimum_multiplier) * strength
                    multipliers[y, x] = min(multipliers[y, x], candidate)
        else:
            radius = carrier_radius if role == Role.CARRIER else worker_radius
            peak = carrier_peak if role == Role.CARRIER else worker_peak
            for y in range(max(0, enemy.y - radius), min(height, enemy.y + radius + 1)):
                for x in range(max(0, enemy.x - radius), min(width, enemy.x + radius + 1)):
                    distance = max(abs(x - enemy.x), abs(y - enemy.y))
                    if distance > radius:
                        continue
                    strength = 1.0 - distance / (radius + 1.0)
                    multipliers[y, x] += peak * strength
    multipliers[~grid] = np.inf
    return DangerMap(role=role, enemy_cells=enemies, traversal_multiplier=multipliers)


def weighted_path_cost(path: Iterable[GridCell], multipliers: np.ndarray) -> float:
    cells = tuple(path)
    total = 0.0
    for first, second in zip(cells, cells[1:]):
        dx = abs(first.x - second.x)
        dy = abs(first.y - second.y)
        if max(dx, dy) != 1 or dx + dy == 0:
            raise ValueError(f"non-adjacent path edge: {first} -> {second}")
        movement = DIAGONAL_COST if dx and dy else CARDINAL_COST
        total += movement * float(multipliers[second.y, second.x])
    return total


class WeightedAStarPlanner:
    """A* minimizing geometric distance multiplied by role-aware traversal cost."""

    def __init__(
        self,
        walkable_grid: np.ndarray,
        traversal_multiplier: np.ndarray,
        *,
        allow_diagonal: bool = True,
    ) -> None:
        grid = np.asarray(walkable_grid)
        costs = np.asarray(traversal_multiplier, dtype=np.float64)
        if grid.ndim != 2 or grid.dtype != np.bool_:
            raise TypeError("walkable_grid must be a 2-D bool array")
        if costs.shape != grid.shape:
            raise ValueError("traversal multiplier shape must match walkable grid")
        if np.any(costs[grid] <= 0.0) or not np.all(np.isfinite(costs[grid])):
            raise ValueError("walkable traversal multipliers must be finite and positive")
        self.walkable_grid = grid.copy()
        self.traversal_multiplier = costs.copy()
        self.allow_diagonal = allow_diagonal
        self._minimum_multiplier = float(np.min(costs[grid]))

    def _inside(self, cell: GridCell) -> bool:
        return 0 <= cell.x < self.walkable_grid.shape[1] and 0 <= cell.y < self.walkable_grid.shape[0]

    @staticmethod
    def _octile(first: GridCell, second: GridCell) -> float:
        dx = abs(first.x - second.x)
        dy = abs(first.y - second.y)
        return max(dx, dy) + (DIAGONAL_COST - 1.0) * min(dx, dy)

    def _neighbors(self, cell: GridCell):
        for dx, dy, movement in NEIGHBORS:
            if not self.allow_diagonal and dx and dy:
                continue
            neighbor = GridCell(cell.x + dx, cell.y + dy)
            if not self._inside(neighbor) or not self.walkable_grid[neighbor.y, neighbor.x]:
                continue
            if dx and dy and (
                not self.walkable_grid[cell.y, cell.x + dx]
                or not self.walkable_grid[cell.y + dy, cell.x]
            ):
                continue
            yield neighbor, movement * self.traversal_multiplier[neighbor.y, neighbor.x]

    def plan(self, start: GridCell, goal: GridCell) -> tuple[GridCell, ...]:
        for label, cell in (("start", start), ("goal", goal)):
            if not self._inside(cell):
                raise ValueError(f"{label} outside map: {cell}")
            if not self.walkable_grid[cell.y, cell.x]:
                raise PathNotFound(f"{label} is blocked: {cell}")
        if start == goal:
            return (start,)
        sequence = count()
        frontier: list[tuple[float, float, int, GridCell]] = []
        heuristic = self._octile(start, goal) * self._minimum_multiplier
        heapq.heappush(frontier, (heuristic, 0.0, next(sequence), start))
        best = {start: 0.0}
        previous: dict[GridCell, GridCell] = {}
        while frontier:
            _, cost, _, current = heapq.heappop(frontier)
            if cost > best.get(current, math.inf):
                continue
            if current == goal:
                reversed_path = [goal]
                while reversed_path[-1] != start:
                    reversed_path.append(previous[reversed_path[-1]])
                return tuple(reversed(reversed_path))
            for neighbor, edge_cost in self._neighbors(current):
                candidate = cost + float(edge_cost)
                if candidate + 1e-12 >= best.get(neighbor, math.inf):
                    continue
                best[neighbor] = candidate
                previous[neighbor] = current
                priority = candidate + self._octile(neighbor, goal) * self._minimum_multiplier
                heapq.heappush(frontier, (priority, candidate, next(sequence), neighbor))
        raise PathNotFound(f"no path from {start} to {goal}")

    def path_cost(self, path: Iterable[GridCell]) -> float:
        return weighted_path_cost(path, self.traversal_multiplier)

    def plan_avoiding(
        self,
        start: GridCell,
        goal: GridCell,
        temporarily_blocked: Iterable[GridCell],
    ) -> tuple[GridCell, ...]:
        updated = self.walkable_grid.copy()
        for cell in temporarily_blocked:
            if cell in (start, goal):
                continue
            if self._inside(cell):
                updated[cell.y, cell.x] = False
        return WeightedAStarPlanner(
            updated,
            self.traversal_multiplier,
            allow_diagonal=self.allow_diagonal,
        ).plan(start, goal)
