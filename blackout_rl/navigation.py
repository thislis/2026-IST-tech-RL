"""Deterministic grid planning and continuous waypoint following."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count
from typing import Iterable

import numpy as np

from .semantic_map import GridCell, cell_center_normalized, normalized_to_cell


CARDINAL_COST = 1.0
DIAGONAL_COST = math.sqrt(2.0)
NEIGHBORS = (
    (1, 0, CARDINAL_COST),
    (0, 1, CARDINAL_COST),
    (-1, 0, CARDINAL_COST),
    (0, -1, CARDINAL_COST),
    (1, 1, DIAGONAL_COST),
    (-1, 1, DIAGONAL_COST),
    (-1, -1, DIAGONAL_COST),
    (1, -1, DIAGONAL_COST),
)


class PathNotFound(RuntimeError):
    pass


def _inside(grid: np.ndarray, cell: GridCell) -> bool:
    return 0 <= cell.x < grid.shape[1] and 0 <= cell.y < grid.shape[0]


def _octile(a: GridCell, b: GridCell) -> float:
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    return max(dx, dy) + (DIAGONAL_COST - 1.0) * min(dx, dy)


def path_cost(path: Iterable[GridCell]) -> float:
    cells = tuple(path)
    total = 0.0
    for first, second in zip(cells, cells[1:]):
        dx = abs(first.x - second.x)
        dy = abs(first.y - second.y)
        if max(dx, dy) != 1 or dx + dy == 0:
            raise ValueError(f"non-adjacent path edge: {first} -> {second}")
        total += DIAGONAL_COST if dx and dy else CARDINAL_COST
    return total


class AStarPlanner:
    """A* on a bottom-left-origin boolean grid with safe diagonal movement."""

    def __init__(self, walkable_grid: np.ndarray, *, allow_diagonal: bool = True):
        grid = np.asarray(walkable_grid)
        if grid.ndim != 2 or grid.dtype != np.bool_:
            raise TypeError("walkable_grid must be a 2-D bool numpy array indexed [y,x]")
        self.walkable_grid = grid.copy()
        self.allow_diagonal = allow_diagonal

    def _neighbors(self, cell: GridCell):
        for dx, dy, move_cost in NEIGHBORS:
            if not self.allow_diagonal and dx and dy:
                continue
            neighbor = GridCell(cell.x + dx, cell.y + dy)
            if not _inside(self.walkable_grid, neighbor):
                continue
            if not self.walkable_grid[neighbor.y, neighbor.x]:
                continue
            if dx and dy:
                # Never cut through the corner of either blocked cardinal cell.
                if not self.walkable_grid[cell.y, cell.x + dx]:
                    continue
                if not self.walkable_grid[cell.y + dy, cell.x]:
                    continue
            yield neighbor, move_cost

    def plan(self, start: GridCell, goal: GridCell) -> tuple[GridCell, ...]:
        for label, cell in (("start", start), ("goal", goal)):
            if not _inside(self.walkable_grid, cell):
                raise ValueError(f"{label} outside map: {cell}")
            if not self.walkable_grid[cell.y, cell.x]:
                raise PathNotFound(f"{label} is blocked: {cell}")
        if start == goal:
            return (start,)

        tie_breaker = count()
        frontier: list[tuple[float, float, int, GridCell]] = []
        heapq.heappush(frontier, (_octile(start, goal), 0.0, next(tie_breaker), start))
        came_from: dict[GridCell, GridCell] = {}
        best_cost = {start: 0.0}
        while frontier:
            _, current_cost, _, current = heapq.heappop(frontier)
            if current_cost > best_cost.get(current, math.inf):
                continue
            if current == goal:
                reversed_path = [goal]
                while reversed_path[-1] != start:
                    reversed_path.append(came_from[reversed_path[-1]])
                return tuple(reversed(reversed_path))
            for neighbor, move_cost in self._neighbors(current):
                candidate = current_cost + move_cost
                if candidate + 1e-12 >= best_cost.get(neighbor, math.inf):
                    continue
                best_cost[neighbor] = candidate
                came_from[neighbor] = current
                priority = candidate + _octile(neighbor, goal)
                heapq.heappush(
                    frontier,
                    (priority, candidate, next(tie_breaker), neighbor),
                )
        raise PathNotFound(f"no path from {start} to {goal}")

    def plan_avoiding(
        self,
        start: GridCell,
        goal: GridCell,
        temporarily_blocked: Iterable[GridCell],
    ) -> tuple[GridCell, ...]:
        """Replan while treating collision cells as temporary obstacles."""
        updated = self.walkable_grid.copy()
        for cell in temporarily_blocked:
            if cell in (start, goal):
                continue
            if _inside(updated, cell):
                updated[cell.y, cell.x] = False
        return AStarPlanner(updated, allow_diagonal=self.allow_diagonal).plan(start, goal)


def grid_line(start: GridCell, goal: GridCell) -> tuple[GridCell, ...]:
    """Integer Bresenham cells used to audit whether a direct route crosses walls."""
    x0, y0 = start.x, start.y
    x1, y1 = goal.x, goal.y
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append(GridCell(x0, y0))
        if x0 == x1 and y0 == y1:
            return tuple(cells)
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


@dataclass
class WaypointFollower:
    """Turn a cell path into bounded actions and expose a simple replan signal."""

    width_cells: int
    height_cells: int
    tolerance: float = 0.012
    stall_limit: int = 25
    progress_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        self.path: tuple[GridCell, ...] = ()
        self.waypoint_index = 0
        self.stalled_steps = 0
        self._last_position: np.ndarray | None = None

    def set_path(self, path: Iterable[GridCell]) -> None:
        cells = tuple(path)
        if not cells:
            raise ValueError("path cannot be empty")
        self.path = cells
        self.waypoint_index = 1 if len(cells) > 1 else 0
        self.stalled_steps = 0
        self._last_position = None

    @property
    def complete(self) -> bool:
        return bool(self.path) and self.waypoint_index >= len(self.path)

    @property
    def replan_required(self) -> bool:
        return self.stalled_steps >= self.stall_limit

    @property
    def current_waypoint(self) -> GridCell | None:
        if not self.path or self.complete:
            return None
        return self.path[self.waypoint_index]

    def action(self, normalized_xy: tuple[float, float] | np.ndarray) -> np.ndarray:
        position = np.asarray(normalized_xy, dtype=np.float32)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite normalized 2-vector")
        if not self.path or self.complete:
            return np.zeros(2, dtype=np.float32)

        if self._last_position is not None:
            if float(np.linalg.norm(position - self._last_position)) <= self.progress_epsilon:
                self.stalled_steps += 1
            else:
                self.stalled_steps = 0
        self._last_position = position.copy()

        while self.waypoint_index < len(self.path):
            target = cell_center_normalized(
                self.path[self.waypoint_index],
                width_cells=self.width_cells,
                height_cells=self.height_cells,
            )
            delta = target - position
            distance = float(np.linalg.norm(delta))
            if distance > self.tolerance:
                return (delta / distance).astype(np.float32)
            self.waypoint_index += 1
        return np.zeros(2, dtype=np.float32)

    def current_cell(self, normalized_xy: tuple[float, float] | np.ndarray) -> GridCell:
        return normalized_to_cell(
            normalized_xy,
            width_cells=self.width_cells,
            height_cells=self.height_cells,
        )
