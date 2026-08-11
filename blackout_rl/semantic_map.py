"""Semantic-map decoding and explicit image/world coordinate conversion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .observation import GRAPHIC_CHANNEL_NAMES, parse_graphic


DEFAULT_RESOLUTION_SCALE = 4


@dataclass(frozen=True, order=True)
class GridCell:
    """Bottom-left-origin Unity map cell."""

    x: int
    y: int


@dataclass(frozen=True)
class SemanticPoint:
    """One active semantic pixel represented in every relevant coordinate frame."""

    row: int
    column: int
    normalized_xy: tuple[float, float]
    cell: GridCell


@dataclass(frozen=True)
class DecodedSemanticMap:
    """Decoded view whose grid arrays use `[y, x]` with y increasing upward."""

    height_pixels: int
    width_pixels: int
    resolution_scale: int
    channel_points: dict[str, tuple[SemanticPoint, ...]]
    channel_cells: dict[str, tuple[GridCell, ...]]
    wall_grid: np.ndarray

    @property
    def width_cells(self) -> int:
        return self.width_pixels // self.resolution_scale

    @property
    def height_cells(self) -> int:
        return self.height_pixels // self.resolution_scale

    @property
    def walkable_grid(self) -> np.ndarray:
        return np.logical_not(self.wall_grid)

    def points(self, channel: str) -> tuple[SemanticPoint, ...]:
        try:
            return self.channel_points[channel]
        except KeyError as exc:
            raise KeyError(f"unknown semantic channel: {channel}") from exc

    def cells(self, channel: str) -> tuple[GridCell, ...]:
        try:
            return self.channel_cells[channel]
        except KeyError as exc:
            raise KeyError(f"unknown semantic channel: {channel}") from exc

    def connected_cell_components(
        self,
        channel: str,
        *,
        diagonal: bool = False,
    ) -> tuple[tuple[GridCell, ...], ...]:
        remaining = set(self.cells(channel))
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if diagonal:
            offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        components: list[tuple[GridCell, ...]] = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            queue = deque([start])
            component = [start]
            while queue:
                current = queue.popleft()
                for dx, dy in offsets:
                    neighbor = GridCell(current.x + dx, current.y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            components.append(tuple(sorted(component)))
        return tuple(components)


def normalized_to_pixel(
    normalized_xy: tuple[float, float] | np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert bottom-left normalized `(x,y)` to top-down numpy `(row,column)`."""
    xy = np.asarray(normalized_xy, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("normalized_xy must be a finite 2-vector")
    if np.any(xy < 0.0) or np.any(xy > 1.0):
        raise ValueError("normalized coordinates must be in [0,1]")
    column = min(int(np.floor(xy[0] * width)), width - 1)
    pixel_y = min(int(np.floor(xy[1] * height)), height - 1)
    return height - 1 - pixel_y, column


def pixel_to_normalized(
    row: int,
    column: int,
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Return the bottom-left normalized center of a top-down numpy pixel."""
    if not (0 <= row < height and 0 <= column < width):
        raise ValueError(f"pixel outside {width}x{height}: row={row}, column={column}")
    pixel_y = height - 1 - row
    return ((column + 0.5) / width, (pixel_y + 0.5) / height)


def normalized_to_cell(
    normalized_xy: tuple[float, float] | np.ndarray,
    *,
    width_cells: int,
    height_cells: int,
) -> GridCell:
    xy = np.asarray(normalized_xy, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("normalized_xy must be a finite 2-vector")
    if np.any(xy < 0.0) or np.any(xy > 1.0):
        raise ValueError("normalized coordinates must be in [0,1]")
    return GridCell(
        min(int(np.floor(xy[0] * width_cells)), width_cells - 1),
        min(int(np.floor(xy[1] * height_cells)), height_cells - 1),
    )


def cell_center_normalized(
    cell: GridCell,
    *,
    width_cells: int,
    height_cells: int,
) -> np.ndarray:
    if not (0 <= cell.x < width_cells and 0 <= cell.y < height_cells):
        raise ValueError(f"cell outside {width_cells}x{height_cells}: {cell}")
    return np.asarray(
        [(cell.x + 0.5) / width_cells, (cell.y + 0.5) / height_cells],
        dtype=np.float32,
    )


class SemanticMapDecoder:
    """Decode BlackOut's HWC one-hot map into points and 24x24 cell layers."""

    def __init__(self, resolution_scale: int = DEFAULT_RESOLUTION_SCALE):
        if resolution_scale < 1:
            raise ValueError("resolution_scale must be positive")
        self.resolution_scale = resolution_scale

    def decode(self, graphic: np.ndarray) -> DecodedSemanticMap:
        channels = parse_graphic(graphic).channels
        height, width, _ = channels.shape
        if height % self.resolution_scale or width % self.resolution_scale:
            raise ValueError(
                f"graphic {width}x{height} is not divisible by scale {self.resolution_scale}"
            )
        width_cells = width // self.resolution_scale
        height_cells = height // self.resolution_scale
        points_by_channel: dict[str, tuple[SemanticPoint, ...]] = {}
        cells_by_channel: dict[str, tuple[GridCell, ...]] = {}
        for index, name in enumerate(GRAPHIC_CHANNEL_NAMES):
            rows_columns = np.argwhere(channels[..., index] == 1.0)
            points = []
            cells: set[GridCell] = set()
            for row_value, column_value in rows_columns:
                row = int(row_value)
                column = int(column_value)
                normalized = pixel_to_normalized(
                    row,
                    column,
                    width=width,
                    height=height,
                )
                cell = normalized_to_cell(
                    normalized,
                    width_cells=width_cells,
                    height_cells=height_cells,
                )
                points.append(SemanticPoint(row, column, normalized, cell))
                cells.add(cell)
            points_by_channel[name] = tuple(points)
            cells_by_channel[name] = tuple(sorted(cells))

        wall_grid = np.zeros((height_cells, width_cells), dtype=np.bool_)
        for cell in cells_by_channel["wall"]:
            wall_grid[cell.y, cell.x] = True
        return DecodedSemanticMap(
            height_pixels=height,
            width_pixels=width,
            resolution_scale=self.resolution_scale,
            channel_points=points_by_channel,
            channel_cells=cells_by_channel,
            wall_grid=wall_grid,
        )
