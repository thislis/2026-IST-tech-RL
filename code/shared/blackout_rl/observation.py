"""Typed parsers for BlackOut's processed vector and semantic-map observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_UNITS = 10
N_ITEMS = 5
N_ITEM_STATES = N_ITEMS + 1  # index 0 means no held item
N_CLASSES = 3
UNIT_BLOCK_SIZE = 2 + 1 + N_ITEM_STATES
VECTOR_SIZE = N_UNITS * UNIT_BLOCK_SIZE + N_CLASSES + 3

GRAPHIC_CHANNEL_NAMES = (
    "empty",
    "wall",
    "ally_storage",
    "enemy_storage",
    "ally_unit",
    "enemy_unit",
    "battery",
    "buff_speed",
    "debuff_speed",
    "buff_size",
    "debuff_size",
)


@dataclass(frozen=True)
class VectorObservation:
    """Field-level view of one processed `float32[96]` observation."""

    positions: np.ndarray
    team_signs: np.ndarray
    holding_item_one_hot: np.ndarray
    holding_item_ids: np.ndarray
    self_class_one_hot: np.ndarray
    self_class_id: int
    own_score: float
    opponent_score: float
    time_left: float


@dataclass(frozen=True)
class GraphicObservation:
    """Named view of one HWC `float32[H,W,11]` semantic map."""

    channels: np.ndarray

    def mask(self, name: str) -> np.ndarray:
        """Return the H×W binary mask for a semantic channel name."""
        try:
            index = GRAPHIC_CHANNEL_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.channels[..., index]


def _require_float32(array: np.ndarray, *, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(array).__name__}")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32, got {array.dtype}")


def _require_one_hot(array: np.ndarray, *, name: str) -> None:
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{name} must contain only 0/1 values")
    if not np.all(array.sum(axis=-1) == 1.0):
        raise ValueError(f"{name} must contain exactly one active entry per row")


def parse_vector(vector: np.ndarray, *, strict: bool = True) -> VectorObservation:
    """Parse and validate the processed BlackOut vector layout.

    Unit blocks always use absolute `unit_0..unit_9` order. Team signs and the
    final score pair are relative to the observing agent. The observer's own
    unit index is not encoded; callers must retain the PettingZoo agent name or
    a canonical slot ID outside the observation tensor.
    """
    _require_float32(vector, name="vector")
    if vector.shape != (VECTOR_SIZE,):
        raise ValueError(f"vector must have shape ({VECTOR_SIZE},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector contains NaN or infinity")

    blocks = vector[: N_UNITS * UNIT_BLOCK_SIZE].reshape(N_UNITS, UNIT_BLOCK_SIZE)
    positions = blocks[:, :2]
    team_signs = blocks[:, 2]
    holding = blocks[:, 3:]
    class_start = N_UNITS * UNIT_BLOCK_SIZE
    self_class = vector[class_start : class_start + N_CLASSES]
    own_score, opponent_score, time_left = vector[class_start + N_CLASSES :]

    if strict:
        if not np.all((positions >= 0.0) & (positions <= 1.0)):
            raise ValueError("unit positions must be normalized to [0,1]")
        if not np.all(np.isin(team_signs, (-1.0, 1.0))):
            raise ValueError("team_sign must be exactly -1 or +1")
        _require_one_hot(holding, name="holding_item_one_hot")
        _require_one_hot(self_class[np.newaxis, :], name="self_class_one_hot")
        scalars = np.asarray([own_score, opponent_score, time_left])
        if not np.all((scalars >= 0.0) & (scalars <= 1.0)):
            raise ValueError("score/time scalars must be normalized to [0,1]")

    return VectorObservation(
        positions=positions.copy(),
        team_signs=team_signs.copy(),
        holding_item_one_hot=holding.copy(),
        holding_item_ids=np.argmax(holding, axis=1).astype(np.int64),
        self_class_one_hot=self_class.copy(),
        self_class_id=int(np.argmax(self_class)),
        own_score=float(own_score),
        opponent_score=float(opponent_score),
        time_left=float(time_left),
    )


def parse_graphic(graphic: np.ndarray, *, strict: bool = True) -> GraphicObservation:
    """Validate and expose a processed HWC semantic map by channel name."""
    _require_float32(graphic, name="graphic")
    if graphic.ndim != 3 or graphic.shape[-1] != len(GRAPHIC_CHANNEL_NAMES):
        raise ValueError(
            "graphic must have shape (H,W,11), "
            f"got {graphic.shape}"
        )
    if not np.all(np.isfinite(graphic)):
        raise ValueError("graphic contains NaN or infinity")
    if strict and not np.all((graphic == 0.0) | (graphic == 1.0)):
        raise ValueError("graphic semantic masks must be binary")
    if strict and not np.all(graphic.sum(axis=-1) == 1.0):
        raise ValueError("every graphic pixel must belong to exactly one channel")
    return GraphicObservation(channels=graphic)


def hwc_to_chw(graphic: np.ndarray) -> np.ndarray:
    """Convert one validated semantic map from `(H,W,C)` to `(C,H,W)`."""
    parsed = parse_graphic(graphic)
    return np.ascontiguousarray(np.transpose(parsed.channels, (2, 0, 1)))
