"""Public canonical-team contract; no hidden state or opponent class inputs."""
from dataclasses import dataclass
import numpy as np

SCHEMA = 'blackout.v12.features.1'
STATE_SCHEMA = 'blackout.v12.state.1'
TEAM_SIZE = 5


@dataclass
class Observation:
    vector: np.ndarray
    semantic: np.ndarray
    positions: np.ndarray
    holding: np.ndarray
    classes: np.ndarray
    enemy_positions: np.ndarray
    enemy_holding: np.ndarray
    unit_ids: np.ndarray
    side: int
    time_left: float
    scores: np.ndarray


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def observe(vector, graphic):
    """Decode one five-row stream and reflect physical B across y=x.

    The map is checked for team equality before a single 24x24 decode. The
    runner must supply the canonical absolute-entity ordering of team rows.
    """
    vector, graphic = as_numpy(vector), as_numpy(graphic)
    if vector.shape != (5, 96) or graphic.shape != (5, 11, 96, 96):
        raise ValueError('v12 requires canonical vector[5,96], graphic[5,11,96,96]')
    if vector.dtype != np.float32 or graphic.dtype != np.float32:
        raise TypeError('public inputs must be float32')
    if not np.isfinite(vector).all() or not np.isfinite(graphic).all():
        raise ValueError('nonfinite public observation')
    if not all(np.array_equal(graphic[0], graphic[i]) for i in range(1, 5)):
        raise ValueError('canonical team maps differ')
    if not np.all(vector[:, :90] == vector[0, :90]):
        raise ValueError('canonical team entity blocks differ')
    if not np.all(vector[:,93:] == vector[0,93:]):
        raise ValueError('canonical team scores or clock differ')
    entities = vector[0, :90].reshape(10, 9).copy()
    own_ids = np.flatnonzero(entities[:, 2] > 0)
    other_ids = np.flatnonzero(entities[:, 2] < 0)
    if len(own_ids) != 5 or len(other_ids) != 5:
        raise ValueError('observation must contain five allies and five opponents')
    side = int(entities[0, 2] < 0)
    image = graphic[0]
    if side:
        entities[:, :2] = entities[:, :2][:, ::-1]
        image = image.transpose(0, 2, 1)[:, ::-1, ::-1]
    # Top-down image -> bottom-left [channel,y,x] cell arrays.
    semantic = image.reshape(11, 24, 4, 24, 4).max(axis=(2, 4))[:, ::-1, :] > .5
    normalized = vector.copy()
    normalized[:, :90] = entities.reshape(90)
    return Observation(normalized, semantic.copy(), entities[own_ids, :2],
                       entities[own_ids, 3:].argmax(-1), vector[:, 90:93].argmax(-1),
                       entities[other_ids, :2], entities[other_ids, 3:].argmax(-1),
                       own_ids, side, float(vector[0, 95]), vector[0, 93:95].copy())
