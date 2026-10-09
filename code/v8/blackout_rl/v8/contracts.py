"""Strict, versioned research contracts; engine truth is never inferred from rewards."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

class ContractError(ValueError):
    pass

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

BUNDLE_RUNTIME_EXCLUSIONS = ('Contents/ML-Agents/Timers/',)

def bundle_files(path):
    """Unity launcher alone does not identify managed code or game assets."""
    root=Path(path)
    files={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.name!='.DS_Store'
           and not str(p.relative_to(root)).startswith(BUNDLE_RUNTIME_EXCLUSIONS)}
    if not files:raise ContractError('empty build bundle')
    return files

def bundle_hash(path):
    return digest(bundle_files(path))

@dataclass(frozen=True)
class StepContext:
    run_id: str
    env_id: str
    episode_id: str
    decision_id: int
    unity_tick: int
    game_time: float
    map_seed: int
    team: int
    behavior_version: int = 0

    def __post_init__(self):
        if not all((self.run_id, self.env_id, self.episode_id)) or self.team not in (0, 1):
            raise ContractError('explicit run/env/episode/team required')
        if self.decision_id < 0 or self.unity_tick < 0 or not math.isfinite(self.game_time) or self.game_time < 0:
            raise ContractError('negative decision/physical clock')

    @property
    def key(self):
        return self.run_id, self.env_id, self.episode_id, self.team

@dataclass(frozen=True)
class OutcomeEvent:
    run_id: str
    episode_id: str
    event_id: str
    decision_id: int
    unity_tick: int
    game_time: float
    final_score_points: Optional[tuple]
    winner_team: Optional[int]
    winner_source: str
    termination_reason: str
    terminated: bool
    truncated: bool
    build_sha256: str
    protocol_sha256: str
    final_observation_id: Optional[str] = None
    score_snapshot_revision: int = 1
    schema_version: str = 'blackout.v8.outcome.v1'

    def __post_init__(self):
        if self.schema_version != 'blackout.v8.outcome.v1' or not all((self.run_id, self.episode_id, self.event_id)):
            raise ContractError('invalid outcome identity/schema')
        if self.terminated == self.truncated:
            raise ContractError('exactly one termination/truncation flag required')
        if self.terminated and (self.winner_source != 'engine' or self.winner_team not in (-1, 0, 1)):
            raise ContractError('authoritative engine winner required; null is not draw')
        if self.final_score_points is not None:
            points = tuple(self.final_score_points)
            if len(points) != 2 or any(type(x) is not int or x < 0 for x in points):
                raise ContractError('score_points must be two nonnegative integers')
            object.__setattr__(self, 'final_score_points', points)
        if self.terminated and self.final_score_points is None:
            raise ContractError('missing authoritative final score')
        if self.truncated and (self.final_observation_id is None or self.winner_team is not None):
            raise ContractError('truncation needs final observation, no invented winner')
        if not self.termination_reason or not math.isfinite(self.game_time) or min(self.decision_id, self.unity_tick, self.game_time) < 0:
            raise ContractError('invalid outcome clock/reason')
        if any(len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in (self.build_sha256, self.protocol_sha256)):
            raise ContractError('build/protocol SHA256 required')

    @property
    def key(self):
        return self.run_id, self.episode_id, self.event_id

    def to_dict(self):
        return asdict(self)
