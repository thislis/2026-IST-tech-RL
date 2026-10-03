"""Explicit, reversible CPU/IO overlay. No model, reward, physics or RNG changes."""
from contextlib import contextmanager
from unittest.mock import patch
import base64
import hashlib
import json
import zlib
import numpy as np
import torch
from .contracts import ContractError
from . import observation, candidates, policy_factory, collector, telemetry, exposure, trainer, evaluation
from .journal import Journal
from .fast_features import FastPlannerAdapter

REVISION = 'v8_cpu_io_v1'


def compact_shared(observations):
    maps = {}
    result = {}
    for name, obs in sorted(observations.items()):
        graphic = np.asarray(obs['graphic'])
        key = id(graphic)
        if key not in maps:
            maps[key] = base64.b64encode(zlib.compress(graphic.argmax(-1).astype('uint8').tobytes(), 1)).decode('ascii')
        result[name] = dict(vector=np.asarray(obs['vector']).tolist(), semantic_ids_zlib=maps[key])
    return result


def validate_shared(observations, team):
    """Memoize identical array objects only within this call; retain old hash bytes."""
    h = hashlib.sha256()
    maps = {}
    for slot in range(5):
        name = f'unit_{5 * team + slot}'
        if name not in observations:
            raise ContractError('missing canonical agent ' + name)
        vector = np.asarray(observations[name]['vector'])
        graphic = np.asarray(observations[name]['graphic'])
        if vector.shape != (96,) or graphic.shape != (96, 96, 11):
            raise ContractError('v8 requires legal observation shape')
        if not np.isfinite(vector).all():
            raise ContractError('nonfinite vector')
        if id(graphic) not in maps:
            if not np.isfinite(graphic).all():
                raise ContractError('nonfinite graphic')
            if not np.all((graphic == 0) | (graphic == 1)) or not np.all(graphic.sum(-1) == 1):
                raise ContractError('zero/interpolated semantic observation')
            if not graphic[..., 2].any() or not graphic[..., 3].any():
                raise ContractError('missing storage IDs')
            maps[id(graphic)] = graphic.tobytes()
        h.update(name.encode())
        h.update(vector.tobytes())
        h.update(maps[id(graphic)])
    return h.hexdigest()


def vectorized_mask(observations, team, planner, alternatives):
    """Preserve float32 endpoint arithmetic, float64 offsets and original reasons."""
    reasons = []
    offsets = np.array(((0, 0), (.014, 0), (-.014, 0), (0, .014), (0, -.014)))
    for slot in range(5):
        obs = observations[f'unit_{team * 5 + slot}']
        own = obs['vector'][:90].reshape(10, 9)[team * 5 + slot, :2]
        walls = obs['graphic'][:, :, 1]
        alternative = alternatives[slot]
        endpoint = own + alternative * (9 * .02 / 24)
        points = own + (endpoint - own)[:, None, :] * np.array((.5, 1.), dtype=endpoint.dtype)[None, :, None]
        cells = (points * 24).astype(int)
        forbidden = ({(c.x, c.y) for c in candidates.HUNTER_SHRINE_CELLS} if slot < 3 else set()) | {
            (candidates.carrier_shrine_cell(team).x, candidates.carrier_shrine_cell(team).y)}
        shrine = np.zeros(8, dtype=bool)
        for cell in forbidden:
            shrine |= (cells == cell).all(-1).any(-1)
        samples = points[:, :, None, :] + offsets
        x, y = samples[..., 0], samples[..., 1]
        inside = (x >= 0) & (x < 1) & (y >= 0) & (y < 1)
        # Invalid samples are indexed safely then excluded, exactly like the scalar branches.
        row = np.clip(((1 - y) * len(walls)).astype(int), 0, len(walls) - 1)
        col = np.clip((x * walls.shape[1]).astype(int), 0, walls.shape[1] - 1)
        blocked = ((walls[row, col] > .5) & inside).any((1, 2))
        boundary = (~inside).any((1, 2))
        duplicate = np.isclose(alternative, planner[slot], atol=1e-5, rtol=1e-5).all(-1)
        for d in range(8):
            reasons.append(sorted(label for yes, label in (
                (duplicate[d], 'duplicate_candidate:planner'),
                (shrine[d], 'heuristic_preference:role_shrine'),
                (boundary[d], 'geometry_estimate:boundary_margin'),
                (blocked[d], 'geometry_estimate:wall_margin')) if yes))
    return torch.tensor([not r for r in reasons], dtype=torch.bool), reasons


class EncodedTelemetry(telemetry.Telemetry):
    """Serialize each immutable transition once, even across overlapping windows."""
    def __init__(self, *args, journal, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal = journal
        self.written_through = -1

    def observe(self, record, triggers=()):
        row = json.dumps(telemetry.plain(record), allow_nan=False, separators=(',', ':'))
        row_id = record.get('global_step', self.counters['transitions'] + 1)
        super().observe((row_id, row), triggers)

    def _save(self, item, censored):
        metadata = {k: v for k, v in item.items() if k != 'rows'}
        metadata.update(censored=censored, inclusion='event_or_random_control', full_state_counterfactual=False)
        fresh = [(i,row) for i,row in item['rows'] if i>self.written_through]
        if fresh:
            self.journal.window(self.directory/'window_rows.jsonl',dict(row_ids=[i for i,_ in fresh]),[row for _,row in fresh])
            self.written_through=fresh[-1][0]
        self.journal.append(self.directory/'windows.jsonl',dict(metadata,rows_encoding='refs-v1',
            rows_store='window_rows.jsonl',row_ids=[i for i,_ in item['rows']]))


@contextmanager
def installed(directory):
    """One interpreter per learner: patches never cross independent runs."""
    from contextlib import ExitStack
    journal = Journal()
    original_state = collector.Collector.state_dict

    def state(instance):
        journal.flush()  # trainer computes log_offsets after this call
        return original_state(instance)

    class Windows(EncodedTelemetry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, journal=journal, **kwargs)

    with ExitStack() as stack:
        for module in (telemetry, exposure, trainer, collector, evaluation):
            stack.enter_context(patch.object(module, 'append', journal.append))
        stack.enter_context(patch.object(collector.Collector, 'state_dict', state))
        stack.enter_context(patch.object(collector, 'Telemetry', Windows))
        stack.enter_context(patch.object(collector, 'compact', compact_shared))
        stack.enter_context(patch.object(policy_factory, 'validate_and_hash', validate_shared))
        stack.enter_context(patch.object(policy_factory, 'build_mask', vectorized_mask))
        stack.enter_context(patch.object(policy_factory, 'PlannerAdapter', FastPlannerAdapter))
        try:
            yield journal
        finally:
            journal.close()
