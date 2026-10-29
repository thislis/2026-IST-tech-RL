"""Cross-process audit of actual initial maps, independent of nominal seeds."""
from contextlib import contextmanager
import os
from pathlib import Path
import time
from .io import ROOT, atomic_json, read_json


class MapSplitCollision(ValueError):
    """One physical map was observed in two disjoint study splits."""


class MapSeedMismatch(ValueError):
    """A paired nominal seed produced different actual initial layouts."""


@contextmanager
def _locked(path, timeout=30.):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'0'); stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('actual-map split registry is locked')
                time.sleep(.025)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def register_map(config, split, map_hash, observation_hash, requested_seed, side):
    """Fail before collection/scoring if an actual layout crosses split bounds.

    The map hash includes battery, storage, wall and special-item channels in
    physical Team A's view. Full initial observation hashes remain evidence,
    but entities or side-dependent observations cannot hide a layout collision.
    A study supplies one registry path to every process and stage.
    """
    path = Path(config.get('map_split_registry', ROOT/'logs/v12/map_splits.json'))
    if split not in ('data', 'train', 'dev', 'confirmation', 'test'):
        raise ValueError('unknown actual-map split: '+str(split))
    with _locked(path.with_suffix(path.suffix+'.lock')):
        registry = read_json(path) if path.exists() else dict(schema='v12.actual_maps.1', maps={}, collisions=[])
        seed_maps = registry.setdefault('seed_maps', {})
        seed_key = f'{split}:{int(requested_seed)}'
        previous_map = seed_maps.get(seed_key)
        if previous_map is not None and previous_map != map_hash:
            mismatch = dict(split=split, requested_seed=int(requested_seed), previous_map=previous_map,
                            map_hash=map_hash, observation_hash=observation_hash, side=int(side))
            mismatches = registry.setdefault('seed_mismatches', [])
            if mismatch not in mismatches:
                mismatches.append(mismatch)
                atomic_json(path, registry)
            if config.get('strict_map_splits', True):
                raise MapSeedMismatch(f'paired seed {seed_key} produced different actual map layouts')
            return dict(valid=False, registry=str(path), seed_mismatch=mismatch)
        existing = registry['maps'].get(map_hash)
        if existing and existing['split'] != split:
            collision = dict(map_hash=map_hash, previous_split=existing['split'], split=split,
                             requested_seed=int(requested_seed), observation_hash=observation_hash)
            if collision not in registry['collisions']:
                registry['collisions'].append(collision)
                atomic_json(path, registry)
            if config.get('strict_map_splits', True):
                raise MapSplitCollision(f'actual map {map_hash} crosses {existing["split"]}/{split}')
            return dict(valid=False, registry=str(path), collision=collision)
        entry = registry['maps'].setdefault(map_hash, dict(split=split, observations=[], seeds=[]))
        seed_maps[seed_key] = map_hash
        evidence = dict(sha256=observation_hash, side=int(side))
        if evidence not in entry['observations']:
            entry['observations'].append(evidence)
        if int(requested_seed) not in entry['seeds']:
            entry['seeds'].append(int(requested_seed))
        atomic_json(path, registry)
    return dict(valid=True, registry=str(path), split=split)
