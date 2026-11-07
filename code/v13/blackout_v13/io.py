"""Atomic records and immutable provenance, shared with existing runtime utilities."""
from blackout_v12.io import ROOT,atomic_json,atomic_torch,read_json,append,digest,file_hash,cpu_state


def source_identity():
    paths=[]
    for folder in ('code/v13','code/shared','code/v12/blackout_v12','code/v11/blackout_v11'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py','.json','.ps1','.sh','.lock') and '__pycache__' not in p.parts)
    return {p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(set(paths))}
