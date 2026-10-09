"""Explicit v8 adoption of the immutable v7 transport, without rewriting v7 locks."""

from project_paths import project_root, project_path, verify_relocated_source

import importlib.metadata
import json
import platform
from pathlib import Path
from .contracts import ContractError,file_hash

ROOT=project_root()

def validate_legacy_sources():
    path=project_path('logs/v7/reports/main_study_registration.json', root=ROOT)
    record=json.loads(path.read_text())
    for name,expected in {**record['sources'],**record['files']}.items():
        try: verify_relocated_source(name,expected)
        except ValueError as error: raise ContractError('registered v7 file changed: '+name) from error
    if file_hash(project_path(record['source_snapshot'], root=ROOT))!=record['source_snapshot_sha256']:
        raise ContractError('v7 snapshot hash changed')
    return dict(registration_sha256=file_hash(path),verified_original_sources=len(record['sources']),
                policy='registered v7 sources verified with audited path relocation; new experiments use current fingerprints')

def validate_native_runtime():
    legacy=validate_legacy_sources()
    path=project_path('logs/v7/reports/acceleration_registration.json', root=ROOT);record=json.loads(path.read_text())
    for package,expected in record['base_packages'].items():
        actual=importlib.metadata.version(package)
        if package=='protobuf' and actual==record['native_protobuf']:
            # install() prepends the separately pinned package directory.
            import google.protobuf
            if not Path(google.protobuf.__file__).resolve().is_relative_to(project_path('logs/v7/fast_runtime/packages', root=ROOT)):
                raise ContractError('unregistered native protobuf path')
        elif actual!=expected:raise ContractError('base package changed: '+package)
    for name,expected in record['files'].items():
        if file_hash(project_path(name, root=ROOT))!=expected:raise ContractError('native overlay changed: '+name)
    return dict(**legacy,overlay_registration_sha256=file_hash(path),overlay_files=len(record['files']))

def runtime_record(backend):
    result=dict(backend=backend,python=platform.python_version(),platform=platform.platform(),
                packages={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','protobuf','mlagents-envs','grpcio','PyYAML','blackout-env')})
    if backend=='native_queue':result['overlay']=validate_native_runtime()
    return result
