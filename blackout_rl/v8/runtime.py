"""Explicit v8 adoption of the immutable v7 transport, without rewriting v7 locks."""
import importlib.metadata
import json
import platform
from pathlib import Path
from .contracts import ContractError,file_hash

ROOT=Path(__file__).resolve().parents[2]

def validate_legacy_sources():
    path=ROOT/'reports/v7/main_study_registration.json'
    record=json.loads(path.read_text())
    for name,expected in {**record['sources'],**record['files']}.items():
        if file_hash(ROOT/name)!=expected:raise ContractError('registered v7 file changed: '+name)
    if file_hash(ROOT/record['source_snapshot'])!=record['source_snapshot_sha256']:
        raise ContractError('v7 snapshot hash changed')
    return dict(registration_sha256=file_hash(path),verified_original_sources=len(record['sources']),
                policy='registered v7 files unchanged; new v8 files have a separate fingerprint')

def validate_native_runtime():
    legacy=validate_legacy_sources()
    path=ROOT/'reports/v7/acceleration_registration.json';record=json.loads(path.read_text())
    for package,expected in record['base_packages'].items():
        actual=importlib.metadata.version(package)
        if package=='protobuf' and actual==record['native_protobuf']:
            # install() prepends the separately pinned package directory.
            import google.protobuf
            if not Path(google.protobuf.__file__).resolve().is_relative_to(ROOT/'logs/v7/fast_runtime/packages'):
                raise ContractError('unregistered native protobuf path')
        elif actual!=expected:raise ContractError('base package changed: '+package)
    for name,expected in record['files'].items():
        if file_hash(ROOT/name)!=expected:raise ContractError('native overlay changed: '+name)
    return dict(**legacy,overlay_registration_sha256=file_hash(path),overlay_files=len(record['files']))

def runtime_record(backend):
    result=dict(backend=backend,python=platform.python_version(),platform=platform.platform(),
                packages={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','protobuf','mlagents-envs','grpcio','PyYAML','blackout-env')})
    if backend=='native_queue':result['overlay']=validate_native_runtime()
    return result
