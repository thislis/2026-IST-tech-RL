"""Research bundles and strict capability checks. No implicit submission approval."""

from project_paths import project_root, project_path

from pathlib import Path
import json
import shutil
import torch
from .contracts import ContractError,digest,file_hash
from .checkpoints import atomic_json,fingerprints

REQUIRED=('official_loader_verified','stateful','explicit_episode_decision_id','canonical_agent_ids','dependencies_verified','device_verified','resource_limits_verified','multi_episode_parity_verified')

def verify_capabilities(contract):
    if contract.get('status')!='approved' or any(contract.get(k) is not True for k in REQUIRED):
        raise ContractError('submission contract unresolved: '+', '.join(k for k in REQUIRED if contract.get(k) is not True))
    if not contract.get('loader_sha256') or not contract.get('resource_limits'):
        raise ContractError('loader fingerprint and actual resource limits required')


def export_bundle(checkpoint_path,config,output,*,submission_contract=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if submission_contract is not None:
        verify_capabilities(submission_contract)
        # Only research packaging is implemented until a concrete official format is supplied.
        raise ContractError('official format adapter has not been certified; research export only')
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if payload['sources']!=fingerprints() or payload['config_sha256']!=digest(config):
        raise ContractError('export requires checkpoint source/config snapshot')
    output.mkdir(parents=True)
    root=project_root()
    shutil.copytree(root/'code',output/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(checkpoint_path,output/'policy.pt')
    shutil.copy2(project_path(config['encoder']['path'], root=root),output/'encoder.pt')
    atomic_json(output/'resolved_config.json',config)
    for name in ('requirements.lock','requirements-v7.lock'):
        shutil.copy2(project_path(name, root=root),output/name)
    (output/'NOTICE.txt').write_text('Research export of the local BlackOut RL project and its frozen encoder.\nDependencies are external and retain their licenses; see requirements locks.\nUnity/API assets are not included. No official-loader certification.\n')
    files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    atomic_json(output/'manifest.json',dict(schema='blackout.v8.research_export.v1',submission_ready=False,files=files,
                                           sources=fingerprints(),config_sha256=digest(config)))
    return output
