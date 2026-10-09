"""Content-addressed checkpoints, validated lineage, explicit restart resume."""

from project_paths import project_root, project_path, source_files

import io
import json
from pathlib import Path
import os
import platform
import random
import torch
from .contracts import ContractError,digest,file_hash

ROOT=project_root()

def fingerprints():
    paths=source_files(ROOT, 'blackout_rl')+list((ROOT/'code/v8/scripts').glob('*v8*.py'))
    paths += [ROOT/'code/shared/project_paths.py', ROOT/'code/shared/legacy_paths.json', ROOT/'code/shared/relocation_sources.json']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

class CheckpointStore:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
    def save(self,payload,parent=None):
        if parent is not None and not (self.directory/(parent+'.pt')).is_file():
            raise ContractError('missing parent checkpoint')
        data=dict(payload,schema='blackout.v8.checkpoint.v1',parent_sha256=parent)
        stream=io.BytesIO();torch.save(data,stream)
        import hashlib
        content=stream.getvalue();h=hashlib.sha256(content).hexdigest();path=self.directory/(h+'.pt')
        if path.exists():
            if file_hash(path)!=h:raise ContractError('corrupt immutable checkpoint')
        else:
            with path.open('xb') as f:f.write(content)
        atomic_json(self.directory/'latest.json',dict(sha256=h,path=path.name,step=payload['global_step']))
        return h
    def load(self,sha=None):
        if sha is None:sha=json.loads((self.directory/'latest.json').read_text())['sha256']
        path=self.directory/(sha+'.pt')
        if file_hash(path)!=sha:raise ContractError('checkpoint content hash mismatch')
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['schema']!='blackout.v8.checkpoint.v1':raise ContractError('checkpoint schema mismatch')
        parent=data['parent_sha256']
        if parent is not None and file_hash(self.directory/(parent+'.pt'))!=parent:raise ContractError('parent hash mismatch')
        return data,sha
