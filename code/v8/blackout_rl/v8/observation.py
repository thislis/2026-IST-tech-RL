"""Lossless legal-observation validation/routing, without hidden engine features."""
import hashlib
import base64
import zlib
import numpy as np
from .contracts import ContractError

def validate_and_hash(observations, team):
    h = hashlib.sha256()
    for slot in range(5):
        name = f'unit_{5*team+slot}'
        if name not in observations:
            raise ContractError('missing canonical agent '+name)
        obs = observations[name]
        vector,graphic = np.asarray(obs['vector']),np.asarray(obs['graphic'])
        if vector.shape != (96,) or graphic.shape != (96,96,11):
            raise ContractError('v8 requires legal 96-vector/96x96x11 HWC contract')
        if not np.isfinite(vector).all() or not np.isfinite(graphic).all():
            raise ContractError('nonfinite observation')
        if not np.all((graphic==0)|(graphic==1)) or not np.all(graphic.sum(-1)==1):
            raise ContractError('zero/interpolated semantic observation')
        if not graphic[...,2].any() or not graphic[...,3].any():
            raise ContractError('missing static wall/storage IDs: empty or stale rendered map')
        h.update(name.encode()); h.update(vector.tobytes()); h.update(graphic.tobytes())
    return h.hexdigest()

def compact(observations):
    return {name: dict(vector=np.asarray(obs['vector']).tolist(),
                       semantic_ids_zlib=base64.b64encode(zlib.compress(np.asarray(obs['graphic']).argmax(-1).astype('uint8').tobytes(),1)).decode('ascii'))
            for name,obs in sorted(observations.items())}

def expand(observations):
    result={}
    for name,obs in observations.items():
        raw=zlib.decompress(base64.b64decode(obs['semantic_ids_zlib']))
        ids=np.frombuffer(raw,np.uint8).reshape(96,96)
        if ids.max()>10:raise ContractError('invalid compressed semantic ID')
        result[name]=dict(vector=np.asarray(obs['vector'],np.float32),graphic=np.eye(11,dtype=np.float32)[ids])
    return result

def world_to_pixel(x,y,width=96,height=96):
    return min(int((1-y)*height),height-1),min(int(x*width),width-1)

def orient_xy(xy,team):
    a=np.asarray(xy)
    return a[...,::-1].copy() if team else a.copy()
