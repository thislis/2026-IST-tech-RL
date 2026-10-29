"""Exactly two deployable files with the complete Python planner embedded."""
import importlib.util
from pathlib import Path
import sys
import torch
from .io import atomic_torch, cpu_state, digest, file_hash


MODULES = ('contracts','navigation','state','features','tasks','model','assignment','policy')


def bundled_source(config, arm, decoder):
    sources={name:Path(__file__).with_name(name+'.py').read_text(encoding='utf-8') for name in MODULES}
    identity='_blackout_v12_'+digest(dict(sources=sources,config=config,arm=arm,decoder=decoder))[:16]
    # Module registration is required by dataclass and relative imports. No files
    # are extracted or imported from this repository at submission time.
    return ('"""Self-contained v12 team policy; one instance owns one team stream."""\n'
        'import sys as _sys\nimport types as _types\n'
        f'_package_name = {identity!r}\n_sources = {sources!r}\n'
        '_package = _types.ModuleType(_package_name)\n_package.__path__ = []\n'
        '_sys.modules[_package_name] = _package\n'
        'for _name, _source in _sources.items():\n'
        '    _qualified = _package_name + "." + _name\n'
        '    _module = _types.ModuleType(_qualified)\n'
        '    _module.__package__ = _package_name\n'
        '    _sys.modules[_qualified] = _module\n'
        '    exec(compile(_source, _qualified, "exec"), _module.__dict__)\n'
        '_BasePolicy = _sys.modules[_package_name + ".policy"].MyPolicy\n'
        'class MyPolicy(_BasePolicy):\n'
        '    def __init__(self, vector_size=96, n_channels=11, **kwargs):\n'
        '        if vector_size != 96 or n_channels != 11:\n'
        '            raise ValueError("v12 public observation contract changed")\n'
        f'        super().__init__(config={config!r}, arm={arm!r}, decoder={decoder!r})\n')


def export_policy(policy,directory,decoder=None):
    directory=Path(directory)
    selected_decoder=decoder or policy.decoder
    source=bundled_source(policy.config,policy.arm,selected_decoder)
    weights=cpu_state(policy)
    if directory.exists():
        if {p.name for p in directory.iterdir()}!={'policy.py','checkpoint.pt'}:
            raise ValueError('existing submission is not a two-file bundle')
        if (directory/'policy.py').read_text(encoding='utf-8')!=source:
            raise ValueError('immutable export source/config differs')
        existing=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)['policy_state']
        if weights.keys()!=existing.keys() or any(not torch.equal(weights[k],existing[k]) for k in weights):
            raise ValueError('immutable export weights differ')
        return directory
    directory.mkdir(parents=True)
    (directory/'policy.py').write_text(source,encoding='utf-8')
    atomic_torch(directory/'checkpoint.pt',dict(policy_state=weights,schema='v12.submission.v1'))
    return directory


def export_checkpoint(checkpoint,directory,arm=None,decoder='map'):
    from .policy import MyPolicy
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    policy=MyPolicy(payload.get('config'),arm=arm or payload['arm'],decoder=decoder).eval()
    policy.scorer.load_state_dict(payload['scorer_state'])
    return str(export_policy(policy,directory,decoder))


def load_export(directory,device='cpu'):
    from blackout_env import load_checkpoint
    directory=Path(directory)
    spec=importlib.util.spec_from_file_location('submitted_v12_'+file_hash(directory/'policy.py')[:16],directory/'policy.py')
    module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True;spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode=previous
    return load_checkpoint(module.MyPolicy,str(directory/'checkpoint.pt'),state_dict_key='policy_state',
                           device=device,vector_size=96,n_channels=11)
