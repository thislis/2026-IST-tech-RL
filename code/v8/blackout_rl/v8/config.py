
from project_paths import project_root, project_path, source_files

from pathlib import Path
import copy
import json
import yaml
from .contracts import ContractError,digest,file_hash,bundle_hash
from .policy_factory import resolve_policy

ROOT=project_root()
CAPABILITIES=dict(explicit_episode_decision_id=True,canonical_agent_ids=True,stateful=True,scope='offline_model_fixtures_only')

def load_config(path,*,require_build=True):
    cfg=yaml.safe_load(project_path(path, root=ROOT).read_text())
    allowed={'schema','experiment_id','policy','encoder','opponent','environment','training','telemetry','runtime','evaluation'}
    if set(cfg)!=allowed or cfg['schema']!='blackout.v8.config.v1':raise ContractError('unknown/incomplete config schema')
    nested=dict(encoder={'path','sha256'},opponent={'path','sha256'},
        environment={'build','build_manifest','max_episode_steps','time_scale','timeout_seconds'},
        training={'seed','steps','rollout','save_every','maps','gamma','gae_lambda','reward','opponent_schedule','ppo'},
        telemetry={'window','control_probability','max_pending'},runtime={'backend','torch_threads'},
        evaluation={'seed','dev_maps','primary'})
    for name,keys in nested.items():
        if set(cfg[name])!=keys:raise ContractError('unknown/incomplete '+name+' config')
    from .trainer import DEFAULT_PPO
    if set(cfg['training']['ppo'])!=set(DEFAULT_PPO):raise ContractError('explicit complete PPO config required')
    if set(cfg['training']['reward'])!={'target_score','terminal_bonus'}:raise ContractError('unknown reward config')
    if cfg['evaluation']['primary']!='provided_api_unverified':raise ContractError('retired research metric; original API requires a new evaluation contract')
    cfg['policy']=resolve_policy(cfg['policy'])
    for key in ('encoder','opponent'):
        if file_hash(project_path(cfg[key]['path'], root=ROOT))!=cfg[key]['sha256']:raise ContractError(key+' checkpoint changed')
    sources=[str(p.relative_to(ROOT)) for p in source_files(ROOT, 'blackout_rl') if 'v8' not in p.relative_to(ROOT).parts]
    cfg['opponent']['source_sha256']={p:file_hash(project_path(p, root=ROOT)) for p in sources}
    train=cfg['training']
    if not 0<train['gamma']<=1 or not 0<=train['gae_lambda']<=1:raise ContractError('invalid return estimator')
    ends=[s['until'] for s in train['opponent_schedule']]
    if not ends or ends!=sorted(set(ends)):raise ContractError('schedule boundaries must increase')
    for key in ('steps','rollout','save_every'):
        if train[key]<1:raise ContractError('positive '+key+' required')
    if train['steps']>train['opponent_schedule'][-1]['until']-1:raise ContractError('schedule does not cover budget')
    for stage in train['opponent_schedule']:
        if set(stage)!={'until','probabilities'}:raise ContractError('unknown opponent schedule field')
        if len(stage['probabilities'])!=3 or abs(sum(stage['probabilities'])-1)>1e-9 or min(stage['probabilities'])<0:raise ContractError('invalid opponent schedule')
    if not train['maps'] or len(set(train['maps']))!=len(train['maps']):raise ContractError('invalid train maps')
    if set(train['maps']) & set(cfg['evaluation']['dev_maps']):raise ContractError('train/dev overlap')
    if cfg['runtime']['backend']!='original':raise ContractError('provided environment must use the original runtime')
    from .runtime import runtime_record
    cfg['runtime']['provenance']=runtime_record(cfg['runtime']['backend'])
    contract=project_path('code/v8/contracts/environment_contract.json', root=ROOT)
    cfg['environment']['protocol_sha256']=file_hash(contract)
    if cfg['environment']['build']!='builds/BlackOut.app' or cfg['environment']['timeout_seconds']!=0:
        raise ContractError('modified game/timer configuration is retired')
    if require_build:
        from .provided_environment import verify_original, reject_research_environment
        verify_original()
        # The old collector requires extra engine fields. Do not invent them or
        # silently train with different rewards/terminal semantics on the raw API.
        reject_research_environment()
    return cfg
