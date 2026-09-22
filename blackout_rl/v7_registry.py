"""Resolved experiment config and fail-closed provenance checks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import yaml
import torch
from blackout_rl.connectome.graph_artifact import GraphArtifact, sha256, digest
from blackout_rl.ppo import PPOConfig
from blackout_rl.v7_2.dynamics import DynamicsConfig

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def load_config(filename):
    cfg = yaml.safe_load(path(filename).read_text())
    if 'extends' in cfg:
        base = load_config(path(filename).parent/cfg.pop('extends'))
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v,dict) and isinstance(a.get(k),dict):
                    merge(a[k],v)
                else:
                    a[k]=v
        merge(base,cfg); cfg=base
    return cfg


def check_config(cfg, *, verify_sources=True):
    if cfg.get('schema_version') != 'blackout.v7.experiment.v1':
        raise ValueError('invalid v7 experiment schema')
    if cfg['mode'] not in ('residual_graph','whole_connectome_direct'):
        raise ValueError('unsupported experiment mode')
    data=cfg['data']; mode=cfg['mode']
    if not data.get('graph_sha256') or not data.get('annotation_sha256') or not data.get('graph_metadata_sha256'):
        raise ValueError('unresolved graph/annotation hashes; run prepare_v7_connectome.py first')
    graph = GraphArtifact.load(path(data['graph_artifact']),expected_sha256=data['graph_sha256'],real=True)
    if sha256(path(data['graph_artifact']).with_suffix('.json')) != data['graph_metadata_sha256']:
        raise ValueError('graph metadata/ports/mapping hash mismatch')
    if graph.metadata['source_dataset'] != data['dataset']:
        raise ValueError('dataset mismatch')
    manifest=json.loads(path(data['source_manifest']).read_text())
    if manifest['files'] != graph.metadata['source_files'] or manifest['files'][0]['sha256'] != data['annotation_sha256']:
        raise ValueError('source manifest mismatch')
    if verify_sources:
        for source in manifest['files']:
            if sha256(path(data['source_manifest']).parent/'raw'/source['name']) != source['sha256']:
                raise ValueError('source checksum mismatch: '+source['name'])
    for section,key in (('model','frozen_encoder_checkpoint'),('training','opponent_checkpoint')):
        if key not in cfg.get(section,{}):
            if mode == 'residual_graph' or section == 'training':
                raise ValueError('missing '+key)
            continue
        expected=cfg[section].get(key+'_sha256')
        if not expected or sha256(path(cfg[section][key])) != expected:
            raise ValueError('unresolved or changed checkpoint: '+key)
    executable = path(cfg['environment']['executable'])
    if not executable.is_file() or sha256(executable) != cfg['environment']['executable_sha256']:
        raise ValueError('Unity executable mismatch')
    splits=json.loads(path(cfg['evaluation']['split_manifest']).read_text())
    if sha256(path(cfg['evaluation']['split_manifest'])) != cfg['evaluation']['split_sha256']:
        raise ValueError('split manifest hash mismatch')
    seen=set()
    for split in ('train','dev','confirmation','test'):
        values=splits[split]
        if not values or len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError('seed splits empty/duplicate/overlapping')
        seen.update(values)
    if cfg['training']['failure_seed_priority'] or cfg['controller'].get('action_repeat',1) != 1:
        raise ValueError('primary v7 protocol uses uniform train seeds and action_repeat=1')
    for k in ('max_environment_steps','rollout_steps','save_every','learning_rate'):
        if cfg['training'][k] <= 0:
            raise ValueError('positive '+k+' required')
    ppo=PPOConfig(**cfg['training']['ppo']); ppo.validate()
    if cfg['sensory']['mode'] != 'S0_semantic_retina' or cfg['sensory']['sensor_interval_game_steps'] != 5:
        raise ValueError('only the registered S0 sensor and 5-step refresh are supported')
    if not 0 <= cfg['training']['exploration'] < 1 or not 0 <= cfg['training']['gamma'] <= 1 or not 0 <= cfg['training']['gae_lambda'] <= 1:
        raise ValueError('invalid exploration or discount')
    if cfg['environment']['max_episode_steps'] < 21003 or cfg['runtime']['torch_threads'] < 1:
        raise ValueError('invalid runtime/episode budget')
    previous=0
    for stage in cfg['training']['opponent_schedule']:
        probabilities=stage['probabilities']
        if stage['until'] <= previous or len(probabilities)!=3 or min(probabilities)<0 or abs(sum(probabilities)-1)>1e-9:
            raise ValueError('invalid fixed opponent schedule')
        previous=stage['until']
    if previous < cfg['training']['max_environment_steps']:
        raise ValueError('opponent schedule does not cover budget')
    if mode == 'residual_graph':
        if cfg['model']['input_context_dim'] != 49 or cfg['model']['slots'] != 5 or cfg['controller']['action_schema'] != 'keep_plus_5x8_v6':
            raise ValueError('v6 residual input/action contract must remain fixed')
        if data['dataset'] != 'flywire_fafb' or cfg['model']['carry_neural_state_between_calls'] or cfg['model']['direct_feature_to_action_bypass']:
            raise ValueError('v7-1 primary model contract violation')
        if not all(graph.reachable(cfg['model']['graph_rounds'])):
            raise ValueError('unreachable graph output')
    else:
        from blackout_rl.v7_2.teammates import LEGACY, REVISION
        if cfg['controller']['remaining_team_policy'] not in (LEGACY,REVISION):
            raise ValueError('unsupported teammate policy')
        if data['dataset'] != 'male_cns' or cfg['controller']['plasticity_enabled'] or cfg['controller']['planner_fallback_for_controlled_slots']:
            raise ValueError('v7-2 primary direct-control contract violation')
        if cfg['controller']['training_mode'] not in ('F0_fixed','F1_readout_ppo'):
            raise ValueError('only F0 and F1 are registered')
        if cfg['controller']['training_mode'] == 'F1_readout_ppo':
            gate_path=cfg['training'].get('causal_gate_report')
            if not gate_path or sha256(path(gate_path)) != cfg['training'].get('causal_gate_sha256'):
                raise ValueError('F1 requires the pinned real-Unity causal validation report')
            gate=json.loads(path(gate_path).read_text())
            if not gate.get('causal_gate_passed') or gate['graph_sha256'] != data['graph_sha256']:
                raise ValueError('F1 causal gate failed or graph changed')
            tested=gate['config']
            if tested['controller']['remaining_team_policy'] != cfg['controller']['remaining_team_policy']:
                raise ValueError('F1 teammate revision has not passed causal validation')
            if tested['brain'] != cfg['brain'] or tested['sensory'] != cfg['sensory'] or tested['controller']['fixed_decoder'] != cfg['controller']['fixed_decoder']:
                raise ValueError('F1 dynamics/sensor differ from validated F0 conditions')

        if cfg['controller']['fixed_decoder'] != dict(dt=.02,turn_gain=40.,turn_limit=2.,movement_threshold=.02):
            raise ValueError('unregistered F0 decoder; create and validate a new protocol')
        if cfg['brain'] != asdict(DynamicsConfig()):
            raise ValueError('unregistered whole-brain dynamics')
        if not graph.metadata.get('retinal_mapping') or graph.n != 166700:
            raise ValueError('full MaleCNS graph and retinal mapping required')
    return graph,splits


def source_fingerprint():
    files = sorted(list((ROOT/'blackout_rl').rglob('*.py')) + list((ROOT/'scripts').glob('*v7*')))
    return {str(p.relative_to(ROOT)):sha256(p) for p in files if p.is_file()}


def make_model(cfg,graph):
    if cfg['mode']=='residual_graph':
        from blackout_rl.model_contract import load_checkpoint
        from blackout_rl.v7_1.model import V7ResidualModel
        actor,_=load_checkpoint(path(cfg['model']['frozen_encoder_checkpoint']))
        c=cfg['model']
        return V7ResidualModel(actor.actor_critic,graph,variant=c['variant'],channels=c['node_channels'],rounds=c['graph_rounds'],
                               leak=c['leak'],initial_override_probability=c['initial_override_probability'])
    from blackout_rl.v7_2.decoder import ReadoutModel
    model = ReadoutModel(len(graph.metadata['output_ports']))
    if cfg['controller']['training_mode'] == 'F0_fixed':
        model.requires_grad_(False)
    return model
