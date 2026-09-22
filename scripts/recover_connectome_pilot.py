#!/usr/bin/env python3
"""Fork the failed F1 run into the explicit teammate-v2 revision, retaining optimizer/RNG.

The parent checkpoint/logs are read-only. Only the registered legacy sources and
allowlisted configuration differences are accepted. Every fork records lineage.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from blackout_rl.connectome.graph_artifact import atomic_json, digest, sha256
from blackout_rl.v7_registry import check_config, load_config, source_fingerprint

PARENT = ROOT/'logs/v7/v7_2_malecns_readout_ppo_S0_one_unit_fly64_proxy_v1_repeat1_plasticity_off/11'
CONFIG = ROOT/'configs/v7/v7_2_readout_ppo.yaml'


def validate_transition(saved, cfg, registered_sources):
    if saved.get('schema') != 'blackout.v7.checkpoint.v1' or saved['seed'] != 11:
        raise ValueError('only the failed registered F1 seed 11 may be migrated')
    if saved['sources'] != registered_sources or digest(saved['config']) != saved['config_sha256']:
        raise ValueError('legacy checkpoint provenance mismatch')
    old = saved['config']
    if old['mode'] != 'whole_connectome_direct' or old['controller']['training_mode'] != 'F1_readout_ppo':
        raise ValueError('only F1 may be recovered')
    if old['controller']['remaining_team_policy'] != 'fixed_scripted_3worker_2guard_radius48':
        raise ValueError('unexpected old teammate policy')
    expected = copy.deepcopy(old)
    expected['experiment_id'] += '_teammate_v2'
    expected['controller']['remaining_team_policy'] = 'fixed_scripted_active_slots_v2'
    for key in ('causal_gate_report','causal_gate_sha256'):
        expected['training'][key] = cfg['training'][key]
    if expected != cfg:
        raise ValueError('recovery must not change model, reward, opponent, seeds, dynamics, or training budget')
    if not 0 < saved['global_step'] < cfg['training']['max_environment_steps']:
        raise ValueError('invalid recovery progress')


def copy_prefix(source, destination, count):
    if count < 0 or (not source.exists() and count) or (source.exists() and source.stat().st_size < count):
        raise ValueError(f'invalid committed log offset: {source}')
    if not source.exists():
        return
    with source.open('rb') as src, destination.open('xb') as dst:
        remaining = count
        while remaining:
            block = src.read(min(remaining, 8*1024*1024))
            if not block:raise ValueError('log ended before its committed offset')
            dst.write(block);remaining -= len(block)


def prepare(check=False):
    cfg = load_config(CONFIG)
    check_config(cfg)
    registration = json.loads((ROOT/'reports/v7/pilot_registration.json').read_text())
    if sha256(ROOT/registration['source_snapshot']) != registration['source_snapshot_sha256']:
        raise ValueError('legacy source archive checksum mismatch')
    checkpoint = PARENT/'checkpoints/latest.pt'
    with (PARENT/'run.lock').open('r') as parent_lock:
        fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent_hash = sha256(checkpoint)
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        validate_transition(saved,cfg,registration['source_files'])
        current_sources = source_fingerprint()
        run = ROOT/'logs/v7'/cfg['experiment_id']/'11'
        if run.exists():
            identity = json.loads((run/'recovery.json').read_text())
            manifest = json.loads((run/'experiment_manifest.json').read_text())
            if (identity['parent_checkpoint_sha256'] != parent_hash
                    or manifest['sources'] != current_sources
                    or json.loads((run/'resolved_config.json').read_text()) != cfg):
                raise ValueError('existing recovery differs; refusing to overwrite')
            print(f'Recovery already prepared: {run}')
            return run
        if json.loads((PARENT/'status.json').read_text())['status'] != 'failed':
            raise ValueError('parent run must be stopped and marked failed')
        if check:
            print(f'Recovery ready: {saved["global_step"]} / {cfg["training"]["max_environment_steps"]} steps → {run}')
            return run
        run.parent.mkdir(parents=True,exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.recovery-',dir=run.parent))
        try:
            event = dict(schema='blackout.v7.recovery.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
                reason='Exclude neural-controlled slots from scripted planning; handle unreachable scripted assignments per unit',
                parent_run=str(PARENT),parent_checkpoint_sha256=parent_hash,parent_config_sha256=saved['config_sha256'],
                parent_sources=saved['sources'],new_sources=current_sources,
                resumed_environment_steps=saved['global_step'],failed_environment_steps=json.loads((PARENT/'status.json').read_text())['environment_steps'],
                old_teammate_policy=saved['config']['controller']['remaining_team_policy'],
                new_teammate_policy=cfg['controller']['remaining_team_policy'],
                unity_resume='fresh episode; saved in-flight neural state retained for audit, not restored into Unity',
                migration_script_sha256=sha256(__file__))
            for name,offset in saved['log_offsets'].items():
                if name not in ('training.jsonl','episodes.jsonl','diagnostics.jsonl'):
                    raise ValueError('unexpected log path')
                copy_prefix(PARENT/name,temporary/name,offset)
            for name in ('graph_audit.json','source_manifest.json'):
                shutil.copyfile(PARENT/name,temporary/name)
            manifest = json.loads((PARENT/'experiment_manifest.json').read_text())
            manifest.update(config_sha256=digest(cfg),sources=current_sources,lineage=[event])
            atomic_json(temporary/'resolved_config.json',cfg)
            atomic_json(temporary/'experiment_manifest.json',manifest)
            atomic_json(temporary/'recovery.json',event)
            # This is a derived checkpoint with explicit lineage, not a relabeling of the parent file.
            saved.update(config=cfg,config_sha256=digest(cfg),sources=current_sources,lineage=saved.get('lineage',[])+[event])
            (temporary/'checkpoints').mkdir()
            torch.save(saved,temporary/'checkpoints/latest.pt')
            atomic_json(temporary/'status.json',dict(status='prepared_resume',environment_steps=saved['global_step']))
            temporary.rename(run)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        print(f'Recovery prepared at {saved["global_step"]} steps: {run}')
        return run


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    prepare(args.check)

if __name__=='__main__':main()
