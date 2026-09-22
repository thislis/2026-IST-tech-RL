#!/usr/bin/env python3
"""Acquire official data, build an audited graph, and pin its hashes in config."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from blackout_rl.v7_registry import load_config,path
from blackout_rl.connectome.datasets import prepare_sources
from blackout_rl.connectome.extract_circuit import build_graph
from blackout_rl.connectome.graph_artifact import GraphArtifact,sha256,atomic_json
from blackout_rl.connectome.graph_controls import rewire


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True); p.add_argument('--download',action='store_true')
    args=p.parse_args(); cfg=load_config(args.config)
    directory=path(cfg['data']['source_manifest']).parent
    manifest=prepare_sources(cfg['data']['dataset'],directory,download=args.download)
    destination=path(cfg['data']['graph_artifact'])
    if destination.exists():
        graph=GraphArtifact.load(destination,real=True)
    else:
        if cfg['model'].get('variant')=='rewired':
            parent=GraphArtifact.load(directory/'cx_primary.npz',real=True)
            graph=rewire(parent,cfg['model']['rewire_seed'],rounds=cfg['model']['graph_rounds'])
        else:
            graph=build_graph(cfg['data']['dataset'],directory,manifest)
        graph.save(destination)
    if manifest['files'] != graph.metadata['source_files']:
        raise ValueError('existing graph source mismatch')
    cfg['data'].update(graph_sha256=sha256(destination),graph_metadata_sha256=sha256(destination.with_suffix('.json')),
                       annotation_sha256=manifest['files'][0]['sha256'])
    if cfg['mode']=='residual_graph':
        cfg['model']['input_port_count']=len(graph.metadata['input_ports'])
        cfg['model']['output_pool_count']=len(graph.metadata['output_ports'])
    # Pin dependencies without substituting any missing paths.
    for section,key in (('model','frozen_encoder_checkpoint'),('training','opponent_checkpoint')):
        if key in cfg[section]: cfg[section][key+'_sha256']=sha256(path(cfg[section][key]))
    if cfg['training'].get('causal_gate_report'):
        cfg['training']['causal_gate_sha256']=sha256(path(cfg['training']['causal_gate_report']))
    cfg['evaluation']['split_sha256']=sha256(path(cfg['evaluation']['split_manifest']))
    target=path(args.config); tmp=target.with_suffix('.yaml.tmp'); tmp.write_text(yaml.safe_dump(cfg,sort_keys=False)); tmp.replace(target)
    atomic_json(directory/(destination.stem+'_audit.json'),graph.audit())
    print(json.dumps(dict(config=str(target),graph=str(destination),**graph.audit()),indent=2))

if __name__=='__main__': main()
