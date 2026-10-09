#!/usr/bin/env python3
"""Inference-only adapter audit of pinned v7 final graphs; no new episodes."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import argparse,json,sys
from pathlib import Path
ROOT=project_root();sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    from blackout_rl.v7_registry import make_model
    from blackout_rl.connectome.graph_artifact import GraphArtifact
    from blackout_rl.v8.distributions import flat,exact_factorization
    from blackout_rl.v8.decoders import decode
    from blackout_rl.v8.contracts import file_hash
    from blackout_rl.v8.checkpoints import atomic_json
    from blackout_rl.v8.runtime import validate_legacy_sources
    provenance=validate_legacy_sources()
    torch.set_num_threads(2);payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);cfg=payload['config']
    graph=GraphArtifact.load(project_path(cfg['data']['graph_artifact'], root=ROOT),expected_sha256=cfg['data']['graph_sha256'],real=True)
    with torch.random.fork_rng():model=make_model(cfg,graph)
    model.load_state_dict(payload['model']);model.eval()
    generator=torch.Generator().manual_seed(927)
    features=torch.randn(96,5,177,generator=generator)*torch.logspace(-3,3,96)[:,None,None]
    with torch.no_grad():
        logits=model.logits(features);valid=torch.ones(96,40,dtype=torch.bool)
        old=flat(logits[:,0],logits[:,1:],valid);new=exact_factorization(logits,valid)
    result=dict(legacy_sources=provenance,checkpoint_sha256=file_hash(a.checkpoint),graph_sha256=file_hash(project_path(cfg['data']['graph_artifact'], root=ROOT)),
                scope='synthetic features, not chronological observation replay or victory evidence',samples=96,
                max_probability_error=float((old.probs-new.probs).abs().max()),max_entropy_error=float((old.entropy()-new.entropy()).abs().max()),
                joint_action_changes=int((decode(old,'joint_argmax_v1')!=decode(new,'joint_argmax_v1')).sum()),
                threshold_action_changes=int((decode(old,'joint_argmax_v1')!=decode(new,'gate_then_conditional_argmax_v1')).sum()),q=new.q.tolist())
    # Existing read-only numerical bound implementation, never resave the legacy checkpoint.
    audit=project_path('artifacts/documents/plans/v8_research_requests_2026-09-27/internal_analysis', root=ROOT)
    if (audit/'analyze.py').exists():
        sys.path.insert(0,str(audit));from analyze import interval_bound
        result['interval_bound']=interval_bound(payload['model'],cfg)
    atomic_json(a.output,result);print(a.output)
if __name__=='__main__':main()
