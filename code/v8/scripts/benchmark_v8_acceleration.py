#!/usr/bin/env python3
"""Offline recorded-observation replay and kernel timings; never starts Unity/PPO."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import time
ROOT=project_root();sys.path.insert(0,str(ROOT))


def main():
    os.chdir(ROOT)
    import numpy as np
    import torch
    from blackout_rl.v8 import candidates,observation
    from blackout_rl.v8.acceleration import vectorized_mask,compact_shared,validate_shared,installed,EncodedTelemetry
    from blackout_rl.v8.export_parity import fixture_from_windows
    from blackout_rl.v8.policy_factory import make_policy
    from blackout_rl.v8.config import CAPABILITIES
    from blackout_rl.v8.contracts import StepContext,file_hash
    from blackout_rl.v8.checkpoints import CheckpointStore,atomic_json
    from blackout_rl.v8.journal import Journal
    from blackout_rl.v8.window_codec import decode_window
    p=argparse.ArgumentParser();p.add_argument('--output',default='logs/v8/reports/acceleration/offline_benchmark.json');a=p.parse_args()
    trace=project_path('logs/v8/c1_smoke_v2/windows.jsonl', root=ROOT);fixture=fixture_from_windows(trace)
    payload,h=CheckpointStore(project_path('logs/v8/c1_smoke_v2/checkpoints', root=ROOT)).load()
    def policy():
        result=make_policy(payload['config'],capabilities=CAPABILITIES,seed=11)
        result.model.load_state_dict(payload['model']);return result
    first=fixture[0][0];obs=observation.expand(first['observation']);ctx=StepContext(**first['context'])
    # Real wrapper shares each team's graphic array. This fixture restores that legal alias.
    first_map=next(iter(obs.values()))['graphic']
    for o in obs.values():
        if np.array_equal(first_map,o['graphic']):o['graphic']=first_map
    torch.set_num_threads(1);actor=policy();actor.reset(ctx);decision=actor.decide(obs,ctx)
    def timed(fn,n=100):
        for _ in range(3):fn()
        samples=[]
        for _ in range(n):
            t=time.perf_counter();fn();samples.append(time.perf_counter()-t)
        return dict(median_seconds=float(np.median(samples)),p95_seconds=float(np.percentile(samples,95)),n=n)
    pairs=dict(mask=(lambda:candidates.build_mask(obs,ctx.team,decision['planner'],decision['alternatives']),
                    lambda:vectorized_mask(obs,ctx.team,decision['planner'],decision['alternatives'])),
        observation_validation=(lambda:observation.validate_and_hash(obs,ctx.team),lambda:validate_shared(obs,ctx.team)),
        observation_compaction=(lambda:observation.compact(obs),lambda:compact_shared(obs)))
    kernels={}
    for name,(reference,fast) in pairs.items():
        baseline=timed(reference);optimized=timed(fast)
        kernels[name]=dict(reference=baseline,optimized=optimized,speedup=baseline['median_seconds']/optimized['median_seconds'])
    results={};reference_rows=None
    compression=None
    with tempfile.TemporaryDirectory() as tmp:
        with trace.open() as stream:window=json.loads(next(stream))
        metadata={k:v for k,v in window.items() if k!='rows'}
        journal=Journal()
        path=Path(tmp)/'compressed.jsonl'
        start=time.perf_counter()
        journal.window(path,metadata,[json.dumps(row,allow_nan=False,separators=(',', ':')) for row in window['rows']]);journal.close()
        assert decode_window(json.loads(path.read_text()))==window
        raw_bytes=len(json.dumps(window,allow_nan=False).encode())
        compression=dict(original_bytes=raw_bytes,compressed_bytes=path.stat().st_size,
                         size_ratio=path.stat().st_size/raw_bytes,wall_seconds=time.perf_counter()-start,lossless=True)
        dedup_dir=Path(tmp)/'dedup';writer=Journal();encoded=EncodedTelemetry(dedup_dir,journal=writer)
        window_count=0;row_count=0
        with trace.open() as stream:
            for line in stream:
                original=json.loads(line);item={k:v for k,v in original.items() if k!='rows'}
                item['rows']=[(row['global_step'],json.dumps(row,allow_nan=False,separators=(',', ':'))) for row in original['rows']]
                encoded._save(item,original['censored']);window_count+=1;row_count+=len(item['rows'])
        writer.close();lookup={}
        with (dedup_dir/'window_rows.jsonl').open() as stream:
            for line in stream:
                block=decode_window(json.loads(line))
                for key,row in zip(block['row_ids'],block['rows']):
                    assert key not in lookup
                    lookup[key]=row
        with trace.open() as before,(dedup_dir/'windows.jsonl').open() as after:
            for a_row,b_row in zip(before,after):assert json.loads(a_row)==decode_window(json.loads(b_row),lookup)
        dedup_bytes=sum(p.stat().st_size for p in dedup_dir.glob('*.jsonl'))
        deduplication=dict(windows=window_count,repeated_rows=row_count,unique_rows=len(lookup),original_bytes=trace.stat().st_size,
                           encoded_bytes=dedup_bytes,size_ratio=dedup_bytes/trace.stat().st_size,lossless=True)
        for name,threads,fast in [('reference_threads2',2,False),('reference_threads1',1,False),('optimized_threads1',1,True)]:
            torch.set_num_threads(threads);samples=[];rows=[]
            with installed(tmp) if fast else nullcontext():
                actor=policy()
                for _ in range(3):
                    for episode in fixture:
                        for i,row in enumerate(episode):
                            c=StepContext(**row['context']);o=observation.expand(row['observation'])
                            g=next(iter(o.values()))['graphic']
                            for unit in o.values():
                                if np.array_equal(g,unit['graphic']):unit['graphic']=g
                            if i==0:actor.reset(c)
                            t=time.perf_counter();d=actor.decide(o,c,behavior=True);samples.append(time.perf_counter()-t)
                            rows.append((d['action'],d['old_log_prob'].item(),d['planner_state_digest'],d['valid'].tolist(),[x.tolist() for x in d['actions'].values()]))
            if reference_rows is None:reference_rows=rows
            max_error=max(abs(a[1]-b[1]) for a,b in zip(reference_rows,rows))
            if any(a[0]!=b[0] or a[2:]!=b[2:] for a,b in zip(reference_rows,rows)) or max_error>1e-6:
                raise ValueError('recorded-observation behavior parity failed: '+name)
            results[name]=dict(requests=len(samples),median_seconds=float(np.median(samples)),p95_seconds=float(np.percentile(samples,95)),
                total_seconds=sum(samples),maximum_log_probability_error=max_error,actions_masks_planner_exact=True)
    atomic_json(a.output,dict(scope='offline recorded-observation inference and CPU microbenchmarks only',unity_started=False,training_started=False,
        checkpoint_sha256=h,trace_sha256=file_hash(trace),kernels=kernels,replay=results,window_compression=compression,window_deduplication=deduplication,
        limitation='No live throughput, six-worker scaling, thermal stability or MPS performance measured.'))
    print(a.output)
if __name__=='__main__':main()
