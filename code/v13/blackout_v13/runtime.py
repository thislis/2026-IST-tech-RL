"""Hardware-aware settings. Live throughput probes run only inside explicit launch."""
import ctypes
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import torch
from .io import ROOT,atomic_json,append,cpu_state
from blackout_v12.runtime import hardware_identity,FileLease,process_options,child_environment


def available_memory():
    if os.name=='nt':
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                ('total','available','page','page_available','virtual','virtual_available','extended')]
        s=Status();s.length=ctypes.sizeof(s)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise ctypes.WinError()
        return int(s.available)
    return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')


def gpu_free():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=5,**process_options())
        return min(float(line)*2**20 for line in r.stdout.strip().splitlines())
    except (OSError,ValueError,subprocess.TimeoutExpired):return None


def resource_room(config,additional_workers=0):
    ram=available_memory();disk=shutil.disk_usage(ROOT).free;gpu=gpu_free()
    return dict(ram=ram,disk=disk,gpu=gpu,
        ok=(ram>(config['memory_reserve_gib']+additional_workers*config['worker_memory_gib'])*2**30
            and disk>config['disk_reserve_gib']*2**30
            and (gpu is None or gpu>(config['gpu_reserve_gib']+additional_workers*config.get('worker_gpu_memory_gib',.25))*2**30)))


def tune_workers(pool,weights,config,directory,stop=lambda:False):
    """Measure completed chunks, excluding startup/warmup; bounded by real memory headroom."""
    rows=[];best=None;plateaus=0;baseline_ram=available_memory();baseline_gpu=gpu_free()
    candidates=([int(config['workers'])] if config['workers']!='auto' else config['worker_candidates'])
    for count in candidates:
        if stop():raise InterruptedError('stopped during resource tuning')
        if count>config['worker_cap']:continue
        room=resource_room(config,max(0,count-len(pool.workers)))
        if not room['ok']:
            rows.append(dict(workers=count,skipped='resource_headroom',resources=room));break
        pool.resize(count)
        pool.collect(count*config['chunk_steps'],weights,'worker',stop=stop)
        samples=[];started=time.perf_counter();elapsed=0.;total=0;timing={}
        while elapsed<config['tune_seconds'] or len(samples)<2:
            begin=time.perf_counter()
            chunks=pool.collect(count*config['chunk_steps'],weights,'worker',stop=stop)
            wall=time.perf_counter()-begin;steps=sum(c['env_steps'] for c in chunks)
            samples.append(steps/wall);total+=steps;elapsed=time.perf_counter()-started
            for c in chunks:
                for k,v in c['timing'].items():timing[k]=timing.get(k,0)+v
        row=dict(workers=count,env_steps_per_second=total/elapsed,median_chunk_rate=statistics.median(samples),
                 samples=samples,timing=timing,resources=resource_room(config),wall_seconds=elapsed)
        rows.append(row);append(Path(directory)/'worker_probes.jsonl',row)
        if best is None or row['env_steps_per_second']>best['env_steps_per_second']*1.02:
            best=row;plateaus=0
        else:plateaus+=1
        if plateaus>=config['tune_plateaus']:break
    if best is None:raise RuntimeError('insufficient resource headroom for one worker')
    pool.resize(best['workers']);pool.reset()
    observed=max(.75,(baseline_ram-best['resources']['ram'])/best['workers']/2**30*1.25)
    observed_gpu=config.get('worker_gpu_memory_gib',.25)
    if baseline_gpu is not None and best['resources']['gpu'] is not None:
        observed_gpu=max(.05,(baseline_gpu-best['resources']['gpu'])/best['workers']/2**30*1.25)
    report=dict(selected_workers=best['workers'],measured_worker_memory_gib=observed,
                measured_worker_gpu_memory_gib=observed_gpu,probes=rows,
                selection='measured completed env steps per second')
    atomic_json(Path(directory)/'workers.json',report)
    return report


def tune_learner(policy,config,directory):
    """Separate synthetic compute probe, performed by the user's explicit run only."""
    import copy
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    if config['learner_device']!='auto':devices=[config['learner_device']]
    rows=[]
    for device in devices:
        for sequences in (1,2,4,8):
            model=copy.deepcopy(policy.worker).to(device).train()
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            length=config['chunk_steps'];n=sequences*5
            try:
                local=torch.randn(length,n,13,9,9,device=device)
                vector=torch.zeros(length,n,32,device=device)
                lagged=torch.zeros(length,n,len(config['lags']),128,device=device);lagged[...,121]=1
                hidden=torch.zeros(n,64,device=device)
                measurements=[]
                for repeat in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    if device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter();encoded=model.encode_lagged(local,vector,lagged);h=hidden
                    losses=[]
                    for j in range(length):
                        logits,h=model.step_encoded(encoded[j],h);losses.append(logits.square().mean())
                    torch.stack(losses).mean().backward();optimizer.step()
                    if device=='cuda':torch.cuda.synchronize()
                    if repeat:measurements.append(time.perf_counter()-start)
                rows.append(dict(device=device,sequences=sequences,seconds=statistics.median(measurements),
                    samples_per_second=length*sequences/statistics.median(measurements)))
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(device=device,sequences=sequences,error='out_of_memory'));break
            finally:
                del model,optimizer
                if device=='cuda':torch.cuda.empty_cache()
    valid=[r for r in rows if 'samples_per_second' in r]
    if not valid:raise RuntimeError('no usable learner device')
    best=max(valid,key=lambda r:r['samples_per_second'])
    result=dict(selected_device=best['device'],sequence_minibatch=best['sequences'],probes=rows,
                scope='synthetic worker training compute, not game throughput')
    atomic_json(Path(directory)/'learner.json',result)
    return result
