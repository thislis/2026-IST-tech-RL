"""Bounded inference and end-to-end probes, always excluded from league standings."""
import gc
import platform
import time
import numpy as np
import torch
from .common import atomic,read
from .check import fixture
from .adapters import Adapter
from .runtime import execute,resource_sample


def benchmark(roster,directory,cfg,stop=lambda:False,workers_override=None):
    directory.mkdir(parents=True,exist_ok=True)
    from blackout_v10.telemetry import hardware_identity
    hardware=hardware_identity();measurements=[];obs=fixture()
    neural=next((s for s in roster['models'] if s['id']=='v10-C1'),next((s for s in roster['models'] if s['kind']=='bundle'),None))
    chosen_threads=1
    if neural:
        for threads in (1,2):
            torch.set_num_threads(threads)
            for device in ('cpu','mps'):
                if device=='mps' and not torch.backends.mps.is_available():continue
                try:
                    a=Adapter(neural,0,42,device=device);v=torch.from_numpy(np.stack([obs[n]['vector'] for n in a.names])).to(device)
                    g=torch.from_numpy(np.stack([obs[n]['graphic'] for n in a.names])).permute(0,3,1,2).to(device)
                    with torch.inference_mode():
                        fn=a.model.log_probs if hasattr(a.model,'log_probs') else a.model
                        output=fn(v,g).cpu()
                        reference=Adapter(neural,0,42).model
                        rf=reference.log_probs if hasattr(reference,'log_probs') else reference
                        torch.testing.assert_close(output,rf(v.cpu(),g.cpu()),atol=2e-5,rtol=1e-4)
                        values=[]
                        for i in range(14):
                            t=time.perf_counter();fn(v,g).cpu()
                            if device=='mps':torch.mps.synchronize()
                            if i>=2:values.append(time.perf_counter()-t)
                    measurements.append(dict(device=device,threads=threads,p50=float(np.median(values)),valid=True))
                    del a,reference;gc.collect()
                except Exception as exc:measurements.append(dict(device=device,threads=threads,valid=False,error=str(exc)))
        # CPU avoids sharing one GPU queue between heterogeneous, stateful workers.
        # Actual CPU threaded end-to-end profile below determines parallelism.
        cpus=[r for r in measurements if r['valid'] and r['device']=='cpu']
        chosen_threads=min(cpus,key=lambda x:x['p50'])['threads'] if cpus else 1
    model_map={s['id']:s for s in roster['models']}
    representatives=[]
    for kind in ('legacy','bundle','v7','v8_comp','v8_legacy'):
        selected=[s for s in roster['models'] if s['kind']==kind]
        if selected:
            # Include the full connectome, which may dominate inference and memory.
            representatives.append(max(selected,key=lambda s:(s.get('config',{}).get('mode')=='whole_connectome_direct' if isinstance(s.get('config'),dict) else False,s['step'])))
    if len(representatives)<2:representatives=roster['models'][:2]
    profiles=[];started=time.monotonic()
    for workers in ([workers_override] if workers_override else [1,2,4,6,8]):
        if stop():raise InterruptedError('benchmark stopped')
        # All profiles see the same model pairs. Run 8 short tasks to avoid
        # a cheap early-finishing model dominating one concurrency setting.
        tasks=[];local=dict(cfg,workers=workers,threads=chosen_threads,device='cpu',episode_seconds=300)
        for i in range(8):
            a=representatives[i%len(representatives)];b=representatives[(i+1)%len(representatives)]
            tasks.append(dict(id=f'probe-{workers}-{i}',a=a['id'],b=b['id'],repeat=i,side=i%2,
                environment_seed=19200+i,action_seeds=[19200+i,29200+i],models=[a,b],config=local,probe_seconds=cfg['probe_seconds']))
        peak=0.;last=0.;resources=None
        def progress(**_):
            nonlocal peak,last,resources
            if time.monotonic()-last<3:return
            last=time.monotonic();resources=resource_sample();peak=max(peak,resources['rss_gib'])
            if peak>cfg['memory_gib'] or resources['free_gib']<cfg['reserve_gib'] or resources['pressure'].get('critical'):
                raise RuntimeError('benchmark resource limit')
            atomic(directory/'status.json',dict(workers=workers,peak_rss_gib=peak,elapsed_seconds=time.monotonic()-started))
        t=time.monotonic()
        try:
            rows=execute(tasks,directory/f'workers-{workers}',local,stop,progress)
            valid=len(rows)==len(tasks) and all(r['valid'] for r in rows)
            elapsed=time.monotonic()-t;steps=sum(r['result']['episode_steps'] for r in rows if r['valid'])
            size=sum(r['result']['trace_bytes'] for r in rows if r['valid'])
            profiles.append(dict(workers=workers,valid=valid,wall_seconds=elapsed,steps=steps,
                steps_per_second=steps/elapsed,peak_rss_gib=peak,trace_bytes_per_step=size/max(1,steps),
                errors=[r['error'] for r in rows if not r['valid']]))
        except Exception as e:
            profiles.append(dict(workers=workers,valid=False,error=str(e),peak_rss_gib=peak));break
        atomic(directory/'result.json',dict(hardware=hardware,inference=measurements,profiles=profiles))
    valid=[p for p in profiles if p['valid']]
    if not valid:raise RuntimeError('no valid end-to-end benchmark profile')
    best=max(p['steps_per_second'] for p in valid)
    selected=min((p for p in valid if p['steps_per_second']>=best*.95),key=lambda p:p['workers'])
    result=dict(hardware=hardware,inference=measurements,profiles=profiles,selected=dict(workers=selected['workers'],threads=chosen_threads,device='cpu'),
                selection_reason='fastest stable end-to-end throughput; within 5% choose fewer workers; heterogeneous legacy policies remain CPU',
                bytes_per_step=selected['trace_bytes_per_step'],steps_per_second=selected['steps_per_second'],league_games=0)
    atomic(directory/'result.json',result);return result
