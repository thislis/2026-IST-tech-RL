"""Synchronized short collection-only scaling test, never a main-study launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--direct',type=int,default=1)
    p.add_argument('--seconds',type=int,default=20)
    a=p.parse_args()
    directory=ROOT/f'reports/v7/acceleration_parallel_{a.workers}_{a.direct}'
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='connectome-scaling-') as tmp:
        barrier=Path(tmp)/'go'
        children=[]
        handles=[]
        try:
            for index in range(a.workers):
                direct=index<a.direct
                config='v7_2_readout_ppo' if direct else 'v7_1_rewired'
                command=[sys.executable,str(ROOT/'scripts/benchmark_connectome_speed.py'),'--fast','both','--engine',
                         '--config',f'configs/v7/main_study/{config}.yaml','--seconds',str(a.seconds),
                         '--barrier',str(barrier),'--output',str(directory/f'{index}.json')]
                if direct:command.append('--mps-brain')
                output=(directory/f'{index}.log').open('wb');handles.append(output)
                children.append(subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='1')))
            deadline=time.monotonic()+120
            while len(list(Path(tmp).glob('go.ready.*')))!=a.workers:
                if any(c.poll() is not None for c in children):raise RuntimeError('worker exited before barrier')
                if time.monotonic()>deadline:raise TimeoutError('worker startup')
                time.sleep(.1)
            barrier.touch()
            for child in children:
                if child.wait()!=0:raise RuntimeError('benchmark failed')
            rows=[json.loads((directory/f'{i}.json').read_text()) for i in range(a.workers)]
            result=dict(workers=a.workers,direct=a.direct,aggregate_sps=sum(r['steps_per_second'] for r in rows),
                        rows=rows,training_updates=0,note='short concurrent collection; excludes PPO updates and long-run thermal effects')
            (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
        finally:
            for c in children:
                if c.poll() is None:c.send_signal(2)
            for c in children:c.wait()
            for h in handles:h.close()


if __name__=='__main__':main()
