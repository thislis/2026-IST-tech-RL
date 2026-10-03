"""Bounded event/control windows plus every-transition exposure and counters."""
from collections import Counter, deque
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time
import numpy as np
import torch


def plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if is_dataclass(value):return plain(asdict(value))
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def append(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(plain(record),allow_nan=False)+'\n')

class Telemetry:
    def __init__(self,directory,window=250,control_probability=.0005,seed=0,max_pending=4):
        self.directory=Path(directory);self.window=window;self.probability=control_probability
        self.rng=np.random.default_rng(seed);self.before=deque(maxlen=window);self.pending=[];self.max_pending=max_pending
        self.counters=Counter();self.sequence=0

    def observe(self,record,triggers=()):
        triggers=list(triggers)
        self.counters['transitions']+=1
        for trigger in triggers:self.counters[trigger]+=1
        if self.rng.random()<self.probability:triggers.append('random_control')
        # Pending windows include subsequent transitions, including episode boundary metadata.
        for pending in self.pending:
            pending['rows'].append(record);pending['remaining']-=1
        done=[w for w in self.pending if w['remaining']==0]
        self.pending=[w for w in self.pending if w['remaining']>0]
        for item in done:self._save(item,False)
        if triggers:
            if len(self.pending)>=self.max_pending:self.counters['dropped_windows']+=1
            else:
                self.pending.append(dict(id=self.sequence,triggers=triggers,rows=list(self.before)+[record],remaining=self.window))
                self.sequence+=1
        self.before.append(record)

    def _save(self,item,censored):
        append(self.directory/'windows.jsonl',dict(**item,censored=censored,inclusion='event_or_random_control',full_state_counterfactual=False))

    def close(self):
        for item in self.pending:self._save(item,True)
        self.pending=[]
        append(self.directory/'counters.jsonl',dict(self.counters))

class RequestTimer:
    def __init__(self):self.samples=[]
    def measure(self,fn,*args,**kwargs):
        start=time.perf_counter()
        try:return fn(*args,**kwargs)
        finally:self.samples.append(time.perf_counter()-start)
    def report(self):
        return dict(n=len(self.samples),**{f'p{q}_seconds':float(np.percentile(self.samples,q)) for q in (50,95,99)}) if self.samples else dict(n=0)
