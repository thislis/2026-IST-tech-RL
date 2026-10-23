"""Bounded, lossless vector/action transition chunks; graphic summaries are proxies."""
import gzip
import json
import os
from pathlib import Path
import numpy as np
from .common import atomic, sha

NAMES=tuple(f'unit_{i}' for i in range(10))

def vectors(obs):
    return np.stack([obs[n]['vector'] for n in NAMES]).astype(np.float32,copy=True)


def map_stats(obs):
    return np.stack([np.asarray(obs[f'unit_{s*5}']['graphic']).sum((0,1)) for s in (0,1)]).astype(np.float32)


class Trace:
    def __init__(self,path,cfg):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.cfg=cfg;self.rows=[];self.chunks=[];self.snapshots=[]
        self.events=gzip.open(self.path/'events.jsonl.gz','wt',encoding='utf8');self.count=0;self.event_count=0

    def event(self,row):
        self.events.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.event_count+=1

    def snapshot(self,obs,step):
        if not obs:return
        path=self.path/f'map-{step:06d}.npz'
        np.savez_compressed(path,**{n:np.asarray(x['graphic']) for n,x in obs.items() if n in ('unit_0','unit_5')})
        self.snapshots.append(dict(step=step,path=path.name,sha256=sha(path)))

    def append(self,**row):
        self.rows.append(row);self.count+=1
        if len(self.rows)>=self.cfg['chunk_steps']:self.flush()

    def flush(self):
        if not self.rows:return
        first=int(self.rows[0]['step']);path=self.path/f'steps-{first:06d}.npz'
        tmp=path.with_suffix('.npz.tmp')
        data={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        before,after=data.pop('before'),data.pop('after')
        if np.array_equal(before[1:],after[:-1],equal_nan=True):
            data['observations']=np.concatenate([before[:1],after])
            data['observation_layout']=np.asarray('chain')
        else:
            data['observations']=np.stack([before,after],axis=1)
            data['observation_layout']=np.asarray('pairs')
        # Float bit patterns are XOR-delta encoded, never rounded or quantized.
        bits=np.asarray(data.pop('observations'),dtype='<f4').view('<u4')
        delta=bits.copy();delta[1:]=np.bitwise_xor(bits[1:],bits[:-1])
        data['observations_xor']=delta
        data['observation_encoding']=np.asarray('xor_uint32_le')
        with tmp.open('wb') as f:np.savez_compressed(f,**data)
        os.replace(tmp,path);self.chunks.append(dict(path=path.name,first=first,count=len(self.rows),sha256=sha(path)));self.rows=[]

    def close(self,complete=False):
        self.flush();self.events.close()
        atomic(self.path/'index.json',dict(schema='blackout.league.trace.v2',complete=complete,steps=self.count,
            names=NAMES,chunks=self.chunks,snapshots=self.snapshots,events=self.event_count,
            vector_alignment='before[t] -> action[t] -> reward[t], after[t]; absent terminal after is NaN with after_present=false',
            graphic='two team representative snapshots; channel counts every transition; not full input replay',
            entity_identity='no assumed correspondence between entity slots and public unit IDs'))


class Behavior:
    def __init__(self,cfg):
        self.cfg=cfg;self.steps=0;self.previous_action=None;self.previous_positions=None;self.previous_previous=None
        self.metrics=np.zeros((2,10),np.float64);self.hist=np.zeros((2,9),np.int64);self.occupancy=np.zeros((2,24,24),np.int64)
        self.timeline=[];self.last_leader=0

    def observe(self,step,before,after,actions,cumulative,trace,terminal=False,obs=None):
        self.steps+=1
        for side in (0,1):
            act=actions[side*5:side*5+5];norm=np.linalg.norm(act,axis=1)
            idx=1+(np.rint(np.arctan2(act[:,1],act[:,0])/(np.pi/4)).astype(int)%8);idx[norm<self.cfg['stop_epsilon']]=0
            self.hist[side]+=np.bincount(idx,minlength=9);self.metrics[side,0]+=np.mean(idx==0)
            self.metrics[side,1]+=float(np.allclose(act,act[0],atol=1e-6))
            if self.previous_action is not None:
                prev=self.previous_action[side*5:side*5+5];pn=np.linalg.norm(prev,axis=1)
                cosine=np.sum(prev*act,axis=1)/np.maximum(pn*norm,1e-12)
                self.metrics[side,2]+=np.mean((pn>self.cfg['stop_epsilon'])&(norm>self.cfg['stop_epsilon'])&(cosine<self.cfg['direction_change_cos']))
            entities=before[side*5,:90].reshape(10,9);own=entities[:,2]>0;positions=entities[own,:2]
            if len(positions):
                cells=np.clip((positions*24).astype(int),0,23)
                np.add.at(self.occupancy[side],(cells[:,1],cells[:,0]),1)
                self.metrics[side,3]+=np.mean(np.linalg.norm(positions-positions.mean(0),axis=1))
                if obs is not None:
                    g=obs[f'unit_{side*5}']['graphic'];yy,xx=np.where(g[:,:,2]>0)
                    if len(xx):
                        base=np.array([(xx.mean()+.5)/96,1-(yy.mean()+.5)/96])
                        self.metrics[side,8]+=np.mean(np.linalg.norm(positions-base,axis=1)<self.cfg['near_base_distance'])
            if after is not None and not terminal:
                nxt=after[side*5,:90].reshape(10,9);move=np.linalg.norm(nxt[:,:2]-entities[:,:2],axis=1)
                self.metrics[side,4]+=np.mean(move[own]>self.cfg['movement_epsilon']) if own.any() else 0
                pickup=int(np.count_nonzero(own&(entities[:,3:].argmax(-1)==0)&(nxt[:,3:].argmax(-1)==1)))
                drop=int(np.count_nonzero(own&(entities[:,3:].argmax(-1)==1)&(nxt[:,3:].argmax(-1)==0)))
                delta=float(after[side*5,93]-before[side*5,93]);respawn=int(np.count_nonzero(own&(move>self.cfg['teleport_distance'])))
                self.metrics[side,5]+=pickup;self.metrics[side,6]+=int(delta>0 and drop>0);self.metrics[side,7]+=respawn
                if pickup or drop or delta or respawn:
                    trace.event(dict(step=step,side=side,kind='observation_proxy',pickup=pickup,drop=drop,delivery=int(delta>0 and drop>0),teleport=respawn,visible_score_delta=delta))
                if self.previous_previous is not None:
                    old=self.previous_previous[side*5,:90].reshape(10,9)
                    self.metrics[side,9]+=np.mean((np.linalg.norm(nxt[:,:2]-old[:,:2],axis=1)<1e-4)&(move>1e-4))
        leader=int(np.sign(cumulative[0]-cumulative[1]))
        if leader!=self.last_leader:trace.event(dict(step=step,kind='reward_lead_change',leader=leader));self.last_leader=leader
        if step%100==0 or terminal:self.timeline.append(dict(step=step,time=float(before[0,95]),reward=cumulative.tolist(),visible_score=before[0,93:95].tolist()))
        self.previous_previous=before.copy();self.previous_action=actions.copy()

    def report(self):
        keys=['stop_fraction','same_action_fraction','turn_fraction','team_spread','moving_entity_fraction','pickup_proxy','delivery_proxy','teleport_proxy','near_base_fraction','backtrack_proxy_fraction']
        output=[]
        for s in (0,1):
            row={k:float(self.metrics[s,i]/max(1,self.steps) if i not in (5,6,7) else self.metrics[s,i]) for i,k in enumerate(keys)}
            row.update(action_histogram=self.hist[s].tolist(),occupancy=self.occupancy[s].tolist());output.append(row)
        return dict(sides=output,timeline=self.timeline,event_truth='observation proxies; entity slots not identified as action recipients')


def load_chunk(path):
    """Restore logical before/after fields from both original and compact chunks."""
    with np.load(path,allow_pickle=False) as stored:
        data={k:stored[k] for k in stored.files}
    if 'observation_encoding' in data:
        encoding=str(data.pop('observation_encoding'))
        if encoding!='xor_uint32_le':raise ValueError('unknown float encoding')
        bits=np.bitwise_xor.accumulate(data.pop('observations_xor'),axis=0)
        data['observations']=bits.astype('<u4',copy=False).view('<f4')
    if 'observation_layout' in data:
        layout=str(data.pop('observation_layout'));obs=data.pop('observations')
        if layout=='chain':data.update(before=obs[:-1],after=obs[1:])
        elif layout=='pairs':data.update(before=obs[:,0],after=obs[:,1])
        else:raise ValueError('unknown observation layout')
    return data
