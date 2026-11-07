"""Persistent environments, fixed-length chunks and immediate episode refill.

One pipe per worker bounds in-flight memory. No full images or frame stacks
cross IPC. Policy snapshots are serialized once per learner generation.
"""
import io
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import random
import time
import traceback
import hashlib
import numpy as np
import torch
from .io import append


def serialize(payload):
    buf=io.BytesIO();torch.save(payload,buf);return buf.getvalue()


def deserialize(data):
    return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)


class Session:
    def __init__(self,config,slot,env_factory=None,opponent_factory=None,teacher_factory=None):
        from .policy import Policy
        from .critic import Critic
        from .actor import Actor
        from .env import make_env
        self.cfg=config;self.slot=slot;self.episode=int(config.get('episode_start',0));self.steps=0
        self.env=(env_factory or (lambda:make_env(config,slot)))()
        self.opponent_factory=opponent_factory;self.teacher_factory=teacher_factory
        self.teacher=None;self.opponent=None;self.obs=None;self.phase=None;self.version=-1
        p=Policy(config)
        if config.get('script_geometry',True):p.tasks=torch.jit.script(p.tasks)
        self.actor=Actor(p,Critic(len(config['lags'])),config)
        self.pending_starts=[];self.pending_results=[]

    def load(self,payload,phase):
        from .critic import Critic
        if self.version!=payload['version'] or self.phase!=phase:
            self.actor.policy.load_state_dict(payload['policy'])
            count=len(self.cfg.get('manager_lags',self.cfg['lags'])) if phase=='manager' else len(self.cfg['lags'])
            with torch.random.fork_rng(devices=[]):self.actor.critic=Critic(count).eval()
            self.actor.critic.load_state_dict(payload['critics'][phase])
            self.actor.refresh_hidden()
            self.version=payload['version'];self.phase=phase
        self.cfg['training_opponents']=[r for r in self.cfg['training_opponents'] if r.get('kind')!='v13']+payload.get('opponent_snapshots',[])

    def start_episode(self):
        from .opponents import make_opponent
        self.episode+=1
        seed=int(self.cfg['train_seed_base'])+self.slot*100000+self.episode
        self.side=(self.episode+self.slot)%2
        opponent=self.cfg['training_opponents'][(self.episode+self.slot)%len(self.cfg['training_opponents'])]
        self.obs,_=self.env.reset(seed=seed)
        self.actor.reset();self.steps=0;self.started=time.monotonic()
        if self.teacher is not None:self.teacher.reset()
        self.opponent=(self.opponent_factory or make_opponent)(opponent,1-self.side,seed+17,str(seed))
        graphic=np.asarray(self.obs['unit_0']['graphic'])
        if graphic.shape==(11,96,96):graphic=graphic.transpose(1,2,0)
        map_hash=hashlib.sha256(np.ascontiguousarray(graphic[..., [1,2,3,6,7,8,9,10]]).tobytes()).hexdigest()
        self.episode_info=dict(seed=seed,side=self.side,opponent=opponent['id'],map_sha256=map_hash,slot=self.slot)
        self.pending_starts.append(self.episode_info.copy())

    def collect(self,count,imitation=False,heartbeat=lambda **k:None,stop=lambda:False):
        from .env import pack,outcome
        from .opponents import Teacher
        if imitation and self.teacher is None:self.teacher=(self.teacher_factory or Teacher)()
        if self.obs is None:self.start_episode()
        snapshot=self.actor.prefix_snapshot();rows=[];timing=dict(policy=0.,opponent=0.,environment=0.)
        started=time.perf_counter()
        for _ in range(count):
            if stop():raise InterruptedError('stop requested during collection')
            v,g,names=pack(self.obs,self.side)
            potential=float(np.tanh(v[0,93]-v[0,94]))
            before=time.perf_counter()
            ta,tt=self.teacher.act(v,g) if imitation else (None,None)
            action,row=self.actor.step(v,g,phase=self.phase,teacher_action=ta,teacher_task=tt)
            timing['policy']+=time.perf_counter()-before
            actions=dict(zip(names,action));before=time.perf_counter()
            actions.update(self.opponent.act(self.obs,self.steps));timing['opponent']+=time.perf_counter()-before
            before=time.perf_counter();nxt,rewards,terms,truncs,infos=self.env.step(actions)
            timing['environment']+=time.perf_counter()-before
            if len(terms)!=10 or any(truncs.values()) or len(set(bool(v) for v in terms.values()))!=1:
                raise ValueError('partial/truncated transition; chunk discarded')
            done=all(terms.values());self.steps+=1
            if done:
                result,scores=outcome(infos,self.side)
                row['reward']=result-self.cfg['potential_beta']*potential
                self.pending_results.append(dict(**self.episode_info,outcome=result,scores=scores,steps=self.steps,
                                                 diagnostics=self.actor.diagnostics.report()))
            else:
                nv,_,_=pack(nxt,self.side)
                if nv[0,95]>v[0,95]+1e-4:raise ValueError('clock reset without natural terminal')
                next_potential=float(np.tanh(nv[0,93]-nv[0,94]))
                row['reward']=self.cfg['potential_beta']*(self.cfg['gamma']*next_potential-potential)
            row['done']=done;rows.append(row);self.obs=nxt
            if self.steps>=self.cfg['max_episode_steps'] and not done:raise TimeoutError('episode step watchdog')
            if time.monotonic()-self.started>self.cfg['episode_wall_seconds']:raise TimeoutError('episode wall watchdog')
            if done:self.start_episode()  # No barrier with other workers, even inside one chunk.
            if len(rows)%32==0:heartbeat(steps=len(rows),episode=self.episode)
        if rows[-1]['done']:bootstrap=0.
        else:
            nv,ng,_=pack(self.obs,self.side);bootstrap=self.actor.bootstrap(nv,ng)
        result=dict(**snapshot,rows=rows,bootstrap=bootstrap,version=self.version,phase=self.phase,
            starts=self.pending_starts,results=self.pending_results,env_steps=len(rows),slot=self.slot,
            episode_counter=self.episode,timing=timing,wall_seconds=time.perf_counter()-started)
        ns=np.random.get_state()
        result['rng']=dict(torch=torch.get_rng_state(),numpy=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),python=random.getstate())
        self.pending_starts=[];self.pending_results=[]
        return result

    def close(self):self.env.close()

    def evaluate(self,task,stop=lambda:False,heartbeat=lambda **k:None):
        from .env import pack,outcome
        from .opponents import make_opponent,catalog
        obs,_=self.env.reset(seed=task['seed']);self.actor.reset();self.obs=None
        side=task['side'];opponent=make_opponent(task['opponent'],1-side,task['seed']+17,str(task['id']))
        baseline=make_opponent(next(r for r in catalog() if r['id']=='v12-before-learned'),
                               side,task['seed']+31,str(task['id'])) if task.get('baseline') else None
        graphic=np.asarray(obs['unit_0']['graphic'])
        if graphic.shape==(11,96,96):graphic=graphic.transpose(1,2,0)
        map_hash=hashlib.sha256(np.ascontiguousarray(graphic[..., [1,2,3,6,7,8,9,10]]).tobytes()).hexdigest()
        started=time.monotonic();latencies=[];stops=0;switches=0
        for step in range(self.cfg['max_episode_steps']):
            if stop():raise InterruptedError('evaluation stopped')
            vector,graphic,names=pack(obs,side)
            before=time.perf_counter()
            if baseline is not None:actions=baseline.act(obs,step)
            else:
                action,row=self.actor.step(vector,graphic,phase='worker',deterministic=True)
                actions=dict(zip(names,action));switches+=int(row['manager'] is not None)
            latencies.append(time.perf_counter()-before)
            stops+=sum(np.linalg.norm(actions[n])<1e-7 for n in names)
            actions.update(opponent.act(obs,step))
            obs,_,terms,truncs,infos=self.env.step(actions)
            if len(terms)!=10 or any(truncs.values()) or len(set(terms.values()))!=1:
                raise ValueError('invalid evaluation transition')
            if all(terms.values()):
                result,scores=outcome(infos,side)
                return dict(id=task['id'],seed=task['seed'],side=side,opponent=task['opponent']['id'],
                    baseline=bool(task.get('baseline')),outcome=result,score_rate=(result+1)/2,
                    scores=scores,steps=step+1,wall_seconds=time.monotonic()-started,map_sha256=map_hash,
                    policy_p99_seconds=float(np.quantile(latencies,.99)),policy_max_seconds=max(latencies),
                    stop_fraction=stops/(5*(step+1)),decision_events=switches,natural_terminal=True,
                    diagnostics=self.actor.diagnostics.report() if baseline is None else None)
            if step%32==0:heartbeat(steps=step)
            if time.monotonic()-started>self.cfg['episode_wall_seconds']:raise TimeoutError('evaluation wall watchdog')
        raise TimeoutError('evaluation step watchdog')


def worker_main(connection,config,slot,stop_event):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed=int(config['seed'])+slot*101
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    from agent_league.new_game_runtime import configure_runtime
    from agent_league.new_game_worker import _ManagedSubprocess
    from blackout_v12.runtime import process_tree_guard
    configure_runtime()
    from mlagents_envs import env_utils
    session=None
    try:
        with process_tree_guard():
            env_utils.subprocess=_ManagedSubprocess()
            session=Session(config,slot)
            rng=config.get('worker_rng_states',{}).get(str(slot))
            if rng:
                torch.set_rng_state(rng['torch']);n=rng['numpy']
                np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]));random.setstate(rng['python'])
            connection.send_bytes(serialize(dict(type='ready')))
            while not stop_event.is_set():
                command=deserialize(connection.recv_bytes())
                if command['type']=='close':break
                if command['type']=='reset':
                    session.obs=None;session.actor.reset();connection.send_bytes(serialize(dict(type='reset')));continue
                session.load(deserialize(command['weights']),command['phase'])
                def heartbeat(**data):connection.send_bytes(serialize(dict(type='heartbeat',**data)))
                if command['type']=='evaluate':
                    result=session.evaluate(command['task'],stop_event.is_set,heartbeat)
                    connection.send_bytes(serialize(dict(type='evaluation',result=result)))
                else:
                    chunk=session.collect(command['count'],command['imitation'],heartbeat,stop_event.is_set)
                    connection.send_bytes(serialize(dict(type='chunk',chunk=chunk)))
    except BaseException as exc:
        try:connection.send_bytes(serialize(dict(type='error',error=str(exc),traceback=traceback.format_exc())))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        if session is not None:
            try:session.close()
            except Exception:pass
        connection.close()


class Pool:
    def __init__(self,config,directory):
        self.cfg=config;self.directory=directory;self.context=mp.get_context('spawn')
        self.stop_event=self.context.Event();self.workers=[];self.pending={};self.stopped=False

    def resize(self,count):
        if self.pending:raise RuntimeError('resize only at chunk boundary')
        while len(self.workers)>count:
            process,connection=self.workers.pop()
            connection.send_bytes(serialize(dict(type='close')));process.join(10)
            if process.is_alive():process.terminate();process.join(5)
            connection.close()
        while len(self.workers)<count:
            slot=len(self.workers)
            parent,child=self.context.Pipe()
            process=self.context.Process(target=worker_main,args=(child,self.cfg,slot,self.stop_event),daemon=False)
            process.start();child.close();self.workers.append((process,parent))
            if not parent.poll(self.cfg['worker_start_seconds']):raise TimeoutError('worker startup watchdog')
            message=deserialize(parent.recv_bytes())
            if message['type']!='ready':raise RuntimeError(message)

    def collect(self,steps,weights,phase,imitation=False,stop=lambda:False):
        chunks=[];received=0;pending_steps=0;size=self.cfg['chunk_steps']
        def dispatch(index):
            nonlocal pending_steps
            connection=self.workers[index][1]
            connection.send_bytes(serialize(dict(type='collect',weights=weights,phase=phase,imitation=imitation,count=size)))
            self.pending[connection]=(index,time.monotonic());pending_steps+=size
        for j in range(len(self.workers)):
            if pending_steps<steps:dispatch(j)
        try:
            while self.pending:
                if stop():raise InterruptedError('supervisor stop requested')
                for process,connection in self.workers:
                    if connection in self.pending and not process.is_alive():raise RuntimeError(f'worker exited: {process.exitcode}')
                ready=wait(list(self.pending),timeout=1.)
                for connection in ready:
                    msg=deserialize(connection.recv_bytes());index,_=self.pending[connection]
                    if msg['type']=='heartbeat':self.pending[connection]=(index,time.monotonic());continue
                    del self.pending[connection];pending_steps-=size
                    if msg['type']=='error':raise RuntimeError(msg['traceback'])
                    if msg['type']!='chunk':raise RuntimeError('unexpected worker message')
                    chunk=msg['chunk'];chunks.append(chunk);received+=chunk['env_steps']
                    if received+pending_steps<steps:dispatch(index)
                if any(time.monotonic()-started>self.cfg['worker_heartbeat_seconds'] for _,started in self.pending.values()):
                    raise TimeoutError('worker heartbeat watchdog')
        except BaseException:
            self.close();raise
        return chunks

    def reset(self):
        if self.pending:raise RuntimeError('reset only at chunk boundary')
        for _,connection in self.workers:connection.send_bytes(serialize(dict(type='reset')))
        for _,connection in self.workers:
            if not connection.poll(30):raise TimeoutError('reset acknowledgement timeout')
            if deserialize(connection.recv_bytes())['type']!='reset':raise RuntimeError('reset failed')

    def evaluate(self,tasks,weights,on_result,stop=lambda:False):
        queue=iter(tasks);results=[]
        def dispatch(index):
            try:task=next(queue)
            except StopIteration:return
            connection=self.workers[index][1]
            connection.send_bytes(serialize(dict(type='evaluate',task=task,weights=weights,phase='worker')))
            self.pending[connection]=(index,time.monotonic())
        for j in range(len(self.workers)):dispatch(j)
        try:
            while self.pending:
                if stop():raise InterruptedError('evaluation stopped')
                for connection in wait(list(self.pending),timeout=1):
                    msg=deserialize(connection.recv_bytes());index,_=self.pending[connection]
                    if msg['type']=='heartbeat':self.pending[connection]=(index,time.monotonic());continue
                    del self.pending[connection]
                    if msg['type']=='error':raise RuntimeError(msg['traceback'])
                    if msg['type']!='evaluation':raise RuntimeError('unexpected evaluation message')
                    on_result(msg['result']);results.append(msg['result']);dispatch(index)
                if any(not p.is_alive() for p,c in self.workers if c in self.pending):raise RuntimeError('evaluation worker exited')
                if any(time.monotonic()-t>self.cfg['worker_heartbeat_seconds'] for i,t in self.pending.values()):
                    raise TimeoutError('evaluation heartbeat watchdog')
        except BaseException:
            self.close();raise
        return results

    def close(self):
        self.stop_event.set()
        for process,connection in self.workers:
            if process.is_alive():
                try:connection.send_bytes(serialize(dict(type='close')))
                except OSError:pass
        for process,connection in self.workers:
            process.join(3)
            if process.is_alive():process.terminate();process.join(5)
            connection.close()
        self.workers=[];self.pending={}
