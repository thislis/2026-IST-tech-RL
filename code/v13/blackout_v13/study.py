"""Resumable teacher -> ablations -> finalists -> held-out confirmation pipeline."""
import copy
import os
from pathlib import Path
import random
import shutil
import time
import zlib
import numpy as np
import torch
from .io import ROOT,read_json,atomic_json,atomic_torch,append,digest,file_hash,source_identity,cpu_state
from .policy import Policy
from .critic import Critic
from .ppo import Learner
from .collector import Pool,serialize,deserialize
from .evaluation import MapRegistry,evaluate,schedule,paired_confirmation,seed_cache
from .runtime import hardware_identity,resource_room,tune_workers,tune_learner


def configuration(path=None,name=None,arms=None,scope='full'):
    cfg=read_json(path or ROOT/'code/v13/configs/default.json')
    if name is not None:cfg['name']=name
    if arms:cfg['arms']=arms
    cfg['scope']=scope
    if cfg['gamma']!=1.:raise ValueError('v13 score-potential objective currently requires gamma=1')
    for key in ('chunk_steps','event_period','chunks_per_worker','sequence_minibatch','teacher_steps','bc_epochs',
                'pilot_steps','alternate_every','evaluation_interval','selfplay_interval','selfplay_history',
                'checkpoint_seconds','ppo_epochs','dev_pairs','confirmation_pairs','finalist_count'):
        if cfg[key]<1:raise ValueError('positive value required: '+key)
    if not cfg['arms'] or not cfg['seeds']:raise ValueError('at least one arm and seed required')
    if len(set(cfg['arms']))!=len(cfg['arms']) or len(set(cfg['seeds']))!=len(cfg['seeds']):
        raise ValueError('duplicate arms/seeds would reuse a run')
    for key in ('lags','manager_lags'):
        if key in cfg and (not cfg[key] or any(not isinstance(lag,int) or lag<0 or lag>64 for lag in cfg[key])):
            raise ValueError('history lags must be integers in [0,64]')
    if any(cfg[k]<0 for k in ('worker_steps','manager_steps','alternating_steps')):raise ValueError('negative budget')
    if cfg['temperature']<=0:raise ValueError('positive assignment temperature required')
    if not 1<=cfg.get('intermediate_dev_pairs',cfg['dev_pairs'])<=cfg['dev_pairs']:
        raise ValueError('intermediate_dev_pairs must be within [1,dev_pairs]')
    if cfg['workers']!='auto' and not 1<=int(cfg['workers'])<=cfg['worker_cap']:raise ValueError('invalid worker count')
    ablations=read_json(ROOT/'code/v13/configs/ablations.json')
    if any(a not in ablations for a in cfg['arms']):raise ValueError('unknown ablation')
    from .opponents import catalog
    all_opponents=catalog()
    cfg['evaluation_opponents']=all_opponents
    cfg['training_opponents']=[r for r in all_opponents if r['id'].startswith(('scripted-','v4-','v5-','v6-','v7-','v12-'))]
    return cfg


def preflight(config,full_hash=False):
    from agent_league.new_game_runtime import runtime_spec,executable,verify_api_snapshot,verify_game_snapshot
    from agent_league.common import verify_assets
    runtime=runtime_spec();game=executable()
    for opponent in config['evaluation_opponents']:verify_assets(opponent)
    free=resource_room(config)
    if not free['ok']:raise RuntimeError('Insufficient RAM/GPU/disk headroom: '+str(free))
    result=dict(runtime=runtime['unity_commit'],api=runtime['api_commit'],executable=str(game),
                resource_headroom=free,opponents=len(config['evaluation_opponents']),game_started=False)
    if full_hash:
        result['api_verification']=verify_api_snapshot()
        result['game_verification']=verify_game_snapshot()['sha256']
    return result


def weights(learner,snapshots=None):
    return serialize(dict(version=learner.version,policy=cpu_state(learner.policy),
                          critics={k:cpu_state(v) for k,v in learner.critics.items()},opponent_snapshots=snapshots or []))


def new_learner(config,device):
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    critics={k:Critic(len(config.get('manager_lags',config['lags'])) if k=='manager' else len(config['lags']))
             for k in ('manager','worker')}
    return Learner(Policy(config),critics,config,device)


def run_configuration(config,arm,seed,resources):
    result={**config,**read_json(ROOT/'code/v13/configs/ablations.json')[arm], 'seed':seed}
    result['train_seed_base']=config['train_seed_base']+config['seeds'].index(seed)*3000000
    result['sequence_minibatch']=resources['sequence_minibatch']
    return result


def save_checkpoint(path,learner,progress,config,identity):
    np_state=np.random.get_state()
    payload=dict(schema='blackout.v13.checkpoint.1',identity=identity,config=config,progress=progress,
        policy=cpu_state(learner.policy),critics={k:cpu_state(v) for k,v in learner.critics.items()},
        optimizers={k:v.state_dict() for k,v in learner.optimizers.items()},version=learner.version,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=(np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]),python_rng=random.getstate())
    atomic_torch(path,payload)


def load_checkpoint(path,learner,identity):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['identity']!=identity:raise ValueError('checkpoint config/source/runtime identity changed')
    learner.policy.load_state_dict(payload['policy'])
    for k,v in learner.critics.items():v.load_state_dict(payload['critics'][k])
    for k,v in learner.optimizers.items():v.load_state_dict(payload['optimizers'][k])
    learner.version=payload['version'];torch.set_rng_state(payload['torch_rng'])
    if torch.cuda.is_available() and payload['cuda_rng']:torch.cuda.set_rng_state_all(payload['cuda_rng'])
    n=payload['numpy_rng'];np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))
    random.setstate(payload['python_rng'])
    return payload['progress']


class Study:
    def __init__(self,config,directory):
        self.cfg=config;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.stop_file=self.directory/'stop.request'
        self.artifacts=ROOT/'artifacts/checkpoints/v13'/config['name']
        self.artifacts.mkdir(parents=True,exist_ok=True)
        self.registry=MapRegistry(self.directory/'map_splits.sqlite3')
        self.pool=None

    def stopped(self):return self.stop_file.exists()

    def status(self,phase,**data):
        atomic_json(self.directory/'status.json',dict(pid=os.getpid(),phase=phase,time=time.time(),**data))

    def registered_identity(self):
        from agent_league.new_game_runtime import runtime_spec
        manifest=dict(config=self.cfg,sources=source_identity(),runtime=runtime_spec())
        identity=digest(manifest)
        path=self.directory/'manifest.json'
        if path.exists():
            if read_json(path)['identity']!=identity:raise ValueError('Registered sources/config/assets changed; use a new run name')
        else:atomic_json(path,dict(identity=identity,**manifest))
        self.identity=identity

    def make_pool(self,cfg):
        if self.pool is not None:self.pool.close()
        cfg={**cfg,'worker_memory_gib':self.resources.get('measured_worker_memory_gib',cfg['worker_memory_gib']),
             'worker_gpu_memory_gib':self.resources.get('measured_worker_gpu_memory_gib',cfg['worker_gpu_memory_gib'])}
        self.pool=Pool(cfg,self.directory)
        if not resource_room(cfg,self.resources['selected_workers'])['ok']:
            raise RuntimeError('Insufficient current resources for the registered worker allocation')
        self.pool.resize(self.resources['selected_workers'])
        return self.pool

    def record_chunks(self,chunks,split='train'):
        for chunk in chunks:
            for row in chunk['starts']:
                if not self.registry.add(row['map_sha256'],split):raise ValueError('train/evaluation map overlap; entire new chunk rejected')
            for row in chunk['results']:append(self.directory/'episodes.jsonl',dict(split=split,**row))

    def teacher_data(self):
        if 'teacher_source' in self.cfg:
            source=self.cfg['teacher_source'];directory=(ROOT/source['directory']).resolve()
            if not directory.is_relative_to((ROOT/'artifacts/data/v13').resolve()):
                raise ValueError('teacher continuation path escaped v13 data')
            meta_path=directory/'dataset.json'
            if file_hash(meta_path)!=source['manifest_sha256']:raise ValueError('imported teacher manifest changed')
            meta=read_json(meta_path)
            if meta['identity']!=source['identity'] or meta['steps']<self.cfg['teacher_steps']:
                raise ValueError('imported teacher dataset is incomplete or incompatible')
            return directory,meta
        directory=ROOT/'artifacts/data/v13'/self.cfg['name']/'teacher';directory.mkdir(parents=True,exist_ok=True)
        meta_path=directory/'dataset.json'
        meta=read_json(meta_path) if meta_path.exists() else dict(identity=self.identity,steps=0,files=[],episode_counter=0)
        if meta['identity']!=self.identity:raise ValueError('teacher dataset source/config mismatch')
        if meta['steps']>=self.cfg['teacher_steps']:return directory,meta
        config={**self.cfg,'episode_start':meta['episode_counter']+1000}
        pool=self.make_pool(config);learner=new_learner(config,'cpu');frozen=weights(learner)
        while meta['steps']<self.cfg['teacher_steps']:
            if self.stopped():raise InterruptedError('teacher stage stopped')
            self.status('teacher',steps=meta['steps'],target=self.cfg['teacher_steps'],workers=len(pool.workers))
            chunks=pool.collect(len(pool.workers)*config['chunk_steps'],frozen,'worker',imitation=True,stop=self.stopped)
            self.record_chunks(chunks)
            for chunk in chunks:
                file=f'chunk_{len(meta["files"]):07d}.ptz'
                data=zlib.compress(serialize(chunk),1)
                temporary=directory/(file+'.tmp');temporary.write_bytes(data);os.replace(temporary,directory/file)
                meta['files'].append(dict(name=file,sha256=file_hash(directory/file),steps=chunk['env_steps']))
                meta['steps']+=chunk['env_steps'];meta['episode_counter']=max(meta['episode_counter'],chunk['episode_counter'])
            atomic_json(meta_path,meta)
            if not resource_room(config)['ok']:raise RuntimeError('teacher dataset resource reserve reached')
        pool.close();self.pool=None
        return directory,meta

    def train(self,arm,seed,pilot,dataset):
        tag=f'{"pilot" if pilot else "main"}_{arm}_s{seed}'
        directory=self.directory/'runs'/tag;directory.mkdir(parents=True,exist_ok=True)
        config=run_configuration(self.cfg,arm,seed,self.resources)
        identity=digest(dict(study=self.identity,config=config,tag=tag))
        checkpoint=self.artifacts/tag/'latest.pt';best_path=checkpoint.with_name('best.pt')
        learner=new_learner(config,self.resources['selected_device'])
        progress=dict(bc_phase=0,bc_position=0,stage=0,stage_steps=0,env_steps=0,episode_counter=2000,
                      best_score=-1.,next_eval=config['evaluation_interval'],completed=False,
                      worker_rng={},snapshots=[],next_snapshot=config['selfplay_interval'])
        if checkpoint.exists():progress=load_checkpoint(checkpoint,learner,identity)
        if progress['completed']:return read_json(directory/'result.json')
        data_dir,meta=dataset
        bc_phases=['manager'] if config.get('worker_mode')=='planner' else ['manager','worker']
        last_checkpoint=time.monotonic()
        for phase_index in range(progress['bc_phase'],len(bc_phases)):
            phase=bc_phases[phase_index]
            total=len(meta['files'])*config['bc_epochs'];position=progress['bc_position']
            while position<total:
                if self.stopped():
                    save_checkpoint(checkpoint,learner,progress,config,identity)
                    raise InterruptedError('BC stopped')
                chunks=[]
                for item in meta['files'][position%len(meta['files']):min(len(meta['files']),position%len(meta['files'])+config['sequence_minibatch'])]:
                    path=data_dir/item['name']
                    if file_hash(path)!=item['sha256']:raise ValueError('teacher data checksum changed')
                    chunk=deserialize(zlib.decompress(path.read_bytes()));chunk['phase']=phase;chunk['version']=learner.version
                    chunks.append(chunk)
                metrics=learner.update(chunks,phase,imitation=True);position+=len(chunks)
                progress.update(bc_phase=phase_index,bc_position=position)
                append(directory/'updates.jsonl',dict(stage='bc_'+phase,position=position,**metrics))
                if time.monotonic()-last_checkpoint>=config['checkpoint_seconds']:
                    save_checkpoint(checkpoint,learner,progress,config,identity);last_checkpoint=time.monotonic()
                self.status('bc',run=tag,actor=phase,position=position,target=total)
            progress.update(bc_phase=phase_index+1,bc_position=0)
            save_checkpoint(checkpoint,learner,progress,config,identity)
        if pilot:
            stages=[('manager',config['pilot_steps'])] if config.get('worker_mode')=='planner' else [
                ('worker',config['pilot_steps']//2),('manager',config['pilot_steps']//2)]
        else:
            stages=[] if config.get('worker_mode')=='planner' else [('worker',config['worker_steps'])]
            stages.append(('manager',config['manager_steps']))
            remainder=config['alternating_steps'];index=0
            while remainder:
                amount=min(remainder,config['alternate_every'])
                phase='manager' if config.get('worker_mode')=='planner' or index%2 else 'worker'
                stages.append((phase,amount));remainder-=amount;index+=1
        config['episode_start']=progress['episode_counter']+1
        config['worker_rng_states']=progress['worker_rng']
        pool=self.make_pool(config)
        for stage_index in range(progress['stage'],len(stages)):
            phase,target=stages[stage_index]
            while progress['stage_steps']<target:
                if self.stopped():raise InterruptedError('training stopped')
                start=time.perf_counter();frozen=weights(learner,progress['snapshots'])
                count=len(pool.workers)*config['chunk_steps']*config['chunks_per_worker']
                chunks=pool.collect(min(count,target-progress['stage_steps']),frozen,phase,stop=self.stopped)
                self.record_chunks(chunks);steps=sum(c['env_steps'] for c in chunks)
                collection=time.perf_counter()-start
                metrics=learner.update(chunks,phase)
                progress['stage_steps']+=steps;progress['env_steps']+=steps
                progress['episode_counter']=max(progress['episode_counter'],max(c['episode_counter'] for c in chunks))
                for c in chunks:progress['worker_rng'][str(c['slot'])]=c['rng']
                save_checkpoint(checkpoint,learner,progress,config,identity)
                if progress['env_steps']>=progress['next_snapshot']:
                    snapshot=checkpoint.parent/'selfplay'/f'version_{learner.version:07d}.pt'
                    snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(checkpoint,snapshot)
                    progress['snapshots']=(progress['snapshots']+[dict(kind='v13',id=f'v13-{tag}-{learner.version}',
                        path=str(snapshot),sha256=file_hash(snapshot))])[-config['selfplay_history']:]
                    progress['next_snapshot']+=config['selfplay_interval']
                    save_checkpoint(checkpoint,learner,progress,config,identity)
                stats=dict(run=tag,stage=phase,env_steps=progress['env_steps'],collection_seconds=collection,
                           env_steps_per_second=steps/collection,**metrics)
                append(directory/'updates.jsonl',stats);self.status('training',**stats)
                if not pilot and progress['env_steps']>=progress['next_eval']:
                    self.dev_evaluation(pool,learner,config,directory,progress,checkpoint,best_path)
                    progress['next_eval']+=config['evaluation_interval']
                    save_checkpoint(checkpoint,learner,progress,config,identity)
                if not resource_room(config)['ok']:raise RuntimeError('runtime resource reserve reached')
            progress.update(stage=stage_index+1,stage_steps=0)
            save_checkpoint(checkpoint,learner,progress,config,identity)
        summary=self.dev_evaluation(pool,learner,config,directory,progress,checkpoint,best_path,screen=not pilot)
        if not pilot and 'intermediate_dev_pairs' in config:
            summary=self.final_dev_evaluation(pool,config,directory,best_path)
        progress['completed']=True
        score=summary['macro_score_rate'] if not pilot and 'intermediate_dev_pairs' in config else progress['best_score']
        result=dict(run=tag,arm=arm,seed=seed,pilot=pilot,best_score=score,
                    checkpoint=str(best_path),env_steps=progress['env_steps'],last_dev=summary)
        if not pilot and 'intermediate_dev_pairs' in config:result['screening_score']=progress['best_score']
        atomic_json(directory/'result.json',result);save_checkpoint(checkpoint,learner,progress,config,identity)
        pool.close();self.pool=None
        return result

    def dev_evaluation(self,pool,learner,config,directory,progress,checkpoint,best_path,screen=True):
        reduced=screen and 'intermediate_dev_pairs' in config
        pairs=config['intermediate_dev_pairs'] if reduced else config['dev_pairs']
        self.status('dev_evaluation',run=directory.name,version=learner.version,screening=reduced,
                    expected=len(config['evaluation_opponents'])*pairs*2)
        tasks=schedule(config['evaluation_opponents'],pairs,config['dev_seed_base'])
        root=directory/'dev'/str(learner.version)
        destination=root/'screen' if reduced else root
        report,_=evaluate(pool,tasks,weights(learner),destination,self.registry,'dev',self.stopped)
        if report['macro_score_rate']>progress['best_score']:
            progress['best_score']=report['macro_score_rate'];best_path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(checkpoint,best_path)
        pool.reset()
        return report

    def final_dev_evaluation(self,pool,config,directory,best_path):
        # A common 88-game set chooses checkpoints within a run. Only the full
        # 220-game set chooses among runs; never compare their raw rates directly.
        payload=torch.load(best_path,map_location='cpu',weights_only=True)
        frozen=serialize(dict(version=payload['version'],policy=payload['policy'],critics=payload['critics']))
        root=directory/'dev'/str(payload['version'])
        tasks=schedule(config['evaluation_opponents'],config['dev_pairs'],config['dev_seed_base'])
        self.status('final_dev_evaluation',run=directory.name,version=payload['version'],expected=len(tasks))
        seed_cache(tasks,[root/'screen',root/'imported_full'],root/'full')
        report,_=evaluate(pool,tasks,frozen,root/'full',self.registry,'dev',self.stopped)
        return report

    def confirmation(self,selected):
        from .export import export_policy
        payload=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
        cfg=payload['config'];learner=new_learner(cfg,'cpu')
        learner.policy.load_state_dict(payload['policy'])
        for k in learner.critics:learner.critics[k].load_state_dict(payload['critics'][k])
        learner.version=payload['version']
        frozen=weights(learner);pool=self.make_pool(cfg)
        opponents=[o for o in cfg['evaluation_opponents'] if o['id']!='v12-before-learned']
        base=cfg['confirmation_seed_base'];pairs=cfg['confirmation_pairs']
        candidate=schedule(opponents,pairs,base,prefix='confirmation')
        baseline=schedule(opponents,pairs,base,baseline=True,prefix='confirmation')
        v12=[o for o in cfg['evaluation_opponents'] if o['id']=='v12-before-learned']
        direct=schedule(v12,pairs,base+100000,prefix='direct')
        self.status('confirmation',selected=selected['run'],expected=len(candidate+baseline+direct))
        _,rows=evaluate(pool,candidate+baseline+direct,frozen,self.directory/'confirmation',self.registry,'confirmation',self.stopped)
        b=[r for r in rows if r['baseline']];d=[r for r in rows if r['opponent']=='v12-before-learned']
        c=[r for r in rows if not r['baseline'] and r['opponent']!='v12-before-learned']
        report=paired_confirmation(c,b,d);report['selected']=selected
        destination=ROOT/'artifacts/submission/v13'/self.cfg['name']
        report['export']=export_policy(learner.policy,destination)
        report['official_container_verified']=False
        report['adoption']='candidate; official container limits and regression review required'
        atomic_json(self.directory/'result.json',report)
        pool.close();self.pool=None
        return report

    def run(self):
        from .export import export_policy
        try:
            self.status('preflight');preflight(self.cfg,full_hash=True);self.registered_identity()
            torch.set_num_threads(1)
            tuning=self.directory/'resources';tuning.mkdir(exist_ok=True)
            resources=tuning/'selected.json'
            if resources.exists():self.resources=read_json(resources)
            else:
                self.status('export_gate')
                policy=Policy(self.cfg).eval()
                export=export_policy(policy,ROOT/'artifacts/submission/v13'/self.cfg['name']/'preflight')
                # CPU synthetic measurements are a pre-training gate, not official GPU certification.
                warm=export['synthetic_call_seconds'][1:]
                if max(warm)*1000>self.cfg['submission_latency_ms']:
                    raise RuntimeError('Full ONNX synthetic latency exceeds the registered gate; inspect preflight/verification.json')
                atomic_json(tuning/'hardware.json',hardware_identity())
                self.status('learner_tuning');compute=tune_learner(policy,self.cfg,tuning)
                if torch.cuda.is_available():torch.cuda.empty_cache()
                self.status('worker_tuning')
                pool=Pool({**self.cfg,'train_seed_base':20000000},self.directory);self.pool=pool
                learner=new_learner(self.cfg,'cpu')
                workers=tune_workers(pool,weights(learner),self.cfg,tuning,self.stopped)
                pool.close();self.pool=None
                self.resources={**compute,**workers};atomic_json(resources,self.resources)
            self.cfg['sequence_minibatch']=self.resources['sequence_minibatch']
            dataset=self.teacher_data()
            pilots=[self.train(arm,self.cfg['seeds'][0],True,dataset) for arm in self.cfg['arms']]
            atomic_json(self.directory/'pilot_results.json',pilots)
            if self.cfg['scope']=='pilot':
                self.status('completed_pilot',results=pilots);return
            chosen=sorted(pilots,key=lambda r:(-r['best_score'],r['arm']))[:self.cfg['finalist_count']]
            results=[self.train(r['arm'],seed,False,dataset) for r in chosen for seed in self.cfg['seeds']]
            atomic_json(self.directory/'main_results.json',results)
            selected=max(results,key=lambda r:(r['best_score'],-r['seed']))
            result=self.confirmation(selected);self.status('completed',result=result)
        except InterruptedError as exc:
            self.status('stopped',reason=str(exc),resume='same launcher command resumes committed checkpoints')
        except BaseException as exc:
            import traceback
            self.status('failed',error=str(exc),traceback=traceback.format_exc());raise
        finally:
            if self.pool is not None:self.pool.close()
            self.registry.close()
