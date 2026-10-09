"""Immutable policy locks and complete attempt records, including failed cells."""
from pathlib import Path
import time
from .contracts import ContractError,digest
from .checkpoints import atomic_json,fingerprints
from .telemetry import append,RequestTimer
from .opponents import make_opponent
from blackout_rl.batching import team_agents

class EvaluationLedger:
    def __init__(self,directory,lock):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=dict(lock);self.sha=digest(lock)
        p=self.directory/'model_lock.json'
        if p.exists():raise FileExistsError('evaluation attempt directory already used; never salvage successful test cells')
        atomic_json(p,dict(sha256=self.sha,lock=lock));self.results=[]
    def start(self,cell):
        append(self.directory/'attempts.jsonl',dict(event='start',lock_sha256=self.sha,**cell))
    def finish(self,cell,**result):
        record=dict(**cell,**result,lock_sha256=self.sha)
        self.results.append(record);append(self.directory/'attempts.jsonl',dict(event='finish',**record))
    def complete(self,expected_cells):
        if len(self.results)!=expected_cells or any(r['status']!='valid' for r in self.results):
            raise ContractError('incomplete/failed evaluation; no optimistic valid-only win rate')
        wins=sum(r['winner_team']==r['side'] for r in self.results);draws=sum(r['winner_team']==-1 for r in self.results)
        result=dict(primary='engine_W_over_N',n=expected_cells,wins=wins,draws=draws,win_rate=wins/expected_cells,
                    auxiliary_utility=(wins+.5*draws)/expected_cells,lock_sha256=self.sha,episodes=self.results)
        atomic_json(self.directory/'result.json',result);return result


def evaluate(env,policy,config,checkpoint_hash,directory,*,maps,split='dev'):
    if split!='dev':raise ContractError('confirmation/test require independently registered manifest and selection lock; not implicitly enabled')
    if not maps or len(set(maps))!=len(maps) or not set(maps)<=set(config['evaluation']['dev_maps']):raise ContractError('unregistered dev maps')
    lock=dict(checkpoint_sha256=checkpoint_hash,config=config,sources=fingerprints(),maps=list(maps),split=split,
              metric='engine_W_over_N',invalid_handling='abort_preserve_attempt',rng_seed=config['evaluation']['seed'],executor=config['policy']['executor'])
    ledger=EvaluationLedger(directory,lock);timer=RequestTimer();start=time.monotonic()
    for seed in maps:
        for side in (0,1):
            cell=dict(map_seed=seed,side=side,training_run=config['experiment_id'],opponent='target',eval_rep=0)
            ledger.start(cell)
            try:
                obs,_=env.reset(seed=seed);policy.reset(env.context(side));policy.generator.manual_seed(config['evaluation']['seed'])
                opponent,opponent_hash=make_opponent('target',1-side,config['opponent'])
                for step in range(config['environment']['max_episode_steps']):
                    result=timer.measure(policy.decide,obs,env.context(side));actions=dict(result['actions'])
                    actions.update(opponent.act(obs,team_agents(1-side)))
                    obs,terminated,truncated,info=env.step(actions)
                    if truncated:raise ContractError('truncated evaluation is invalid, not a loss/draw fixture')
                    if terminated:
                        event=info['outcome']
                        ledger.finish(cell,status='valid',winner_team=event.winner_team,score_points=event.final_score_points,
                                      outcome=event.to_dict(),steps=step+1,opponent_hash=opponent_hash)
                        break
                else:raise ContractError('evaluation watchdog')
            except BaseException as exc:
                ledger.finish(cell,status='invalid',error=type(exc).__name__,detail=str(exc));raise
    report=ledger.complete(len(maps)*2)
    atomic_json(Path(directory)/'timing.json',dict(wall_seconds=time.monotonic()-start,request=timer.report()))
    return report
