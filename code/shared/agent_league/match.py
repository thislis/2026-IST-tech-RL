import hashlib
import time
import numpy as np
from .trace import Trace,Behavior,NAMES,vectors,map_stats


def play(task,directory,env_factory,adapter_factory=None,heartbeat=lambda **k:None,stop=lambda:False):
    from .adapters import Adapter
    from blackout_v10.trajectory import pack
    adapter_factory=adapter_factory or Adapter
    cfg=task['config'];start=time.monotonic();env=None;trace=Trace(directory/'trace',cfg);behavior=Behavior(cfg)
    totals={n:0. for n in NAMES};complete=False;steps=0;last_heartbeat=0.;previous_time=None
    policies=[];initial=None;window_audits=[];map_hash=None;official_winner=None;official_available=False
    try:
        for i,s in enumerate(task['models']):
            heartbeat(phase='loading_model',steps=0)
            policies.append(adapter_factory(s,i^task['side'],task['action_seeds'][i],task['id'],device=cfg['device']))
        heartbeat(phase='environment_start',steps=0);env=env_factory()
        obs,_=env.reset(seed=task['environment_seed'])
        h=hashlib.sha256()
        for n in NAMES:
            h.update(np.asarray(obs[n]['vector']).tobytes());h.update(np.asarray(obs[n]['graphic']).tobytes())
        initial=h.hexdigest()
        # Fixed physical Blue view excludes agent-position channels so the two
        # sides of a paired match can audit the actual generated terrain/items.
        graphic=np.asarray(obs['unit_0']['graphic'],dtype=np.float32)
        map_hash=hashlib.sha256(np.ascontiguousarray(graphic[..., [1,2,3,6,7,8,9,10]]).tobytes()).hexdigest()
        trace.snapshot(obs,0);game_start=time.monotonic()
        for step in range(cfg['max_steps']):
            if stop():raise InterruptedError('match interrupted')
            if set(env.agents)!=set(NAMES) or set(obs)!=set(NAMES):raise ValueError('partial active observations')
            for side in (0,1):pack(obs,NAMES[side*5:side*5+5])
            now=time.monotonic()
            if now-last_heartbeat>=2:
                heartbeat(phase='match',steps=step,game_seconds=now-game_start);last_heartbeat=now
                if task.get('audit_windows',True) and len(window_audits)<3:
                    from blackout_v10.windows import visible_windows
                    audit=visible_windows();window_audits.append(audit)
                    if not audit.get('verified') or audit.get('onscreen_owned_windows'):raise RuntimeError('hidden-window audit failed')
            before=vectors(obs);clock=float(before[0,95])
            if previous_time is not None and clock>previous_time+1e-3:raise ValueError('time reset without terminal')
            previous_time=clock
            actions={};latencies=np.zeros(2,np.float64)
            for p in policies:
                t=time.perf_counter();action=p.act(obs,step);latencies[p.team]=time.perf_counter()-t
                if set(action)!=set(p.names):raise ValueError('adapter returned wrong team')
                actions.update(action)
                if p.metadata:trace.event(dict(step=step,side=p.team,kind='policy_decision',data=p.metadata))
            for a in actions.values():
                if a.shape!=(2,) or a.dtype!=np.float32 or not np.isfinite(a).all() or (np.abs(a)>1).any():raise ValueError('invalid action')
            arr=np.stack([actions[n] for n in NAMES]);counts=map_stats(obs)
            next_obs,rewards,terms,truncs,infos=env.step(actions)
            if any(set(x)!=set(NAMES) for x in (rewards,terms,truncs)):raise ValueError('partial transition')
            reward=np.array([rewards[n] for n in NAMES],np.float64)
            if not np.isfinite(reward).all():raise ValueError('nonfinite reward')
            for n in NAMES:totals[n]+=float(rewards[n])
            done=all(terms.values());valid_transition=not any(truncs.values()) and len(set(bool(x) for x in terms.values()))==1 and done==(not env.agents)
            after_present=set(next_obs)==set(NAMES)
            after=vectors(next_obs) if after_present else None
            if after_present:
                for side in (0,1):pack(next_obs,NAMES[side*5:side*5+5])
            cumulative=np.array([sum(totals[n] for n in set(NAMES[s*5:s*5+5])) for s in (0,1)])
            changed=np.array([np.count_nonzero(obs[f'unit_{s*5}']['graphic']!=next_obs[f'unit_{s*5}']['graphic']) if after_present else -1 for s in (0,1)])
            trace.append(step=step,wall_time=time.time(),game_time=clock,before=before,actions=arr,rewards=reward,
                after=after if after_present else np.full((10,96),np.nan,np.float32),after_present=after_present,
                terminated=np.array([terms[n] for n in NAMES]),truncated=np.array([truncs[n] for n in NAMES]),
                cumulative_reward=cumulative,inference_seconds=latencies,map_channel_counts=counts,graphic_changed_values=changed,
                transition_valid=valid_transition)
            steps=step+1
            behavior.observe(step,before,after,arr,cumulative,trace,terminal=done,obs=obs)
            if done or steps%cfg['snapshot_every']==0:trace.snapshot(next_obs,steps) if after_present else trace.snapshot(obs,step)
            if not valid_transition:raise ValueError('truncated/partial terminal; no result awarded')
            if done:
                complete=True
                values=[infos.get(n,{}).get('winner') for n in NAMES]
                official_available=all(w in (-1,0,1) for w in values) and len(set(values))==1
                if official_available:official_winner=int(values[0])
                break
            if task.get('probe_seconds') and time.monotonic()-game_start>=task['probe_seconds']:break
            obs=next_obs
        if not complete and not task.get('probe_seconds'):raise RuntimeError('step watchdog; no natural terminal')
        sides=[sum(totals[n] for n in set(NAMES[s*5:s*5+5])) for s in (0,1)]
        winner=0 if sides[0]>sides[1] else 1 if sides[1]>sides[0] else None
        return dict(valid=complete or bool(task.get('probe_seconds')),natural_terminal=complete,
                    probe=bool(task.get('probe_seconds')),episode_steps=steps,side_rewards=sides,winner_side=winner if complete else None,
                    winner=None if winner is None or not complete else task['a'] if winner==task['side'] else task['b'],
                    wall_seconds=time.monotonic()-start,game_seconds=time.monotonic()-game_start,
                    objective='provided_runner_v1',requested_environment_seed=task['environment_seed'],map_seed_verified=False,
                    initial_observation_sha256=initial,initial_map_sha256=map_hash,
                    official_winner_available=official_available,official_winner_side=official_winner,
                    official_winner=(None if official_winner in (None,-1) else task['a'] if official_winner==task['side'] else task['b']),
                    winner_criteria_agree=(None if not official_available else (winner if winner is not None else -1)==official_winner),
                    window_audits=window_audits,behavior=behavior.report())
    finally:
        try:trace.close(complete=complete)
        finally:
            if env is not None:env.close()
