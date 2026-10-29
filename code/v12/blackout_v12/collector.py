"""Frozen-policy natural episodes; replay stores support, never regenerates it."""
from pathlib import Path
import time
import numpy as np
import torch
from blackout_v11.trajectory import canonical_names, pack
from blackout_v11.evaluation import initial_map_hash, observation_hash
from .critic import critic_features
from .execution import compatible_manifest, semantic_config
from .io import append, atomic_json, atomic_torch, digest, file_hash, read_json
from .map_splits import register_map
from .rewards import OBJECTIVE, potential, shape_episode, terminal_outcome
from .trips import TripLedger


def collect_episode(policy, opponent, task, env_factory):
    side, cfg = int(task['side']), task['config']
    on_policy = not task.get('teacher', False) and not task.get('deterministic', False)
    own, other = canonical_names(side), canonical_names(1-side)
    expected = set(own+other)
    env = None
    started = time.monotonic()
    events, ledger = [], TripLedger(side)
    totals = np.zeros(2, np.float64)
    steps = 0
    try:
        env = env_factory()
        obs, _ = env.reset(seed=task['requested_seed'])
        initial = observation_hash(obs)
        map_hash = initial_map_hash(obs)
        split_audit = register_map(cfg, task.get('split', cfg.get('data_split', 'train')),
                                  map_hash, initial, task['requested_seed'], side)
        vector, graphic = pack(obs, own)
        policy.reset()
        for step in range(int(cfg.get('max_episode_steps', 22000))):
            if time.monotonic()-started > float(cfg.get('episode_wall_seconds', 1800)):
                raise TimeoutError('episode wall watchdog; partial episode excluded')
            if cfg.get('stop_file') and Path(cfg['stop_file']).exists():
                raise InterruptedError('episode stopped; partial episode excluded')
            with torch.inference_mode():
                result = policy.step(torch.from_numpy(vector),
                                     torch.from_numpy(graphic.copy())[None].expand(5,-1,-1,-1),
                                     deterministic=bool(task.get('deterministic', False)))
            if result['event']:
                record = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                          for k,v in result['decision'].items()}
                record['old_log_prob'] = torch.as_tensor(record.get('old_log_prob', record['log_prob'])).float().reshape(())
                record['temperature'] = torch.as_tensor(record['temperature']).float().reshape(())
                record['critic_features'] = critic_features(record['features'], record['legal'])
                record.update(step=step, potential=potential(vector), on_policy=on_policy)
                events.append(record)
                ledger.decision(step, record)
            selected = result['action'].detach().cpu().numpy().astype(np.float32)
            actions = dict(zip(own, selected))
            actions.update(opponent.act({n:obs[n] for n in other}, other))
            nxt, rewards, terms, truncs, infos = env.step(actions)
            steps = step+1
            if any(set(x) != expected for x in (rewards, terms, truncs)):
                raise ValueError('incomplete ten-agent transition')
            if any(truncs.values()) or len(set(bool(x) for x in terms.values())) != 1:
                raise ValueError('truncated/partial episode excluded')
            for s in (0,1):
                totals[s] += sum(float(rewards[n]) for n in canonical_names(s))
            if not np.isfinite(totals).all():
                raise FloatingPointError('nonfinite raw reward')
            terminal = all(terms.values())
            nv = ng = None
            if set(nxt) == expected:
                nv, ng = pack(nxt, own)
            elif not terminal:
                raise ValueError('nonterminal observation missing')
            clock_reset = nv is not None and float(nv[0,95]) > float(vector[0,95])+1e-3
            if clock_reset and not terminal:
                raise ValueError('clock reset without terminal')
            terminal_valid = nv is not None and not clock_reset
            ledger.observe(vector, nv, steps, terminal, terminal_valid, graphic)
            if terminal:
                if not events:
                    raise ValueError('natural episode contains no decision event')
                durations = np.diff([e['step'] for e in events]+[steps])
                outcome = terminal_outcome(totals, side)
                event_rewards = shape_episode([e['potential'] for e in events], durations, outcome, cfg.get('gamma', 1.))
                for e,k,r in zip(events,durations,event_rewards):
                    e.update(duration=int(k), reward=float(r))
                after_score = nv[0,93:95].tolist() if terminal_valid else None
                return dict(schema='v12.episode.v1', valid=True, natural_terminal=True,
                            objective=OBJECTIVE, policy_sha256=task.get('policy_sha256'),
                            episode_id=task['id'], arm=task.get('arm'), side=side,
                            env_steps=steps, events=events, event_count=len(events),
                            requested_seed=task['requested_seed'], action_seed=task['action_seed'],
                            initial_observation_sha256=initial, initial_map_hash=map_hash,
                            split_audit=split_audit, split=task.get('split', cfg.get('data_split', 'train')),
                            opponent=task['opponent'], family=task.get('family'),
                            runner_reward_totals=[float(totals[side]),float(totals[1-side])],
                            runner_winner=0 if outcome>0 else 1 if outcome<0 else None,
                            raw_points=(outcome+1)/2,
                            score_before_terminal=vector[0,93:95].tolist(),
                            score_after_terminal=after_score, score_valid=terminal_valid,
                            score_invalid_reason=None if terminal_valid else 'terminal observation missing/reset',
                            score_points=(float(np.sign(after_score[0]-after_score[1]))+1)/2 if terminal_valid else None,
                            trip_ledger=ledger.report(), wall_seconds=time.monotonic()-started,
                            on_policy=on_policy)
            obs, vector, graphic = nxt, nv, ng
        raise RuntimeError('episode watchdog; partial episode excluded')
    except BaseException as exc:
        if task.get('directory'):
            invalid=dict(valid=False,error=str(exc),
                type=type(exc).__name__, env_steps=steps, event_count=len(events),
                interrupted=isinstance(exc,InterruptedError),
                policy_sha256=task.get('policy_sha256'), no_optimizer_commit=True)
            atomic_json(Path(task['directory'])/'invalid.json', invalid)
            append(Path(task['directory'])/'discarded_attempts.jsonl',invalid)
            if invalid['interrupted']:
                append(Path(task['directory'])/'interrupted_attempts.jsonl',invalid)
        raise
    finally:
        if env is not None:
            env.close()


def run_episode(task, env_factory=None, actor_threads=None):
    """Spawn-safe entry, optionally using a caller-owned environment factory.

    Persistent actors inject a lease whose close() releases the episode but
    keeps Unity alive. Policies, opponents and RNG are still rebuilt for every
    immutable task, including after a behavior snapshot changes.
    """
    if task['config'].get('stop_file') and Path(task['config']['stop_file']).exists():
        raise InterruptedError('collection stopped before environment start')
    from .env import make_env
    from .opponents import make_opponent
    from .policy import MyPolicy
    torch.set_num_threads(int(task['config'].get('actor_threads',1)
                              if actor_threads is None else actor_threads))
    torch.manual_seed(int(task['action_seed']))
    directory = Path(task['directory']); directory.mkdir(parents=True,exist_ok=True)
    task_path = directory/'task.json'
    previous_task=read_json(task_path) if task_path.exists() else None
    if previous_task is not None:
        if not compatible_manifest(previous_task,task):
            raise ValueError('immutable episode task changed')
    else:
        atomic_json(task_path, task)
    semantic_task=digest(dict(task,config=semantic_config(task['config'])))
    if (directory/'episode.pt').exists():
        result = torch.load(directory/'episode.pt', map_location='cpu', weights_only=True)
        accepted_hashes={digest(task),digest(previous_task)} if previous_task is not None else {digest(task)}
        task_matches=(result.get('task_sha256') in accepted_hashes
                      or result.get('task_semantic_sha256')==semantic_task)
        if (not task_matches or not result.get('natural_terminal')
                or result.get('policy_sha256') != task.get('policy_sha256')):
            raise ValueError('cached episode does not match the immutable wave')
        summary = {k:v for k,v in result.items() if k not in ('events','trip_ledger')}
        summary['trajectory'] = str(directory/'episode.pt')
        summary['trip_summary'] = {k:v for k,v in result['trip_ledger'].items() if k!='rows'}
        atomic_json(directory/'result.json', summary)
        atomic_json(directory/'trips.json', result['trip_ledger'])
        return summary
    if task.get('checkpoint'):
        if task.get('policy_sha256') and file_hash(task['checkpoint']) != task['policy_sha256']:
            raise ValueError('behavior snapshot changed during episode wave')
        payload = torch.load(task['checkpoint'], map_location='cpu', weights_only=True)
    else:
        payload = {}
    policy = MyPolicy(task['config'], arm=task.get('arm','F')).eval()
    if payload:
        policy.scorer.load_state_dict(payload['scorer_state'])
    opponent = make_opponent(task['opponent'], task['action_seed']+1)
    result = collect_episode(policy, opponent, task,
                             env_factory if env_factory is not None else lambda:make_env(task))
    result['task_sha256'] = digest(task)
    result['task_semantic_sha256'] = semantic_task
    atomic_torch(directory/'episode.pt',result)
    summary = {k:v for k,v in result.items() if k not in ('events','trip_ledger')}
    summary['trajectory'] = str(directory/'episode.pt')
    summary['trip_summary'] = {k:v for k,v in result['trip_ledger'].items() if k!='rows'}
    atomic_json(directory/'result.json',summary)
    atomic_json(directory/'trips.json',result['trip_ledger'])
    return summary
