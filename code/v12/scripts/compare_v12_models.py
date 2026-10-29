"""Paired, side-swapped comparison of two immutable exported v12 policies."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'code/shared/project_paths.py').is_file())
sys.path.insert(0, str(ROOT/'code/shared'))
from project_paths import activate
activate()

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import time
import traceback
import numpy as np
import torch
from blackout_v12.io import atomic_json, digest, file_hash, read_json
from blackout_v12.runtime import FileLease, child_environment, managed_popen, process_options, process_tree_guard


def play(task):
    from blackout_v12.export import load_export
    from blackout_v12.env import make_env
    from blackout_v12.map_splits import register_map
    from blackout_v11.evaluation import initial_map_hash, observation_hash
    from blackout_v11.trajectory import canonical_names
    torch.set_num_threads(1)
    torch.manual_seed(task['seed'])
    policies = []
    for spec in task['models']:
        directory = Path(spec['directory'])
        for name, expected in spec['hashes'].items():
            if file_hash(directory/name) != expected:
                raise ValueError('Export changed: '+str(directory/name))
        policies.append(load_export(directory, device='cpu'))
    before_side = task['before_side']
    by_side = [policies[0 if side == before_side else 1] for side in (0, 1)]
    teams = [canonical_names(side) for side in (0, 1)]
    expected = set(teams[0]+teams[1])
    env = None
    start = time.monotonic()
    totals = [0., 0.]
    output = Path(task['output'])
    try:
        env = make_env(task)
        obs, _ = env.reset(seed=task['seed'])
        map_hash = initial_map_hash(obs)
        audit = register_map(task['config'], 'test', map_hash, observation_hash(obs), task['seed'], before_side)
        for step in range(22000):
            if time.monotonic()-start > 1800:
                raise TimeoutError('Match wall watchdog; incomplete match is not a draw')
            actions = {}
            with torch.inference_mode():
                for side in (0, 1):
                    actions.update(by_side[side].act({n:obs[n] for n in teams[side]}))
            if set(actions) != expected:
                raise ValueError('Missing team action')
            nxt, rewards, terms, truncs, infos = env.step(actions)
            if (any(set(x) != expected for x in (rewards, terms, truncs))
                    or any(truncs.values()) or len(set(map(bool, terms.values()))) != 1):
                raise ValueError('Incomplete or truncated game transition')
            for side in (0, 1):
                totals[side] += sum(float(rewards[n]) for n in teams[side])
            if not np.isfinite(totals).all():
                raise ValueError('Nonfinite reward')
            if all(terms.values()):
                winners = [infos.get(n, {}).get('winner') for n in sorted(expected)]
                official = (len(winners) == 10 and all(w in (-1, 0, 1) for w in winners)
                            and len(set(winners)) == 1)
                scores = None
                if set(nxt) == expected:
                    v = np.asarray(nxt[teams[0][0]]['vector'])
                    previous = np.asarray(obs[teams[0][0]]['vector'])
                    if float(v[95]) <= float(previous[95])+1e-3:
                        scores = v[93:95].tolist()
                if official:
                    winner = int(winners[0])
                    criterion = 'terminal_info.winner'
                elif scores is not None:
                    winner = 0 if scores[0] > scores[1] else 1 if scores[1] > scores[0] else -1
                    criterion = 'terminal_score_fallback'
                else:
                    raise ValueError('Terminal winner and usable terminal score are both missing')
                runner_winner = 0 if totals[0] > totals[1] else 1 if totals[1] > totals[0] else -1
                row = dict(valid=True, id=task['id'], pair=task['pair'], seed=task['seed'],
                    before_side=before_side, after_side=1-before_side, map_hash=map_hash,
                    map_audit=audit, steps=step+1, winner_side=winner,
                    winner='draw' if winner == -1 else 'before' if winner == before_side else 'after',
                    criterion=criterion, scores_A_B=scores, runner_totals_A_B=totals,
                    runner_winner='draw' if runner_winner == -1 else 'before' if runner_winner == before_side else 'after',
                    wall_seconds=time.monotonic()-start)
                atomic_json(output/'cells'/(task['id']+'.json'), row)
                return row
            if set(nxt) != expected:
                raise ValueError('Nonterminal observations missing')
            obs = nxt
            if (step+1) % 500 == 0:
                atomic_json(output/'live'/(task['id']+'.json'),
                            dict(step=step+1, elapsed=time.monotonic()-start, updated_at=time.time()))
        raise RuntimeError('Step watchdog; incomplete match is not a draw')
    except BaseException as exc:
        atomic_json(output/'cells'/(task['id']+'.json'), dict(valid=False, id=task['id'],
            pair=task['pair'], before_side=before_side, seed=task['seed'], error=str(exc),
            traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()


def summarize(rows, planned):
    valid = [r for r in rows if r.get('valid')]
    for pair in sorted({r['pair'] for r in valid}):
        paired = [r for r in valid if r['pair'] == pair]
        if len(paired) == 2 and (len({r['map_hash'] for r in paired}) != 1
                                or {r['before_side'] for r in paired} != {0, 1}):
            raise ValueError('Side-swapped pairing mismatch')
    def counts(group, criterion='winner'):
        return {key:sum(r[criterion] == key for r in group) for key in ('before', 'after', 'draw')}
    return dict(planned_games=planned, completed_games=len(valid), failed_games=len(rows)-len(valid),
                complete=len(valid) == planned, wins=counts(valid), runner_wins=counts(valid, 'runner_winner'),
                by_before_side={str(side):dict(games=sum(r['before_side'] == side for r in valid),
                    **counts([r for r in valid if r['before_side'] == side])) for side in (0, 1)},
                official_winner_games=sum(r['criterion'] == 'terminal_info.winner' for r in valid),
                winner_disagreements=sum(r['winner'] != r['runner_winner'] for r in valid), rows=valid)


def worker(output):
    plan = read_json(output/'plan.json')
    rows = []
    pending = []
    for task in plan['tasks']:
        path = output/'cells'/(task['id']+'.json')
        if path.exists():
            row = read_json(path)
            if row.get('valid'):
                rows.append(row)
                continue
            raise RuntimeError('Previous failed match requires review: '+str(path))
        pending.append(task)
    atomic_json(output/'status.json', dict(state='running', completed=len(rows), planned=len(plan['tasks'])))
    with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(play, task):task for task in pending}
        for future in as_completed(jobs):
            rows.append(future.result())
            report = summarize(rows, len(plan['tasks']))
            atomic_json(output/'status.json', dict(state='running', completed=len(rows),
                        planned=len(plan['tasks']), wins=report['wins'], updated_at=time.time()))
            print(f"Completed {len(rows)}/{len(plan['tasks'])}: {report['wins']}", flush=True)
    result = summarize(rows, len(plan['tasks']))
    atomic_json(output/'result.json', result)
    atomic_json(output/'status.json', dict(state='complete', completed=len(rows), wins=result['wins']))
    print(json_summary(result), flush=True)


def json_summary(result):
    import json
    return json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='before_after_20261028')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..') or not 1 <= args.workers <= 16:
        parser.error('Simple comparison name and workers 1..16 required')
    output = ROOT/'logs/v12/comparisons'/args.name
    if args.worker:
        worker(output)
        return
    output.mkdir(parents=True, exist_ok=True)
    with FileLease(ROOT/'logs/runtime/compute.lock', blocking=False), \
         FileLease(output/'experiment.lock', blocking=False), process_tree_guard():
        models = []
        for name in ('scripted9_before_learned_latest', 'scripted9_learned_v12'):
            directory = ROOT/'artifacts/submission/v12'/name
            models.append(dict(directory=str(directory),
                               hashes={f:file_hash(directory/f) for f in ('policy.py', 'checkpoint.pt')}))
        cfg = dict(time_scale=50, map_split_registry=str(output/'map_splits.json'), strict_map_splits=True)
        if not (output/'map_splits.json').exists():
            prior = read_json(ROOT/'logs/v12/scripted9_learned_v12/map_splits.json')
            atomic_json(output/'map_splits.json', dict(schema='v12.actual_maps.1',
                maps={h:dict(split='previous_experiments', observations=[], seeds=[]) for h in prior['maps']},
                collisions=[], seed_maps={}))
        tasks = []
        for pair in range(10):
            seed = 8_100_000 + int(digest(['before-after-v12', 20261028, pair])[:12], 16) % 100000
            for before_side in (0, 1):
                tasks.append(dict(id=f'pair_{pair:02d}_before_side_{before_side}', pair=pair,
                                  seed=seed, before_side=before_side, models=models, config=cfg, output=str(output)))
        if len({task['seed'] for task in tasks}) != 10:
            raise ValueError('Duplicate paired seeds')
        plan = dict(models=models, tasks=tasks, workers=args.workers,
                    description='Original pre-change bundle versus selected post-training bundle; code and weights both differ')
        if (output/'plan.json').exists() and read_json(output/'plan.json') != plan:
            raise ValueError('Comparison manifest changed')
        atomic_json(output/'plan.json', plan)
        with (output/'console.log').open('ab', buffering=0) as log:
            child = managed_popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--name', args.name],
                cwd=ROOT, env=child_environment(), stdout=log, stderr=log, **process_options())
            print('Comparison running: '+str(output), flush=True)
            code = child.wait()
        if code:
            atomic_json(output/'status.json', dict(state='failed', error=f'Comparison worker exited {code}'))
            raise RuntimeError('Comparison failed; inspect '+str(output/'console.log'))
        print(json_summary(read_json(output/'result.json')), flush=True)


if __name__ == '__main__':
    main()
