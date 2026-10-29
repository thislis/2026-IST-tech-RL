"""Standalone, resumable F-policy fine tuning against the nine scripted agents.

Importing this module does not start Unity. A run first measures the frozen
warm start on development maps, then alternates complete 16-episode PPO waves
and development checks. Confirmation is a single final, independent check;
its outcomes never influence training or checkpoint selection.
"""
from contextlib import ExitStack
from collections import Counter
import copy
import math
from pathlib import Path
import random

import numpy as np
import torch

from .critic import Critic
from .evaluation import evaluate
from .execution import compatible_manifest, semantic_config
from .io import ROOT, append, atomic_json, atomic_torch, cpu_state, digest, file_hash, read_json, state_hash
from .policy import MyPolicy
from .ppo import update
from .runtime import FileLease, learner_lease
from .training import make_batch


SCRIPTED_IDS = ('scripted-collector', 'scripted-default', 'scripted-defender',
                'scripted-noop', 'scripted-raider', 'scripted-random',
                'scripted-rush', 'scripted-special', 'scripted-weak')
WAVE_SIZE = 16
DEFAULTS = dict(max_steps=8_388_608, eval_interval=524_288, dev_pairs=4,
                confirmation_pairs=32, minimum_opponent_probability=.05,
                adaptive_power=2., minibatch_events=128)
FRESH_SPLITS = dict(train=[4_100_000, 4_999_999], data=[5_000_000, 5_099_999],
                    dev=[5_100_000, 5_199_999], confirmation=[5_200_000, 5_299_999],
                    test=[5_300_000, 5_399_999])


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def _resolved_config(cfg, output_dir, workers):
    cfg = copy.deepcopy(cfg)
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    for key, value in cfg.get('ppo', {}).items():
        cfg.setdefault({'clip': 'ppo_clip', 'epochs': 'ppo_epochs'}.get(key, key), value)
    for key in ('max_steps', 'eval_interval', 'dev_pairs', 'confirmation_pairs',
                'max_episode_steps', 'actor_threads', 'learner_threads'):
        if key in cfg:
            cfg[key] = _positive_integer(cfg[key], key)
    floor = float(cfg['minimum_opponent_probability'])
    power = float(cfg['adaptive_power'])
    if not math.isfinite(floor) or not 0 < floor <= 1 / len(SCRIPTED_IDS):
        raise ValueError('minimum_opponent_probability must be in (0, 1/9]')
    if not math.isfinite(power) or power <= 0:
        raise ValueError('adaptive_power must be finite and positive')
    if cfg.get('strict_map_splits', True) is not True:
        raise ValueError('scripted fine tuning requires strict actual-map split auditing')
    cfg['splits'] = copy.deepcopy(cfg.get('splits') or FRESH_SPLITS)
    bounds = {}
    for split in ('train', 'dev', 'confirmation', 'test'):
        interval = cfg['splits'][split]
        interval = interval['seeds'] if isinstance(interval, dict) else interval
        if (len(interval) != 2 or any(isinstance(v, bool) or int(v) != v for v in interval)
                or int(interval[0]) > int(interval[1])):
            raise ValueError('invalid frozen seed interval for ' + split)
        bounds[split] = tuple(map(int, interval))
    for left, a in bounds.items():
        for right, b in bounds.items():
            if left < right and max(a[0], b[0]) <= min(a[1], b[1]):
                raise ValueError('seed intervals must be disjoint: ' + left + '/' + right)
    from .opponents import frozen_catalog
    rows = cfg.get('opponents') or frozen_catalog()
    if isinstance(rows, dict):
        rows = rows.get('catalog', list(rows.values()))
    scripted = [row for row in rows if row.get('id', '').startswith('scripted-')]
    if len(scripted) != 9 or {row['id'] for row in scripted} != set(SCRIPTED_IDS):
        raise ValueError('exactly the nine distinct scripted opponents are required')
    cfg['opponents'] = sorted(scripted, key=lambda row: row['id'])
    cfg.update(rollout_episodes_per_wave=WAVE_SIZE, actor_device='cpu',
               evaluation_worker_capacity=workers, strict_map_splits=True, data_split='train')
    # The deployment decoder is evaluated deterministically. Collection uses
    # task.deterministic=False and therefore samples the on-policy assignment.
    cfg['decoder'] = 'map'
    cfg.pop('worker_allocation_file', None)
    cfg.setdefault('map_split_registry', str(output_dir/'map_splits.json'))
    cfg.setdefault('data_directory', str(ROOT/'artifacts/data/v12'/output_dir.name))
    return cfg


def strict_report(report, opponents, pairs):
    """Count every planned cell, including invalids, in strict win rates.

    The objective remains provided_runner_v1. A valid terminal score is
    required as evidence; its numerical winner is reported separately.
    """
    ids = [row['id'] for row in opponents]
    expected = {(name, rep, side) for name in ids for rep in range(pairs) for side in (0, 1)}
    indexed = {}
    malformed = 0
    for row in report.get('rows', []):
        key = (row.get('opponent', {}).get('id'), row.get('replicate'), row.get('side'))
        if key not in expected or key in indexed:
            malformed += 1
        else:
            indexed[key] = row
    invalid_pairs = set()
    for name in ids:
        for rep in range(pairs):
            left, right = (indexed.get((name, rep, side), {}) for side in (0, 1))
            if (not left.get('initial_map_hash')
                    or left.get('initial_map_hash') != right.get('initial_map_hash')
                    or left.get('requested_seed') != right.get('requested_seed')):
                invalid_pairs.add((name, rep))

    def valid(row, name, rep):
        return (bool(row.get('valid')) and bool(row.get('natural_terminal'))
                and bool(row.get('score_valid')) and row.get('raw_points') in (0., .5, 1.)
                and row.get('score_points') in (0., .5, 1.)
                and row.get('split_audit', {}).get('valid') is True
                and row.get('actual_map_pairing_valid') is not False
                and (name, rep) not in invalid_pairs)

    per = {}
    for name in ids:
        rows = [(rep, side, indexed.get((name, rep, side), {}))
                for rep in range(pairs) for side in (0, 1)]
        good = [(rep, side, row) for rep, side, row in rows if valid(row, name, rep)]
        wins = sum(row['raw_points'] == 1. for _, _, row in good)
        draws = sum(row['raw_points'] == .5 for _, _, row in good)
        losses = sum(row['raw_points'] == 0. for _, _, row in good)
        per[name] = dict(games=2*pairs, wins=wins, draws=draws, losses=losses,
                         invalid_games=2*pairs-len(good), strict_win_rate=wins/(2*pairs),
                         score_wins=sum(row['score_points'] == 1. for _, _, row in good),
                         sides={str(side): dict(games=pairs,
                             wins=sum(row['raw_points'] == 1. for _, s, row in good if s == side),
                             strict_win_rate=sum(row['raw_points'] == 1. for _, s, row in good if s == side)/pairs)
                             for side in (0, 1)})
    rates = [row['strict_win_rate'] for row in per.values()]
    complete = (len(indexed) == len(expected) and not malformed
                and report.get('completeness') is True and not report.get('interrupted'))
    return dict(objective='provided_runner_v1', planned_games=len(expected),
                observed_games=len(report.get('rows', [])), completeness=complete,
                malformed_games=malformed, opponents=per,
                worst_opponent_win_rate=min(rates), opponent_macro_win_rate=sum(rates)/len(rates),
                wins=sum(row['wins'] for row in per.values()),
                draws=sum(row['draws'] for row in per.values()),
                losses=sum(row['losses'] for row in per.values()),
                invalid_games=sum(row['invalid_games'] for row in per.values()) + malformed,
                all_wins=complete and all(row['wins'] == 2*pairs for row in per.values()),
                rows_path=report.get('rows_path'),
                claim='Finite empirical results on frozen seeds; no guarantee on unseen games.')


def sampling_weights(dev, cfg):
    """Use development evidence only, with a nonzero probability floor."""
    floor = float(cfg['minimum_opponent_probability'])
    power = float(cfg['adaptive_power'])
    weakness = {name: (.05 + 1. - (dev or {}).get('opponents', {}).get(name, {}).get(
        'strict_win_rate', 0.)) ** power for name in SCRIPTED_IDS}
    total = sum(weakness.values())
    return {name: floor + (1.-9*floor)*value/total for name, value in weakness.items()}


def _rng_state(device):
    name, keys, position, gaussian, cached = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=dict(name=name, keys=torch.tensor(keys.astype(np.int64)),
                           position=position, gaussian=gaussian, cached=cached),
                cuda=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])


def _restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['name'], value['keys'].numpy().astype(np.uint32),
                         value['position'], value['gaussian'], value['cached']))
    if device.startswith('cuda'):
        torch.cuda.set_rng_state_all(state['cuda'])


def _snapshot(path, scorer_state, cfg, seed, identity):
    payload = dict(schema='v12.scripted_finetune.behavior.v1', scorer_state=scorer_state,
                   config=cfg, seed=seed, arm='F', identity=identity)
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=True)
        if (state_hash(previous['scorer_state']) != state_hash(scorer_state)
                or previous.get('identity') != identity
                or semantic_config(previous['config']) != semantic_config(cfg)):
            raise ValueError('immutable fine-tuning snapshot changed: ' + str(path))
    else:
        atomic_torch(path, payload)
    return file_hash(path)


def _wave_plan(directory, cfg, seed, counters, state, scorer_state, signature):
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory/'behavior.pt'
    policy_hash = _snapshot(snapshot, scorer_state, cfg, seed,
                            dict(signature=signature, wave=counters['wave']))
    weights = sampling_weights(state['latest_dev'], cfg)
    assigned = copy.deepcopy(counters['opponent_episodes'])
    sides = copy.deepcopy(counters['opponent_sides'])
    roster = {row['id']: row for row in cfg['opponents']}
    tasks = []
    # Every wave covers every opponent, even when one weak agent dominates
    # the adaptive mixture. The remaining seven slots follow weighted deficits.
    coverage = sorted(SCRIPTED_IDS, key=lambda name: (assigned[name], -weights[name], name))
    lower, upper = cfg['splits']['train'] if not isinstance(cfg['splits']['train'], dict) else cfg['splits']['train']['seeds']
    for index in range(WAVE_SIZE):
        serial = counters['episodes'] + index
        name = coverage[index] if index < 9 else max(SCRIPTED_IDS,
            key=lambda name: (weights[name]*(serial+1)-assigned[name], weights[name], name))
        side = min((0, 1), key=lambda side: (sides[name][side], (side-serial) % 2))
        assigned[name] += 1
        sides[name][side] += 1
        h = int(digest(['v12-scripted-finetune-train', seed, serial])[:15], 16)
        data_wave = Path(cfg['data_directory'])/directory.name
        tasks.append(dict(id=f'w{counters["wave"]:06d}_{index:02d}',
            directory=str(data_wave/f'episode_{index:02d}'), config=cfg, arm='F',
            split='train', opponent=roster[name], family='scripted', side=side,
            checkpoint=str(snapshot), policy_sha256=policy_hash,
            requested_seed=int(lower)+h % (int(upper)-int(lower)+1),
            action_seed=h % (2**31-1), deterministic=False))
    plan = dict(schema='v12.scripted_finetune.wave.v1', signature=signature,
                policy_sha256=policy_hash, sampling_weights=weights, tasks=tasks)
    path = directory/'plan.json'
    if path.exists():
        previous = read_json(path)
        if not compatible_manifest(previous, plan):
            raise ValueError('immutable fine-tuning episode wave changed')
    else:
        atomic_json(path, plan)
    return tasks, weights


def _validate_wave(tasks, summaries, policy_hash):
    by_id = {task['id']: task for task in tasks}
    if (len(summaries) != len(tasks)
            or {row['episode_id'] for row in summaries} != set(by_id)):
        raise ValueError('incomplete or duplicate fine-tuning wave')
    episodes = []
    for row in sorted(summaries, key=lambda row: row['episode_id']):
        task = by_id[row['episode_id']]
        episode = torch.load(row['trajectory'], map_location='cpu', weights_only=True)
        if (episode.get('episode_id') != task['id'] or episode.get('policy_sha256') != policy_hash
                or not episode.get('valid') or not episode.get('natural_terminal')
                or not episode.get('on_policy') or episode.get('opponent') != task['opponent']
                or episode.get('side') != task['side'] or episode.get('split') != 'train'
                or episode.get('requested_seed') != task['requested_seed']
                or episode.get('action_seed') != task['action_seed']
                or episode.get('split_audit', {}).get('valid') is not True
                or int(episode.get('env_steps', 0)) < 1
                or int(episode.get('event_count', -1)) != len(episode.get('events', []))):
            raise ValueError('invalid or foreign fine-tuning episode')
        episodes.append(episode)
    return episodes


def run(output_dir: Path, cfg: dict, initial_checkpoint: Path, workers=16,
        learner_device='cuda', seed=20261027):
    """Warm start scorer + critic, reset Adam, then resume only this run.

    The finite step budget is checked between complete natural episode waves.
    Consequently committed steps can exceed it by less than one wave (at most
    16 * max_episode_steps); partial waves never receive an optimizer commit.
    """
    output_dir, initial_checkpoint = Path(output_dir).resolve(), Path(initial_checkpoint).resolve()
    workers = _positive_integer(workers, 'workers')
    if workers > WAVE_SIZE:
        raise ValueError('at most 16 resident Unity workers are allowed')
    if output_dir == initial_checkpoint.parent:
        raise ValueError('fine tuning requires a separate output directory')
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)
    learner_device = str(learner_device)
    if learner_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA learner requested but CUDA is unavailable')
    cfg = _resolved_config(cfg, output_dir, workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    with FileLease(output_dir/'.finetune.lock', blocking=False):
        return _run_locked(output_dir, cfg, initial_checkpoint, workers, learner_device, int(seed))


def _run_locked(output_dir, cfg, initial_checkpoint, workers, learner_device, seed):
    from .finetune_workers import PersistentEpisodePool
    signature = digest(dict(schema='v12.scripted_finetune.v1', seed=seed, arm='F',
                            initial_sha256=file_hash(initial_checkpoint), config=semantic_config(cfg)))
    latest = output_dir/'latest.pt'
    best_path = output_dir/'best.pt'
    resolved_path = output_dir/'resolved_config.json'
    if resolved_path.exists() and semantic_config(read_json(resolved_path)) != semantic_config(cfg):
        raise ValueError('fine-tuning configuration changed; use a new output directory')
    atomic_json(resolved_path, cfg)
    torch.set_num_threads(int(cfg.get('learner_threads', 1)))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    scorer = MyPolicy(cfg, arm='F').scorer.to(learner_device)
    critic = Critic().to(learner_device)
    actor_opt = torch.optim.Adam(scorer.parameters(), lr=float(cfg.get('actor_lr', 1e-4)))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=float(cfg.get('critic_lr', 3e-4)))
    counters = dict(env_steps=0, events=0, episodes=0, wave=0, discarded_env_steps=0,
                    opponent_episodes={name: 0 for name in SCRIPTED_IDS},
                    opponent_steps={name: 0 for name in SCRIPTED_IDS},
                    opponent_sides={name: [0, 0] for name in SCRIPTED_IDS})
    state = dict(phase='dev_pending', dev_round=0, next_eval_step=cfg['eval_interval'], latest_update=None,
                 latest_dev=None, best_dev=None, confirmation=None, termination_reason=None)

    def payload():
        return dict(schema='v12.scripted_finetune.v1', signature=signature, arm='F', seed=seed,
            feature_schema=('blackout.v12.features.learned.1' if cfg.get('policy_mode') == 'learned'
                            else 'blackout.v12.features.1'),
            initial_checkpoint=str(initial_checkpoint), initial_sha256=file_hash(initial_checkpoint),
            config=cfg, scorer_state=cpu_state(scorer), critic_state=cpu_state(critic),
            actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(),
            counters=copy.deepcopy(counters), state=copy.deepcopy(state), rng_state=_rng_state(learner_device))

    def restore(checkpoint):
        nonlocal counters, state
        if checkpoint.get('signature') != signature or checkpoint.get('schema') != 'v12.scripted_finetune.v1':
            raise ValueError('fine-tuning source/configuration identity changed; use a new output directory')
        scorer.load_state_dict(checkpoint['scorer_state'])
        critic.load_state_dict(checkpoint['critic_state'])
        actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        critic_opt.load_state_dict(checkpoint['critic_optimizer'])
        counters, state = copy.deepcopy(checkpoint['counters']), copy.deepcopy(checkpoint['state'])
        _restore_rng(checkpoint['rng_state'], learner_device)

    def save():
        atomic_torch(latest, payload())

    def progress(status, error=None):
        atomic_json(output_dir/'progress.json', dict(status=status, phase=state['phase'],
            counters=counters, env_steps=counters['env_steps'], requested_steps=cfg['max_steps'],
            parent_env_steps=int(cfg.get('parent_env_steps', 0)),
            lineage_env_steps=int(cfg.get('parent_env_steps', 0))+counters['env_steps'],
            policy_mode=cfg.get('policy_mode', 'residual'),
            next_eval_step=state['next_eval_step'], latest_dev=state['latest_dev'], best_dev=state['best_dev'],
            confirmation=state['confirmation'], termination_reason=state['termination_reason'],
            latest_update=state.get('latest_update'),
            latest_checkpoint=str(latest), selected_checkpoint=str(best_path) if best_path.exists() else None,
            empirical_target_met=bool(state['best_dev'] and state['best_dev']['all_wins']
                and state['confirmation'] and state['confirmation']['all_wins']),
            overshoot=max(0, counters['env_steps']-cfg['max_steps']),
            maximum_wave_steps=WAVE_SIZE*int(cfg.get('max_episode_steps', 22000)),
            error=error))

    def check_stop():
        if cfg.get('stop_file') and Path(cfg['stop_file']).exists():
            raise InterruptedError('fine tuning stopped; uncommitted wave will be replayed')

    if latest.exists():
        restore(torch.load(latest, map_location='cpu', weights_only=True))
    else:
        if best_path.exists() or (output_dir/'result.json').exists():
            raise ValueError('incomplete fine-tuning output lacks its resumable latest checkpoint')
        source = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
        if source.get('arm', 'F') != 'F' or 'critic_state' not in source:
            raise ValueError('warm start requires an F scorer and critic checkpoint')
        scorer.load_state_dict(source['scorer_state'])
        critic.load_state_dict(source['critic_state'])
        save()  # Adam is deliberately fresh; never copy source optimizers.

    pool_stack = ExitStack()
    pool = None
    def close_pool():
        nonlocal pool, pool_stack
        pool_stack.close()
        pool_stack = ExitStack()
        pool = None

    def run_evaluation(split, checkpoint, directory, pairs):
        close_pool()  # Evaluation may own 16 environments; never overlap pools.
        check_stop()
        checkpoint_payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if checkpoint_payload.get('signature') != signature:
            raise ValueError('evaluation checkpoint identity changed')
        scorer_state = checkpoint_payload['scorer_state']
        if split == 'confirmation' and state_hash(scorer_state) != state['best_dev']['candidate_scorer_sha256']:
            raise ValueError('selected best checkpoint changed since development evaluation')
        candidate = directory/'candidate.pt'
        _snapshot(candidate, scorer_state, cfg, seed,
                  dict(signature=signature, split=split, directory=str(directory)))
        rng = _rng_state(learner_device)
        try:
            report = evaluate(directory, cfg, candidate, 'F', cfg['opponents'], pairs, split, workers)
        finally:
            _restore_rng(rng, learner_device)
        check_stop()
        result = strict_report(report, cfg['opponents'], pairs)
        result.update(env_steps=counters['env_steps'], evaluation_directory=str(directory),
                      candidate_scorer_sha256=state_hash(scorer_state))
        atomic_json(directory/'strict_summary.json', result)
        return result

    try:
        while state['phase'] != 'complete':
            check_stop()
            if state['phase'] == 'dev_pending':
                progress('baseline_evaluation' if counters['wave'] == 0 else 'development_evaluation')
                directory = output_dir/'evaluations'/f'dev_{state["dev_round"]:04d}'
                dev = run_evaluation('dev', latest, directory, cfg['dev_pairs'])
                state['latest_dev'] = dev
                rank = lambda row: (row['worst_opponent_win_rate'], row['opponent_macro_win_rate'])
                if state['best_dev'] is None or rank(dev) > rank(state['best_dev']):
                    state['best_dev'] = dev
                    atomic_torch(best_path, payload())
                state['dev_round'] += 1
                while state['next_eval_step'] <= counters['env_steps']:
                    state['next_eval_step'] += cfg['eval_interval']
                if dev['all_wins'] or counters['env_steps'] >= cfg['max_steps']:
                    state['phase'] = 'confirmation_pending'
                    state['termination_reason'] = 'development_all_wins' if dev['all_wins'] else 'step_budget_exhausted'
                else:
                    state['phase'] = 'training'
                save()
            elif state['phase'] == 'confirmation_pending':
                progress('confirmation_evaluation')
                state['confirmation'] = run_evaluation('confirmation', best_path,
                    output_dir/'evaluations'/'confirmation', cfg['confirmation_pairs'])
                state['phase'] = 'complete'
                save()
            elif state['phase'] == 'training':
                progress('training')
                wave = counters['wave']
                directory = output_dir/'waves'/f'wave_{wave:06d}'
                tasks, weights = _wave_plan(directory, cfg, seed, counters, state, cpu_state(scorer), signature)
                before = torch.load(latest, map_location='cpu', weights_only=True)
                try:
                    if pool is None:
                        pool = pool_stack.enter_context(PersistentEpisodePool(workers, visible=cfg.get('visible_training', True)))
                    summaries = pool.run(tasks)
                    check_stop()
                    episodes = _validate_wave(tasks, summaries, tasks[0]['policy_sha256'])
                    with learner_lease(learner_device):
                        batch = make_batch(episodes, critic, cfg.get('gamma', 1.), cfg.get('gae_lambda', .9975))
                        with torch.no_grad():
                            residual = scorer(batch['features'][:128].to(learner_device))
                            diagnostics = dict(learned_residual_abs_mean=float(residual.abs().mean()),
                                learned_residual_abs_max=float(residual.abs().max()))
                        metrics = update(scorer, critic, actor_opt, critic_opt, batch, 'F', cfg)
                        with torch.no_grad():
                            diagnostics['actor_parameter_delta_l2'] = math.sqrt(sum(float((value.detach().cpu()
                                - before['scorer_state'][name]).square().sum()) for name, value in scorer.named_parameters()))
                        metrics = dict(metrics, **diagnostics)
                    chosen_tasks = Counter()
                    for episode in episodes:
                        for event in episode['events']:
                            if 'task_ids' in event:
                                chosen_tasks.update(event['task_ids'][int(choice)].split(':')[0]
                                                    for choice in event['assignment'])
                    metrics['selected_task_counts'] = dict(chosen_tasks)
                    # Observation diagnostics only; these never alter actions,
                    # rewards, candidate eligibility, or curriculum weights.
                    if cfg.get('policy_mode') == 'learned':
                        metrics['max_no_progress_log'] = max(float(event['features'][..., 10].max())
                            for episode in episodes for event in episode['events'])
                    check_stop()
                    for episode in episodes:
                        name, side, steps = episode['opponent']['id'], episode['side'], episode['env_steps']
                        counters['env_steps'] += steps
                        counters['events'] += episode['event_count']
                        counters['episodes'] += 1
                        counters['opponent_episodes'][name] += 1
                        counters['opponent_steps'][name] += steps
                        counters['opponent_sides'][name][side] += 1
                    for task in tasks:
                        audit = Path(task['directory'])/'discarded_attempts.jsonl'
                        if audit.exists():
                            import json
                            counters['discarded_env_steps'] += sum(int(json.loads(line)['env_steps'])
                                for line in audit.read_text(encoding='utf-8').splitlines() if line)
                    counters['wave'] += 1
                    state['latest_update'] = dict(wave=wave, **metrics)
                    if counters['env_steps'] >= min(state['next_eval_step'], cfg['max_steps']):
                        state['phase'] = 'dev_pending'
                    save()  # One atomic checkpoint is the only wave commit.
                except BaseException as exc:
                    restore(before)
                    append(output_dir/'rollback_waves.jsonl', dict(wave=wave, error=str(exc),
                        type=type(exc).__name__, no_optimizer_commit=True))
                    raise
                append(output_dir/'updates.jsonl', dict(wave=wave, counters=copy.deepcopy(counters),
                    sampling_weights=weights, metrics=metrics))
                atomic_json(directory/'commit.json', dict(wave=wave, episodes=len(episodes),
                    committed_env_steps=counters['env_steps'], policy_sha256=tasks[0]['policy_sha256']))
            else:
                raise ValueError('unknown fine-tuning checkpoint phase: ' + str(state['phase']))
        target = bool(state['best_dev']['all_wins'] and state['confirmation']['all_wins'])
        result = dict(schema='v12.scripted_finetune.result.v1', status='complete',
            checkpoint=str(best_path), latest_checkpoint=str(latest), arm='F', seed=seed,
            initial_checkpoint=str(initial_checkpoint), signature=signature,
            counters=counters, env_steps=counters['env_steps'], requested_steps=cfg['max_steps'],
            parent_env_steps=int(cfg.get('parent_env_steps', 0)),
            lineage_env_steps=int(cfg.get('parent_env_steps', 0))+counters['env_steps'],
            policy_mode=cfg.get('policy_mode', 'residual'),
            overshoot=max(0, counters['env_steps']-cfg['max_steps']),
            best_dev=state['best_dev'], latest_dev=state['latest_dev'], confirmation=state['confirmation'],
            empirical_target_met=target, target_status='met_on_frozen_samples' if target else 'unmet',
            termination_reason=state['termination_reason'],
            confirmation_used_for_training=False, test_evaluated=False,
            judgement='Finite empirical results on frozen development and independent confirmation maps; '
                      '100% on these samples does not guarantee universal 100% win rate.')
        atomic_json(output_dir/'result.json', result)
        progress('complete')
        return result
    except BaseException as exc:
        progress('interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', str(exc))
        raise
    finally:
        close_pool()
