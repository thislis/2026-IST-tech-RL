"""Evaluation-only reduction, separate from the immutable v11.study.2 budget.

Completed baseline/pilot tasks retain their original plans. Confirmation is a
five-opponent reproducibility diagnostic; only the locked submitted checkpoint
receives the final full-league performance assessment.
"""
from pathlib import Path
from .io import atomic_json, digest, read_json
from .opponents import dev_catalog, frozen_catalog, reference_spec


SCHEMA = 'v11.evaluation.1'
MAIN_DEV_MAP_PAIRS = 5
CONFIRMATION_MAP_PAIRS = 10
FINAL_MAP_PAIRS = 10
STRONG_FINAL_MAP_PAIRS = 30


def confirmation_catalog():
    scripted = {'scripted-noop', 'scripted-random', 'scripted-collector'}
    pool = [s for s in frozen_catalog()
            if s['id'] in scripted or s['id'].startswith(('v4-', 'v6-'))]
    if len(pool) != 5 or not scripted.issubset({s['id'] for s in pool}):
        raise ValueError('confirmation requires noop/random/collector and v4/v6')
    return pool


def final_map_pairs():
    return {s['id']: STRONG_FINAL_MAP_PAIRS if s['id'].startswith(('v4-', 'v6-'))
            else FINAL_MAP_PAIRS for s in frozen_catalog()}


def build_evaluation_schedule(cfg):
    dev = [s['id'] for s in dev_catalog()]
    confirmation = [s['id'] for s in confirmation_catalog()]
    final = final_map_pairs()
    reference = reference_spec()['id']
    main_candidates = len(cfg['arms']) * len(cfg['seeds'])
    seeds = list(cfg['seeds']) + list(cfg['confirmation_seeds'])
    confirmation_candidates = 2 * len(seeds)
    candidate_games = 2 * sum(final.values())
    reference_games = 2 * sum(n for opponent, n in final.items() if opponent != reference)
    schedule = dict(
        schema=SCHEMA, learning_plan_schema='v11.study.2', learning_budgets_changed=False,
        reuse_completed=dict(stages=['lifecycle', 'baseline', 'pilot'],
                            original_task_plans_and_decoders_retained=True),
        main_dev=dict(split='dev', opponents=dev, map_pairs_per_opponent=MAIN_DEV_MAP_PAIRS,
                      candidates=main_candidates, games_per_candidate=2*len(dev)*MAIN_DEV_MAP_PAIRS,
                      total_games=2*len(dev)*MAIN_DEV_MAP_PAIRS*main_candidates,
                      role='arm_and_submission_seed_selection'),
        confirmation=dict(split='confirmation', opponents=confirmation,
                          map_pairs_per_opponent=CONFIRMATION_MAP_PAIRS,
                          arm_roles=['winner', 'comparison'], training_seeds=seeds,
                          candidates=confirmation_candidates,
                          games_per_candidate=2*len(confirmation)*CONFIRMATION_MAP_PAIRS,
                          total_games=2*len(confirmation)*CONFIRMATION_MAP_PAIRS*confirmation_candidates,
                          role='reproducibility_diagnostic_only', selection_allowed=False,
                          performance_gate_blocking=False, full_league_G4_G5_G6_applied=False),
        frozen_test=dict(split='test', opponents=list(final), map_pairs_by_opponent=final,
                         submission_candidates=1, submission_games=candidate_games,
                         reference_id=reference, reference_excludes_self=True,
                         reference_games=reference_games, total_games=candidate_games+reference_games,
                         role='locked_submission_final_performance', selection_allowed=False,
                         existing_G4_G5_G6_thresholds_retained=True),
        selection_lock_required_before_confirmation=True,
        distinct_splits=['dev', 'confirmation', 'test'],
    )
    schedule['total_remaining_evaluation_games'] = sum(
        schedule[stage]['total_games'] for stage in ('main_dev', 'confirmation', 'frozen_test'))
    return schedule


def record_evaluation_schedule(directory, cfg):
    schedule = build_evaluation_schedule(cfg)
    path = Path(directory) / 'evaluation_schedule.json'
    if path.exists():
        if read_json(path) != schedule:
            raise ValueError('registered evaluation schedule changed')
    else:
        atomic_json(path, schedule)
    return schedule, digest(schedule)
