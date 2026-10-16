"""Conditional evaluation summaries; do not treat repeated initial maps as iid maps."""
import math
import numpy as np


def wilson(wins, count, z=1.959963984540054):
    if not count: return [0., 1.]
    p = wins/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0., center-radius), min(1., center+radius)]


def summarize(rows, invalid=0):
    if not rows: raise ValueError("cannot summarize empty evaluation")
    families = {}
    for row in rows:
        families.setdefault(row["opponent"]["kind"], []).append(row)
    def group(items):
        wins = sum(x["winner"] == 0 for x in items)
        draws = sum(x["winner"] is None for x in items)
        return {"n": len(items), "wins": wins, "draws": draws, "losses": len(items)-wins-draws,
                "win_rate": wins/len(items), "score_rate": (wins+.5*draws)/len(items),
                "conditional_binomial_interval": wilson(wins, len(items))}
    groups = {k: group(v) for k, v in families.items()}
    sides = {str(s): group([r for r in rows if r["side"] == s]) for s in (0, 1) if any(r["side"] == s for r in rows)}
    return {**group(rows), "families": groups, "sides": sides,
            "macro_win_rate": float(np.mean([g["win_rate"] for g in groups.values()])),
            "worst_family": min(g["win_rate"] for g in groups.values()),
            "invalid": invalid, "invalid_rate": invalid/(len(rows)+invalid),
            "distinct_initial_hashes_by_side": {str(s): len({r["initial_observation_sha256"] for r in rows if r["side"] == s}) for s in (0, 1)},
            "map_independence_unverified": True,
            "interval_scope": "conditional action-RNG repeats if independent; not unseen-map generalization",
            "objective": "provided_runner_v1", "game_outcome_verified": False}


def run_interval(results, seed=930, repetitions=2000):
    """Run-level uncertainty. Never converts all-zero data to certainty."""
    means = np.asarray([r["macro_win_rate"] for r in results])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(len(means), size=(repetitions, len(means)))].mean(1)
    return {"mean": float(means.mean()), "run_bootstrap": np.quantile(samples, [.025, .975]).tolist(),
            "train_seeds": len(means), "bootstrap_degenerate": bool(np.ptp(samples) == 0),
            "generalization_certified": False, "note": "few-seed descriptive interval; inspect per-run binomial intervals and repeated-map limitation"}
