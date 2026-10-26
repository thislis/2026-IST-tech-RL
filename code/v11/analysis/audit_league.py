"""Read-only v11 planning audit; does not load policies or launch Unity.

Reconcile every completed cell, its attempt result, initial map, and final
transition with the published league analysis. Intermediate trace chunks are
not reread. Output belongs under logs/v11/, never in the source league.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/shared"))
from agent_league.trace import load_chunk


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source):
    manifest = read(source / "manifest.json")
    analysis = read(source / "analysis_data.json")
    published = {m["id"]: m for m in analysis["models"]}
    tasks = {t["id"]: t for t in manifest["tasks"]}
    files = sorted((source / "cells").glob("*.json"))
    assert len(files) == len(tasks) == len(manifest["tasks"])
    assert {p.stem for p in files} == set(tasks)
    models = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(list)
    maps = {k: Counter() for k in ("all", "walls", "storage", "battery")}
    paired_maps = defaultdict(list)
    counts = Counter()
    terminal_cases = {}
    hashes = hashlib.sha256()
    case_ids = {c["id"] for c in analysis["cases"]}
    for path in files:
        cell = read(path)
        task, result = cell["task"], cell["result"]
        for key, value in tasks[path.stem].items():
            assert task[key] == value, (path, key)
        directory = Path(cell["directory"])
        # Resolve saved absolute paths after a workspace relocation.
        if not directory.exists():
            directory = source / "attempts" / directory.parent.name / directory.name
        assert read(directory / "result.json")["result"] == result, path
        assert cell["valid"] and result["valid"] and result["natural_terminal"]
        assert not result["probe"] and result["objective"] == "provided_runner_v1"
        side_ids = [None, None]
        side_ids[task["side"]], side_ids[1-task["side"]] = task["a"], task["b"]
        rewards = result["side_rewards"]
        assert np.isfinite(rewards).all()
        rw = 0 if rewards[0] > rewards[1] else 1 if rewards[1] > rewards[0] else None
        assert rw == result["winner_side"]
        assert result["winner"] == (side_ids[rw] if rw is not None else None)
        trace = directory / "trace"
        index = read(trace / "index.json")
        assert index["complete"] and index["steps"] == result["episode_steps"]
        assert sum(c["count"] for c in index["chunks"]) == index["steps"]
        first, last = index["snapshots"][0], index["chunks"][-1]
        assert first["step"] == 0
        assert sha(trace / first["path"]) == first["sha256"]
        assert sha(trace / last["path"]) == last["sha256"]
        with np.load(trace / first["path"], allow_pickle=False) as snap:
            graphic = snap["unit_0"]
            for key, channels in (("all", None), ("walls", [1]), ("storage", [2, 3]), ("battery", [6])):
                data = graphic if channels is None else graphic[..., channels]
                digest = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
                maps[key][digest] += 1
        paired_maps[(task["a"], task["b"], task["repeat"])].append(first["sha256"])
        chunk = load_chunk(trace / last["path"])
        assert int(chunk["step"][-1]) + 1 == result["episode_steps"]
        assert chunk["terminated"][-1].all() and not chunk["truncated"][-1].any()
        assert bool(chunk["after_present"][-1]) and bool(chunk["transition_valid"][-1])
        assert np.allclose(chunk["cumulative_reward"][-1], rewards, atol=1e-6, rtol=0)
        before = chunk["before"][-1, 0, 93:95]
        after = chunk["after"][-1, 0, 93:95]
        assert np.isfinite(after).all()
        # Score is an observed diagnostic, not the discarded official winner info.
        sw = 0 if after[0] > after[1] else 1 if after[1] > after[0] else None
        bw = 0 if before[0] > before[1] else 1 if before[1] > before[0] else None
        counts["games"] += 1
        counts["steps"] += result["episode_steps"]
        counts["reward_draws"] += rw is None
        counts["opposite_decisive_winners"] += rw is not None and sw is not None and rw != sw
        counts["score_tie_reward_decisive"] += sw is None and rw is not None
        counts["reward_tie_score_decisive"] += rw is None and sw is not None
        counts["score_winner_changed_final_transition"] += bw != sw
        if path.stem in case_ids:
            terminal_cases[path.stem] = dict(sides=side_ids, reward=rewards,
                score_before=before.tolist(), score_after=after.tolist(), steps=result["episode_steps"])
        for side, model in enumerate(side_ids):
            bucket = models[model]
            bucket["reward_points"].append(.5 if rw is None else float(rw == side))
            bucket["score_points"].append(.5 if sw is None else float(sw == side))
            bucket["terminal_score"].append(float(after[side]) * 100)
            bucket["side_"+str(side)].append(.5 if rw is None else float(rw == side))
            for key, value in result["behavior"]["sides"][side].items():
                if isinstance(value, (float, int)):
                    bucket[key].append(value)
        pairs[(task["a"], task["b"])].append(.5 if rw is None else float(rw == task["side"]))
        hashes.update(path.name.encode())
        hashes.update(bytes.fromhex(sha(path)))
    rows = []
    for model, values in sorted(models.items()):
        n = len(values["reward_points"])
        row = dict(id=model, n=n, wins=values["reward_points"].count(1.),
                   draws=values["reward_points"].count(.5), losses=values["reward_points"].count(0.))
        row.update({k: float(np.mean(v)) for k, v in values.items()})
        p = published[model]
        assert (row["n"], row["wins"], row["draws"], row["losses"]) == (p["n"], p["wins"], p["draws"], p["losses"])
        assert abs(row["score_points"] - p["terminal_score_rate"]) < 1e-10
        for key, value in p["behavior"].items():
            assert abs(row[key] - value) < 1e-10, (model, key)
        rows.append(row)
    for pair in analysis["pairs"]:
        values = pairs[(pair["a"], pair["b"])]
        assert len(values) == pair["n"] and abs(np.mean(values)-pair["rate"]) < 1e-10
    assert counts["opposite_decisive_winners"] == analysis["audit"]["apparent_score_opposite"]
    assert counts["score_tie_reward_decisive"] == analysis["audit"]["visible_tie_reward_decisive"]
    return dict(schema="blackout.v11.planning_audit.v1", checked_at=datetime.now(timezone.utc).isoformat(),
        source=str(source), source_hashes={n: sha(source/n) for n in
            ("manifest.json", "analysis_data.json", "summary.json", "ranking.csv", "status.json")},
        cell_digest=hashes.hexdigest(), counts=dict(counts),
        map_diversity={k: dict(unique=len(v), max_repeats=max(v.values())) for k, v in maps.items()},
        side_pairs=len(paired_maps), equal_initial_map_pairs=sum(len(v)==2 and v[0]==v[1] for v in paired_maps.values()),
        reconciled_models=rows, reconciled_pairs=len(pairs), cases=terminal_cases,
        coverage="all cells + attempt results + initial maps + final chunks; intermediate chunks not reread",
        official_winner_reconstructed=False, new_matches_run=0, source_results_modified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"logs/shared/agent_league/version_best_ties_league")
    parser.add_argument("--output", type=Path, default=ROOT/"logs/v11/reports/league_audit.json")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        parser.error("output must be outside the source league")
    result = audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "counts": result["counts"],
                      "reconciled_pairs": result["reconciled_pairs"]}, ensure_ascii=False))
