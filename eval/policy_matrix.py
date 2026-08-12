#!/usr/bin/env python3
"""Build and validate a common-seed side-swapped random/scripted/IPPO matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl.logging_schema import validate_series_log, write_json
from eval.evaluator import summarize_episodes
from eval.paired_series import parse_seeds


MATRIX_SCHEMA_VERSION = "blackout.policy_matrix.v1"
REQUIRED_POLICIES = ("random", "scripted", "ippo")


def build_policy_matrix(
    series_by_policy: Mapping[str, Mapping[str, Any]],
    *,
    seeds: list[int],
) -> dict[str, Any]:
    if tuple(sorted(series_by_policy)) != tuple(sorted(REQUIRED_POLICIES)):
        raise ValueError(f"matrix requires policies {REQUIRED_POLICIES}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("matrix seeds must be non-empty and unique")
    results: dict[str, Any] = {}
    opponent_ids: set[str] = set()
    opponent_hashes: set[str] = set()
    opponent_policy_seeds_by_environment_seed: dict[int, set[int]] = {
        seed: set() for seed in seeds
    }
    executable_hashes: set[str] = set()
    for policy in REQUIRED_POLICIES:
        series = series_by_policy[policy]
        validate_series_log(series)
        episodes = [episode for episode in series["episodes"] if episode["seed"] in seeds]
        expected_count = 2 * len(seeds)
        if len(episodes) != expected_count:
            raise ValueError(f"{policy} has {len(episodes)} selected episodes, expected {expected_count}")
        for seed in seeds:
            selected = [episode for episode in episodes if episode["seed"] == seed]
            if {episode["side_assignment"]["model_team"] for episode in selected} != {0, 1}:
                raise ValueError(f"{policy} seed {seed} is not side-swapped")
        policy_opponents = {episode["opponent"]["policy_id"] for episode in episodes}
        if len(policy_opponents) != 1:
            raise ValueError(f"{policy} does not use one fixed opponent")
        opponent_ids.update(policy_opponents)
        opponent_hashes.update(
            episode["opponent"]["checkpoint_sha256"] for episode in episodes
        )
        for episode in episodes:
            opponent_policy_seeds_by_environment_seed[int(episode["seed"])].add(
                int(episode["policy_seeds"]["opponent"])
            )
        executable_hashes.update(
            episode["environment"]["executable_sha256"] for episode in episodes
        )
        results[policy] = {
            "model_policy_id": episodes[0]["model"]["policy_id"],
            "source_series_id": series["series_id"],
            "source_episode_ids": [episode["episode_id"] for episode in episodes],
            "summary": summarize_episodes(episodes),
        }
    if len(opponent_hashes) != 1:
        raise ValueError("matrix policies use different opponent artifacts")
    if any(len(values) != 1 for values in opponent_policy_seeds_by_environment_seed.values()):
        raise ValueError("matrix policies do not share one opponent policy seed per environment seed")
    if len(executable_hashes) != 1:
        raise ValueError("matrix policies use different environment executables")
    payload = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_config": {
            "seeds": seeds,
            "side_swap": True,
            "opponent_id": "random-v1",
            "source_opponent_ids": sorted(opponent_ids),
            "opponent_sha256": next(iter(opponent_hashes)),
            "opponent_policy_seeds": {
                str(seed): next(iter(opponent_policy_seeds_by_environment_seed[seed]))
                for seed in seeds
            },
            "executable_sha256": next(iter(executable_hashes)),
            "winner_source": "terminal_info.winner",
        },
        "results": results,
        "ranking": sorted(
            REQUIRED_POLICIES,
            key=lambda policy: (
                results[policy]["summary"]["win_rate"],
                results[policy]["summary"]["mean_model_score_diff"],
            ),
            reverse=True,
        ),
    }
    validate_policy_matrix(payload)
    return payload


def validate_policy_matrix(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported policy matrix schema")
    if set(payload.get("results", {})) != set(REQUIRED_POLICIES):
        raise ValueError("policy matrix is missing required policies")
    config = payload["matrix_config"]
    if config.get("side_swap") is not True:
        raise ValueError("policy matrix must use side swap")
    if config.get("winner_source") != "terminal_info.winner":
        raise ValueError("policy matrix must use terminal winner")
    seeds = config.get("seeds", [])
    for policy, result in payload["results"].items():
        summary = result["summary"]
        if summary["episodes"] != 2 * len(seeds):
            raise ValueError(f"{policy} matrix episode count mismatch")
        if summary["side_bias_diagnostics"]["evaluator_side_attribution_passed"] is not True:
            raise ValueError(f"{policy} side attribution failed")
        if summary["winner_derived_from_unity_shaping"] is not False:
            raise ValueError(f"{policy} winner used Unity shaping")
    if set(payload.get("ranking", [])) != set(REQUIRED_POLICIES):
        raise ValueError("policy matrix ranking is invalid")


def parse_series(value: str) -> tuple[str, Path]:
    try:
        policy, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("series must use POLICY=PATH") from exc
    if policy not in REQUIRED_POLICIES:
        raise argparse.ArgumentTypeError(f"unknown matrix policy: {policy}")
    return policy, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", type=parse_series, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {policy: json.loads(path.read_text()) for policy, path in args.series}
    matrix = build_policy_matrix(inputs, seeds=args.seeds)
    write_json(args.output, matrix)
    print(json.dumps(matrix["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
