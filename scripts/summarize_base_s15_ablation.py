#!/usr/bin/env python3
"""Compare battery-only and special-item paired series and record promotion decision."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl.logging_schema import validate_series_log, write_json


def variant_summary(series: dict) -> dict:
    summary = series["summary"]
    episodes = series["episodes"]
    event_totals = summary.get("scripted_event_totals", {})
    return {
        "source": series.get("_source"),
        "seeds": series["series_config"]["seeds"],
        "side_swap": series["series_config"]["side_swap"],
        "episodes": summary["episodes"],
        "wins": summary["wins"],
        "draws": summary["draws"],
        "losses": summary["losses"],
        "win_rate": summary["win_rate"],
        "mean_model_score_diff": summary["mean_model_score_diff"],
        "mean_episode_steps": sum(e["episode_length"]["steps"] for e in episodes) / len(episodes),
        "by_model_side": summary["by_model_side"],
        "special_item_event_totals": {
            key: int(event_totals.get(key, 0))
            for key in (
                "special_item_assigned",
                "special_item_pickup",
                "special_item_deposit",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", required=True, type=Path)
    parser.add_argument("--special", required=True, type=Path)
    parser.add_argument("--default-policy-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    battery = json.loads(args.battery.read_text())
    special = json.loads(args.special.read_text())
    validate_series_log(battery)
    validate_series_log(special)
    battery["_source"] = str(args.battery)
    special["_source"] = str(args.special)
    if battery["series_config"]["seeds"] != special["series_config"]["seeds"]:
        raise ValueError("ablation series must use identical seeds")
    if not battery["series_config"]["side_swap"] or not special["series_config"]["side_swap"]:
        raise ValueError("both ablation series must use side swap")

    battery_by_key = {
        (episode["seed"], episode["side_assignment"]["model_team"]): episode
        for episode in battery["episodes"]
    }
    special_by_key = {
        (episode["seed"], episode["side_assignment"]["model_team"]): episode
        for episode in special["episodes"]
    }
    if set(battery_by_key) != set(special_by_key):
        raise ValueError("ablation episodes do not share seed/team keys")
    comparisons = []
    for seed, team in sorted(battery_by_key):
        baseline = battery_by_key[(seed, team)]
        candidate = special_by_key[(seed, team)]
        comparisons.append(
            {
                "seed": seed,
                "model_team": team,
                "model_side": "A" if team == 0 else "B",
                "battery_result": baseline["model_result"],
                "special_result": candidate["model_result"],
                "battery_score_diff": baseline["score"]["model_minus_opponent"],
                "special_score_diff": candidate["score"]["model_minus_opponent"],
                "special_minus_battery_score_diff": (
                    candidate["score"]["model_minus_opponent"]
                    - baseline["score"]["model_minus_opponent"]
                ),
                "battery_steps": baseline["episode_length"]["steps"],
                "special_steps": candidate["episode_length"]["steps"],
                "special_minus_battery_steps": (
                    candidate["episode_length"]["steps"]
                    - baseline["episode_length"]["steps"]
                ),
            }
        )

    battery_summary = variant_summary(battery)
    special_summary = variant_summary(special)
    win_rate_delta = special_summary["win_rate"] - battery_summary["win_rate"]
    score_diff_delta = (
        special_summary["mean_model_score_diff"]
        - battery_summary["mean_model_score_diff"]
    )
    mean_steps_delta = (
        special_summary["mean_episode_steps"] - battery_summary["mean_episode_steps"]
    )
    # The project contract says to retain special items only on a real gain.
    # Require no win-rate regression and a strictly better score margin.
    promote = win_rate_delta >= 0.0 and score_diff_delta > 0.0
    default_config = json.loads(args.default_policy_config.read_text())
    if bool(default_config["special_items"]) != promote:
        raise ValueError("default policy config disagrees with ablation promotion decision")
    payload = {
        "schema_version": "blackout.base_s15_special_item_ablation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "battery_only": battery_summary,
        "special_items": special_summary,
        "paired_episode_comparisons": comparisons,
        "delta_special_minus_battery": {
            "win_rate": win_rate_delta,
            "mean_model_score_diff": score_diff_delta,
            "mean_episode_steps": mean_steps_delta,
        },
        "decision": {
            "promotion_rule": "win_rate_delta >= 0 and mean_score_diff_delta > 0",
            "promote_special_items": promote,
            "default_policy_config": str(args.default_policy_config),
            "reason": (
                "special-item variant improved the paired score margin"
                if promote
                else "special-item variant did not improve the paired mean score margin"
            ),
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload["delta_special_minus_battery"], indent=2))
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
