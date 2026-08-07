"""Versioned JSON logging schema and artifact hashing for BlackOut evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EPISODE_SCHEMA_VERSION = "blackout.episode.v1"
SERIES_SCHEMA_VERSION = "blackout.paired_series.v1"
SHA256_HEX_LENGTH = 64


def sha256_file(path: str | Path) -> str:
    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    digest = hashlib.sha256()
    with artifact.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} is not hexadecimal") from exc


def validate_episode_log(record: Mapping[str, Any]) -> None:
    """Fail closed when a required PREP-10 episode field is missing or invalid."""
    required = {
        "schema_version",
        "episode_id",
        "pair_id",
        "pair_index",
        "seed",
        "policy_seeds",
        "side_assignment",
        "model",
        "opponent",
        "environment",
        "episode_length",
        "score",
        "winner",
        "model_result",
        "rewards",
        "termination",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"episode log missing fields: {sorted(missing)}")
    if record["schema_version"] != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema: {record['schema_version']}")
    if record["seed"] < 0:
        raise ValueError("seed must be non-negative")
    if record["side_assignment"]["model_team"] not in (0, 1):
        raise ValueError("model_team must be 0 or 1")
    model_team = record["side_assignment"]["model_team"]
    opponent_team = record["side_assignment"]["opponent_team"]
    if opponent_team != 1 - model_team:
        raise ValueError("model and opponent must occupy opposite teams")
    expected_model_side = "A" if model_team == 0 else "B"
    expected_opponent_side = "A" if opponent_team == 0 else "B"
    if record["side_assignment"].get("model_side", expected_model_side) != expected_model_side:
        raise ValueError("model_side does not match model_team")
    if record["side_assignment"].get("opponent_side", expected_opponent_side) != expected_opponent_side:
        raise ValueError("opponent_side does not match opponent_team")
    for actor in ("model", "opponent"):
        validate_sha256(record[actor]["checkpoint_sha256"], f"{actor}.checkpoint_sha256")
    validate_sha256(record["environment"]["executable_sha256"], "environment.executable_sha256")
    if record["episode_length"]["steps"] <= 0:
        raise ValueError("episode length must be positive")
    if record["winner"]["team"] not in (-1, 0, 1):
        raise ValueError("winner.team must be -1, 0, or 1")
    if record["model_result"] not in ("win", "draw", "loss"):
        raise ValueError("model_result must be win/draw/loss")
    score = record["score"]
    if score["team_a"] < 0 or score["team_b"] < 0:
        raise ValueError("scores must be non-negative")
    expected = winner_from_scores(score["team_a"], score["team_b"])
    if "model" in score:
        expected_model_score = score["team_a"] if model_team == 0 else score["team_b"]
        expected_opponent_score = score["team_b"] if model_team == 0 else score["team_a"]
        if score["model"] != expected_model_score or score["opponent"] != expected_opponent_score:
            raise ValueError("model/opponent score does not match side assignment")
        if score["model_minus_opponent"] != score["model"] - score["opponent"]:
            raise ValueError("model score difference is inconsistent")
    if record["model_result"] != model_result(record["winner"]["team"], model_team):
        raise ValueError("model_result does not match terminal winner and model side")
    if record["rewards"].get("unity_shaping_used_for_winner", False) is not False:
        raise ValueError("Unity shaping reward must not determine the winner")
    if record["termination"]["score_is_preterminal"] is False and record["winner"]["team"] != expected:
        raise ValueError("terminal winner does not match final score")


def validate_series_log(record: Mapping[str, Any]) -> None:
    required = {"schema_version", "series_id", "created_at_utc", "episodes", "summary"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"series log missing fields: {sorted(missing)}")
    if record["schema_version"] != SERIES_SCHEMA_VERSION:
        raise ValueError(f"unsupported series schema: {record['schema_version']}")
    for episode in record["episodes"]:
        validate_episode_log(episode)
    if record["summary"]["episodes"] != len(record["episodes"]):
        raise ValueError("summary episode count mismatch")
    result_count = sum(record["summary"][key] for key in ("wins", "draws", "losses"))
    if result_count != len(record["episodes"]):
        raise ValueError("summary W/D/L count mismatch")
    diagnostics = record["summary"].get("side_bias_diagnostics")
    if diagnostics is not None and diagnostics.get("evaluator_side_attribution_passed") is not True:
        raise ValueError("evaluator side-attribution audit failed")
    if record.get("series_config", {}).get("side_swap") is True:
        pairs: dict[str, list[Mapping[str, Any]]] = {}
        for episode in record["episodes"]:
            pairs.setdefault(episode["pair_id"], []).append(episode)
        for pair_id, episodes in pairs.items():
            teams = {episode["side_assignment"]["model_team"] for episode in episodes}
            indices = {episode["pair_index"] for episode in episodes}
            seeds = {episode["seed"] for episode in episodes}
            if len(episodes) != 2 or teams != {0, 1} or indices != {0, 1} or len(seeds) != 1:
                raise ValueError(f"invalid side-swapped pair: {pair_id}")


def winner_from_scores(team_a: int, team_b: int) -> int:
    return 0 if team_a > team_b else 1 if team_b > team_a else -1


def model_result(winner: int, model_team: int) -> str:
    if winner == -1:
        return "draw"
    return "win" if winner == model_team else "loss"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
