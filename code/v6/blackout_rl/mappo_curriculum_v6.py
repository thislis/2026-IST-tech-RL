"""V6 readiness gates are baseline-relative; the final strength target is fixed."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class V6Stage:
    name: str
    minimum_steps: int
    maximum_steps: int
    opponent_weights: dict[str, float]
    evaluation_opponent: str
    win_rate: float | None
    minimum_side_win_rate: float
    exploration_start: float
    exploration_end: float

    def exploration(self, steps: int):
        progress = min(max(steps/self.maximum_steps, 0), 1)
        return self.exploration_start + progress*(self.exploration_end-self.exploration_start)


DEFAULT_STAGES = (
    V6Stage("planner_preservation", 4096, 4096, {"base_scripted":1.}, "base_scripted", None, .0, 0., 0.),
    V6Stage("balanced_residual", 51200, 200704,
            {"base_scripted":.65, "weak_win70":.25, "full_win70":.10}, "base_scripted", None, .4, .15, .10),
    V6Stage("target_mix", 100352, 600064,
            {"base_scripted":.20, "weak_win70":.30, "full_win70":.50}, "full_win70", .50, .3, .10, .06),
    V6Stage("full_win70", 200704, 1001472,
            {"base_scripted":.10, "full_win70":.90}, "full_win70", .70, .5, .08, .04),
    V6Stage("robust_historical_mix", 200704, 2000896,
            {"full_win70":.80, "historical":.20}, "full_win70", .85, .7, .06, .02),
)


def scaled_stages(scale: float, rollout: int):
    if not math.isfinite(scale) or scale <= 0 or rollout <= 0:
        raise ValueError("positive finite curriculum scale/rollout required")
    return tuple(V6Stage(**{**asdict(s),
        "minimum_steps": max(rollout, math.ceil(s.minimum_steps*scale/rollout)*rollout),
        "maximum_steps": max(rollout, math.ceil(s.maximum_steps*scale/rollout)*rollout)}) for s in DEFAULT_STAGES)


def gate_passed(stage: V6Stage, summary: dict, baseline: dict, *, stage_steps: int):
    if stage_steps < stage.minimum_steps:
        return False
    preservation = stage.name == "planner_preservation"
    win_floor = max(0., baseline["win_rate"]-(0. if preservation else .05)) if stage.win_rate is None else stage.win_rate
    score_floor = baseline["mean_model_score_diff"]-(0.01 if preservation else 5.) if stage.win_rate is None else -5.
    if summary["win_rate"]+1e-8 < win_floor or summary["mean_model_score_diff"] < score_floor:
        return False
    return all(summary["by_model_side"][side]["win_rate"]+1e-8 >= (
        baseline["by_model_side"][side]["win_rate"] if preservation else stage.minimum_side_win_rate)
        for side in ("A", "B"))


def validate_seed_splits(payload):
    if payload.get("schema_version") != "blackout.seed_splits.v6":
        raise ValueError("unsupported v6 seed schema")
    seen = set()
    for name in ("train", "dev", "confirmation", "test"):
        seeds = payload[name]
        if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError(f"invalid {name} seeds")
        if seen & set(seeds):
            raise ValueError(f"seed leakage in {name}")
        seen.update(seeds)
    if payload["dev"] != [3101,3102,3103,3104,3105]:
        raise ValueError("v6 retains the committed ten-game target evaluation")
    if len(payload["confirmation"]) < 15:
        raise ValueError("confirmation requires at least fifteen distinct map seeds")
