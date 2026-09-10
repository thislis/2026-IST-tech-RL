#!/usr/bin/env python3
"""V6 coordinated residual PPO. --check never opens Unity or updates parameters."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackout_rl.env import ContractBlackOutEnv
from blackout_rl.frozen_opponent import FrozenScriptedOpponent
from blackout_rl.logging_schema import validate_series_log
from blackout_rl.model_contract import load_checkpoint
from blackout_rl.mappo_v6 import V6Model, SCHEMA_VERSION, ROLES
from blackout_rl.mappo_v6_training import FailureSeedSampler, V6Collector, update_v6
from blackout_rl.mappo_v6_artifacts import (atomic_json, sha256, source_manifest, archive_sources,
    save_v6_checkpoint, load_v6_checkpoint, export_v6)
from blackout_rl.mappo_curriculum_v6 import scaled_stages, gate_passed, validate_seed_splits
from blackout_rl.policy import DeterministicCheckpointPolicy, PolicyArtifact
from blackout_rl.ppo import PPOConfig
from blackout_rl.training_reward import TrainingRewardConfig, RewardMode
from blackout_rl.evaluation_protocol import paired_seed_bootstrap_ci
from eval.evaluator import evaluate_episode, summarize_episodes, _executable_in

GAME_COMMIT = "d2220a7d01be88d413f551efd529f4758833be8b"
PYTHON_API_COMMIT = "6ba7d9993cf1bdefe1ed480c8efbcabcb923f539"


def now():
    return datetime.now(timezone.utc).isoformat()


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version":SCHEMA_VERSION, "created_at_utc":now(), **record},
                                ensure_ascii=False, allow_nan=False)+"\n")


def recover_log_tails(log_dir, offsets):
    """Keep crash-ahead records as recovery files before returning to saved offsets."""
    recovered = []
    for name, offset in offsets.items():
        if name not in ("training.jsonl", "training_episodes.jsonl"):
            raise ValueError("invalid resume log name")
        path = log_dir/name
        if not path.exists() and offset == 0:
            continue
        if not path.exists() or path.stat().st_size < offset:
            raise ValueError(f"resume log is shorter than checkpoint offset: {name}")
        if path.stat().st_size > offset:
            backup = log_dir/f"recovery_{uuid.uuid4().hex}_{name}"
            with path.open("rb") as stream:
                stream.seek(offset)
                with backup.open("xb") as output:
                    shutil.copyfileobj(stream, output)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            recovered.append(str(backup))
    return recovered


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT/"builds/BlackOut.app")
    parser.add_argument("--initial-checkpoint", type=Path, default=ROOT/"checkpoints/win_70_vs_scripted.pt")
    parser.add_argument("--opponent-checkpoint", type=Path, default=ROOT/"checkpoints/win_70_vs_scripted.pt")
    parser.add_argument("--seed-splits", type=Path, default=ROOT/"configs/seed_splits_v6.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT/"logs/mappo_planner_residual_v6")
    for flag, filename in (("latest", "latest"), ("target-best", "target_best"), ("stage-best", "stage_best")):
        parser.add_argument(f"--{flag}-checkpoint", type=Path, default=ROOT/f"checkpoints/mappo_planner_residual_v6_{filename}.pt")
    parser.add_argument("--target-checkpoint", type=Path, default=ROOT/"checkpoints/mappo_win_85_vs_win70_v6.pt")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT/"checkpoints/mappo_v6_snapshots")
    parser.add_argument("--export-dir", type=Path, default=ROOT/"submission/v6")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--check", action="store_true", help="read-only preflight; no Unity, no training")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=10600)
    parser.add_argument("--max-env-steps", type=int, default=4_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--curriculum-scale", type=float, default=1.)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--initial-override-probability", type=float, default=.10)
    parser.add_argument("--gamma", type=float, default=.9995)
    parser.add_argument("--gae-lambda", type=float, default=.99)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--entropy-coef", type=float, default=.003)
    parser.add_argument("--target-kl", type=float, default=.02)
    parser.add_argument("--score-delta-weight", type=float, default=1.)
    parser.add_argument("--unity-shaping-weight", type=float, default=0.)
    parser.add_argument("--terminal-win-reward", type=float, default=1.)
    parser.add_argument("--rollback-drop-tolerance", type=float, default=.10)
    parser.add_argument("--max-rollbacks", type=int, default=3)
    parser.add_argument("--time-scale", type=float, default=50.)
    parser.add_argument("--eval-time-scale", type=float, default=100.)
    parser.add_argument("--eval-workers", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=22000)
    return parser


def validate_args(args):
    if args.resume and args.resume_latest:
        raise ValueError("--resume and --resume-latest are mutually exclusive")
    for name in ("max_env_steps", "rollout_steps", "curriculum_scale", "eval_every", "save_every", "learning_rate",
                 "update_epochs", "minibatch_size", "target_kl", "time_scale", "eval_time_scale", "eval_workers", "max_episode_steps"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("entropy_coef", "score_delta_weight", "unity_shaping_weight", "terminal_win_reward", "seed", "max_rollbacks"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    for name in ("gamma", "gae_lambda", "rollback_drop_tolerance"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if not 0 < args.initial_override_probability < .5:
        raise ValueError("initial override probability must be in (0,.5)")
    if args.max_env_steps < args.rollout_steps:
        raise ValueError("max-env-steps must cover a rollout")
    if args.max_episode_steps < 21003:
        raise ValueError("max-episode-steps must cover the 420-second game (21003 steps)")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    paths = [args.latest_checkpoint, args.target_best_checkpoint, args.stage_best_checkpoint, args.target_checkpoint]
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != 4 or set(resolved) & {args.initial_checkpoint.resolve(), args.opponent_checkpoint.resolve()}:
        raise ValueError("output checkpoints must be distinct and must not overwrite input checkpoints")


def preflight(args):
    validate_args(args)
    splits = json.loads(args.seed_splits.read_text())
    validate_seed_splits(splits)
    stages = scaled_stages(args.curriculum_scale, args.rollout_steps)
    for path in (args.initial_checkpoint, args.opponent_checkpoint, _executable_in(args.build)):
        if not path.is_file():
            raise FileNotFoundError(path)
    with torch.random.fork_rng(devices=[]):
        _, initial = load_checkpoint(args.initial_checkpoint)
        _, opponent = load_checkpoint(args.opponent_checkpoint)
    expected_guard = {"version":"scripted_counter_v1", "mode":"planner_override", "chase_radius_cells":48}
    for payload in (initial, opponent):
        guard = payload.get("inference_guardrail", {})
        if any(guard.get(k) != v for k,v in expected_guard.items()) or guard.get("roles") != [r.value for r in ROLES]:
            raise ValueError("v6 baseline contract requires the audited 3-worker/2-guard full planner")
    if initial["model_config"]["recurrent_version"] != "none":
        raise ValueError("recurrent initial encoders are unsupported")
    resume = args.resume or (args.latest_checkpoint if args.resume_latest else None)
    if resume is None:
        conflicts = [p for p in (args.latest_checkpoint, args.target_best_checkpoint, args.stage_best_checkpoint,
            args.target_checkpoint, args.log_dir/"training.jsonl", args.log_dir/"run_summary.json", args.log_dir/"source_snapshot.zip") if p.exists()]
        if args.snapshot_dir.exists() and any(args.snapshot_dir.iterdir()):
            conflicts.append(args.snapshot_dir)
        if args.export_dir.exists() and any(args.export_dir.iterdir()):
            conflicts.append(args.export_dir)
        if conflicts:
            raise FileExistsError("fresh-run outputs already exist; use --resume-latest or new output paths: "+str(conflicts))
    elif not resume.is_file():
        raise FileNotFoundError(resume)
    source = source_manifest(ROOT)
    source.update(initial_checkpoint_sha256=sha256(args.initial_checkpoint), opponent_sha256=sha256(args.opponent_checkpoint),
                  executable_sha256=sha256(_executable_in(args.build)))
    if source["executable_sha256"] != "49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69":
        raise ValueError("Unity executable does not match the audited game contract")
    config = {k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()
              if k not in ("check", "resume", "resume_latest")}
    config.update(seed_splits=splits, stages=[asdict(s) for s in stages],
                  algorithm="coordinated_team_residual_PPO_with_centralized_critic",
                  actor_encoder_frozen=True, behavior_distribution="41_way_masked_mixture",
                  environment_resume="new_episode; in-flight Unity state is not serializable")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("v6_schema") != SCHEMA_VERSION:
            raise ValueError("cannot resume an earlier generation as v6")
        old = saved["mappo_v6_training"]["config"]
        for key in config:
            if key not in ("max_env_steps", "eval_workers") and old.get(key) != config[key]:
                raise ValueError(f"resume config mismatch: {key}")
        if args.max_env_steps < saved["mappo_v6_training"]["global_step"]:
            raise ValueError("resume max-env-steps precedes saved progress")
        for key in ("source_sha256", "initial_checkpoint_sha256", "opponent_sha256", "executable_sha256"):
            if saved["source"].get(key) != source[key]:
                raise ValueError(f"resume provenance mismatch: {key}")
        # Logs and selected checkpoints are part of resume, not optional hints.
        runtime = saved["mappo_v6_training"]["runtime_state"]
        for name, identity in runtime.get("artifact_identities", {}).items():
            path = Path(identity["path"])
            if not path.is_file() or sha256(path) != identity["sha256"]:
                raise ValueError(f"resume artifact missing/changed: {name}")
        if not (args.log_dir/"training.jsonl").is_file():
            raise ValueError("resume requires its original training.jsonl")
        for name, offset in runtime.get("log_offsets", {}).items():
            path = args.log_dir/name
            if (name not in ("training.jsonl", "training_episodes.jsonl")
                or (not path.is_file() and offset != 0) or (path.is_file() and path.stat().st_size < offset)):
                raise ValueError("resume log is missing or shorter than its committed offset")
        terminal_status = runtime.get("terminal_status")
        if terminal_status in ("target_reached", "stage_blocked", "rollback_limit"):
            raise ValueError(f"run already ended with {terminal_status}; unchanged resume cannot solve this condition")
    return splits, stages, source, config, resume


class OpponentMixture:
    def __init__(self, team, target, weights, rng, snapshots):
        self.team, self.target, self.weights = team, target, weights
        self.rng, self.snapshots = rng, snapshots
        self.selected = None
        self.selection_counts = {}

    def reset(self):
        names = list(self.weights)
        probabilities = np.asarray([self.weights[n] for n in names])
        selected = str(self.rng.choice(names, p=probabilities/probabilities.sum()))
        if selected == "historical" and not self.snapshots:
            selected = "full_win70"
        if selected in ("full_win70", "historical"):
            path = self.target if selected == "full_win70" else Path(str(self.rng.choice(self.snapshots)))
            with torch.random.fork_rng(devices=[]):
                self.policy = DeterministicCheckpointPolicy(path, team=self.team)
        else:
            kwargs = {} if selected == "base_scripted" else dict(roles=ROLES, chase_radius_cells=12, policy_id="weak-win70-v1")
            self.policy = FrozenScriptedOpponent(self.team, **kwargs)
        self.policy.reset()
        self.selected = selected
        self.selection_counts[selected] = self.selection_counts.get(selected, 0)+1

    def act(self, obs, agents):
        return self.policy.act(obs, agents)


def evaluation_job(job):
    torch.set_num_threads(1)
    env = ContractBlackOutEnv(env_path=job["build"], background=True, no_graphics=False, time_scale=job["time_scale"])
    episodes = []
    try:
        for team in (0,1):
            candidate = DeterministicCheckpointPolicy(job["candidate"], team=team, seed=106000+job["seed"])
            if job["opponent_id"] == "full_win70":
                opponent = DeterministicCheckpointPolicy(job["opponent"], team=1-team, seed=107000+job["seed"])
                artifact = PolicyArtifact.from_file("full-win70", "hybrid-planner", job["opponent"])
            else:
                opponent = FrozenScriptedOpponent(1-team, seed=107000+job["seed"])
                artifact = PolicyArtifact.from_file("scripted-battery-v1", "scripted-policy-config", ROOT/"configs/policies/scripted_battery_v1.json")
            episodes.append(evaluate_episode(env, build=job["build"], game_commit=GAME_COMMIT,
                python_api_commit=PYTHON_API_COMMIT, seed=job["seed"], model_team=team,
                model_policy=candidate, opponent_policy=opponent,
                model_artifact=PolicyArtifact.from_file("mappo-planner-residual-v6", "coordinated-team-residual", job["candidate"]),
                opponent_artifact=artifact, pair_id=f"v6-{job['split']}-{job['opponent_id']}-{job['seed']}",
                pair_index=team, max_steps=job["max_steps"]))
    finally:
        env.close()
    return episodes


def evaluate(args, checkpoint, seeds, opponent_id, split, step, label):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation requires distinct environment seeds")
    jobs = [dict(build=str(args.build), candidate=str(checkpoint), opponent=str(args.opponent_checkpoint),
        seed=seed, opponent_id=opponent_id, split=split, time_scale=args.eval_time_scale,
        max_steps=args.max_episode_steps) for seed in seeds]
    if args.eval_workers == 1:
        pairs = [evaluation_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.eval_workers,len(jobs))) as pool:
            pairs = list(pool.map(evaluation_job, jobs))
    episodes = [row for pair in pairs for row in pair]
    summary = summarize_episodes(episodes)
    output = args.log_dir/f"{label}_eval_step_{step}.json"
    payload = dict(schema_version="blackout.paired_series.v1", series_id=str(uuid.uuid4()), created_at_utc=now(),
        series_config=dict(seeds=seeds, split=split, held_out=True, side_swap=True, opponent=opponent_id,
            training_global_step=step, unique_map_seeds=len(set(seeds)), checkpoint_sha256=sha256(checkpoint)),
        episodes=episodes, summary=summary)
    validate_series_log(payload)
    atomic_json(output, payload)
    enriched = {**summary, "global_step":step, "evaluation_log":str(output.resolve()),
        "win_rate_ci":paired_seed_bootstrap_ci(episodes, metric="win_rate", seed=10600).to_dict(),
        "score_diff_ci":paired_seed_bootstrap_ci(episodes, metric="score_diff", seed=10601).to_dict()}
    # Percentile bootstrap collapses at all-win/all-loss samples. Also report a
    # distribution-free bound on the mean of independent seed-pair scores [0,1].
    # This assumes seeds represent independent draws from the target map family.
    radius = math.sqrt(math.log(40)/(2*len(seeds)))
    enriched["win_rate_seed_bound"] = dict(method="hoeffding_seed_pairs", confidence=.95,
        lower=max(0., summary["win_rate"]-radius), upper=min(1., summary["win_rate"]+radius), pairs=len(seeds))
    payload["uncertainty"] = {k:enriched[k] for k in ("win_rate_ci", "score_diff_ci", "win_rate_seed_bound")}
    atomic_json(output, payload)
    print(f"evaluation step={step} split={split} opponent={opponent_id} wins={summary['wins']}/{summary['episodes']} score_diff={summary['mean_model_score_diff']:.2f}", flush=True)
    return enriched


def run(args, preflight_result):
    splits, stages, source, config, resume = preflight_result
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if resume:
        model, optimizer, payload = load_v6_checkpoint(resume, args.device)
        saved = payload["mappo_v6_training"]
        state = saved["runtime_state"]
        step, update, run_id = saved["global_step"], saved["update"], saved["run_id"]
        recovered = recover_log_tails(args.log_dir, state.get("log_offsets", {}))
        for name, alias in (("target_best", args.target_best_checkpoint), ("stage_best", args.stage_best_checkpoint)):
            identity = state.get("artifact_identities", {}).get(name)
            if identity and (not alias.exists() or sha256(alias) != identity["sha256"]):
                temporary = alias.with_suffix(alias.suffix+".tmp")
                alias.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(identity["path"], temporary)
                temporary.replace(alias)
        for side, episode in state.get("inflight_episodes", {}).items():
            if episode["steps"]:
                state.setdefault("collector_counts", {}).setdefault(side, {}).setdefault("discarded_partial_episodes", 0)
                state["collector_counts"][side]["discarded_partial_episodes"] += 1
    else:
        recovered = []
        actor, _ = load_checkpoint(args.initial_checkpoint)
        model = V6Model(actor.actor_critic, args.initial_override_probability).to(args.device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        step, update, run_id = 0, 0, str(uuid.uuid4())
        state = dict(stage_index=0, stage_start_step=0, rollback_count=0, snapshots=[], baseline={},
                     best_target_evaluation=None, best_stage_evaluation=None, side_updates=[0,0], artifact_identities={})
    rng = np.random.default_rng(args.seed)
    if "rng_state" in state:
        rng.bit_generator.state = state["rng_state"]
    samplers = [FailureSeedSampler(splits["train"], args.seed+i, state.get("samplers", {}).get(str(i))) for i in (0,1)]
    collectors = {}
    interrupted = [False]
    started = time.monotonic()
    elapsed_before = state.get("elapsed_seconds", 0.)
    log = args.log_dir/"training.jsonl"
    log.touch(exist_ok=True)
    last_target = state.get("latest_target_evaluation")
    last_stage = state.get("latest_stage_evaluation")
    status = "running"
    last_eval_step = state.get("last_eval_step", -1)
    reward_config = TrainingRewardConfig(mode=RewardMode.COMBINED, score_delta_weight=args.score_delta_weight,
        unity_shaping_weight=args.unity_shaping_weight, terminal_win_reward=args.terminal_win_reward)
    ppo_config = PPOConfig(update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
                           entropy_coef=args.entropy_coef, target_kl=args.target_kl)
    old_handlers = {sig:signal.signal(sig, lambda *_: interrupted.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    collection_start_count = 0

    def runtime():
        state["samplers"] = {str(i):samplers[i].state_dict() for i in (0,1)}
        state["rng_state"] = rng.bit_generator.state
        state["elapsed_seconds"] = elapsed_before+time.monotonic()-started
        counts = state.setdefault("collector_counts", {})
        for i,c in collectors.items():
            counts[str(i)] = dict(c.counts)
        state["latest_target_evaluation"], state["latest_stage_evaluation"] = last_target, last_stage
        state["last_eval_step"] = last_eval_step
        state["log_offsets"] = {name:((args.log_dir/name).stat().st_size if (args.log_dir/name).exists() else 0)
            for name in ("training.jsonl", "training_episodes.jsonl")}
        state["terminal_status"] = status
        state["inflight_episodes"] = {str(i):dict(seed=c.seed, steps=c.episode_steps) for i,c in collectors.items() if c.episode_steps}
        return state

    def save(path):
        save_v6_checkpoint(path, model, optimizer, training=dict(schema_version=SCHEMA_VERSION,
            run_id=run_id, global_step=step, update=update, config=config, runtime_state=runtime()), source=source)

    def identify(name, path):
        state["artifact_identities"][name] = {"path":str(path.resolve()), "sha256":sha256(path)}

    def save_best(name, alias):
        # Resume references immutable bytes even if a crash follows an alias update.
        immutable = args.snapshot_dir/f"{name}_step_{step}_{uuid.uuid4().hex[:8]}.pt"
        save(immutable)
        temporary = alias.with_suffix(alias.suffix+".tmp")
        alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(immutable, temporary)
        temporary.replace(alias)
        identify(name, immutable)

    def summary(reason=None):
        atomic_json(args.log_dir/"run_summary.json", dict(schema_version=SCHEMA_VERSION, run_id=run_id,
            updated_at_utc=now(), global_step=step, update=update, status=status, reason=reason,
            stage=stages[state["stage_index"]].name, stage_steps=step-state["stage_start_step"],
            target_win_rate=.85, target_wins=9, target_reached=status=="target_reached",
            latest_target_evaluation=last_target, latest_stage_evaluation=last_stage,
            best_target_evaluation=state["best_target_evaluation"], best_stage_evaluation=state["best_stage_evaluation"],
            runtime_state=runtime(), config=config, source=source, elapsed_seconds=state["elapsed_seconds"]))

    def close_collectors():
        closing = list(collectors.items())
        collectors.clear()
        for i, c in closing:
            if c.episode_steps:
                c.counts["discarded_partial_episodes"] += 1
                append_record(log, dict(record_type="partial_episode_discarded", global_step=step,
                    learner_team=i, seed=c.seed, episode_steps=c.episode_steps))
            state.setdefault("collector_counts", {})[str(i)] = dict(c.counts)
            try:
                c.env.close()
            except Exception as exc:
                append_record(log, dict(record_type="environment_close_warning", global_step=step,
                                        learner_team=i, error=repr(exc)))

    def build_collectors():
        stage = stages[state["stage_index"]]
        for i in (0,1):
            env = ContractBlackOutEnv(env_path=str(args.build), background=True, no_graphics=False, time_scale=args.time_scale)
            mixture = OpponentMixture(1-i, args.opponent_checkpoint, stage.opponent_weights, rng, state["snapshots"])
            def episode(record, team=i):
                append_record(args.log_dir/"training_episodes.jsonl", dict(record_type="training_episode", run_id=run_id,
                    global_step=step+collectors[team].counts["environment_steps"]-collection_start_count,
                    stage=stage.name, opponent=collectors[team].opponent.selected, **record))
            collector = V6Collector(env, model, mixture, team=i, sampler=samplers[i], reward_config=reward_config,
                max_episode_steps=args.max_episode_steps, episode_callback=episode)
            collector.counts.update(state.get("collector_counts", {}).get(str(i), {}))
            collectors[i] = collector

    def evaluate_pair(opponent, split, label, checkpoint=None):
        return evaluate(args, checkpoint or args.latest_checkpoint, splits[split], opponent, split, step, label)

    def is_better(candidate, best):
        return best is None or (candidate["win_rate"], candidate["mean_model_score_diff"]) > (best["win_rate"], best["mean_model_score_diff"])

    def finish_selected_target():
        nonlocal status
        exported = export_v6(args.target_checkpoint, args.export_dir)
        atomic_json(args.log_dir/"submission_export.json", exported)
        state["submission_export"] = exported
        if "final_test_evaluation" not in state:
            # Report only: this split never changes selection or promotion.
            state["final_test_evaluation"] = evaluate_pair("full_win70", "test", "final_test",
                                                           checkpoint=args.target_checkpoint)
        status = "target_reached"

    try:
        if not resume:
            archive_sources(ROOT, args.log_dir/"source_snapshot.zip", source)
            atomic_json(args.log_dir/"run_config.json", config)
        append_record(log, dict(record_type="resume" if resume else "start", run_id=run_id, global_step=step,
            source=source, recovered_log_tails=recovered,
            discarded_inflight_on_resume=state.get("inflight_episodes", {}) if resume else {}))
        save(args.latest_checkpoint)
        # Baselines are measured on exactly the same dev AND distinct confirmation
        # seeds as their later candidate comparisons; no hard-coded 8/10 assumption.
        for opponent in ("base_scripted", "full_win70"):
            for split in ("dev", "confirmation"):
                key = opponent+":"+split
                if key not in state["baseline"]:
                    state["baseline"][key] = evaluate_pair(opponent, split, "baseline_"+opponent+"_"+split,
                                                          checkpoint=args.initial_checkpoint)
                    save(args.latest_checkpoint)
                    summary()
        if state["best_target_evaluation"] is None:
            state["best_target_evaluation"] = state["baseline"]["full_win70:dev"]
            save_best("target_best", args.target_best_checkpoint)
        save(args.latest_checkpoint)
        if state.get("target_qualified"):
            finish_selected_target()
            save(args.latest_checkpoint)
            summary()
            return 0
        if not interrupted[0]:
            build_collectors()
        while step < args.max_env_steps and not interrupted[0]:
            stage = stages[state["stage_index"]]
            stage_steps = step-state["stage_start_step"]
            # B-side extra exposure: A,B,B,A,B pattern; both persistent collectors
            # still finish episodes. This is declared 40:60, not labelled balanced.
            side = (0,1,1,0,1)[update % 5]
            collector = collectors[side]
            before = dict(collector.counts)
            collection_start_count = collector.counts["environment_steps"]
            exploration = stage.exploration(stage_steps)
            length = min(args.rollout_steps, args.max_env_steps-step, stage.maximum_steps-stage_steps)
            if length <= 0:
                status = "stage_blocked"
                break
            batch, diagnostics = collector.collect(length, exploration=exploration, gamma=args.gamma, gae_lambda=args.gae_lambda,
                                                    force_planner=stage.name == "planner_preservation")
            use_ppo = stage.name != "planner_preservation"
            metrics = update_v6(model, optimizer, batch, ppo_config) if use_ppo else dict(
                policy_loss=0., value_loss=0., entropy=0., approximate_kl=0., clip_fraction=0.,
                gradient_norm=0., explained_variance=None, epochs_completed=0, minibatches=0,
                samples=length, early_stopped=False)
            step += length
            update += 1
            state["side_updates"][side] += 1
            rollout = {k:collector.counts[k]-before[k] for k in collector.counts}
            append_record(log, dict(run_id=run_id, update=update, global_step=step, stage=stage.name,
                stage_steps=step-state["stage_start_step"], learner_team=side, use_ppo=use_ppo,
                exploration_floor=exploration, **diagnostics, rollout=rollout, ppo=metrics,
                mean_return=float(batch["return_"].mean()), opponent_selection_counts=collector.opponent.selection_counts,
                seed_probabilities=collector.sampler.probabilities().tolist()))
            print(f"update={update} step={step} side={side} stage={stage.name} override={diagnostics['team_override_rate']:.4f}", flush=True)
            need_eval = (last_eval_step < 0 or step-last_eval_step >= args.eval_every
                or step-state["stage_start_step"] >= stage.maximum_steps
                or (stage.name == "planner_preservation" and step-state["stage_start_step"] >= stage.minimum_steps)
                or step >= args.max_env_steps)
            if step//args.save_every != (step-length)//args.save_every or need_eval or interrupted[0]:
                save(args.latest_checkpoint)
                summary()
            if interrupted[0]:
                break
            if not need_eval:
                continue
            last_target = evaluate_pair("full_win70", "dev", "target")
            last_stage = last_target if stage.evaluation_opponent == "full_win70" else evaluate_pair("base_scripted", "dev", "stage_"+stage.name)
            last_eval_step = step
            baseline = state["baseline"]["full_win70:dev"]
            floor = max(0., baseline["win_rate"]-args.rollback_drop_tolerance)
            if last_target["win_rate"] < floor:
                confirmed = evaluate_pair("full_win70", "confirmation", "confirmation_rollback")
                confirm_floor = max(0., state["baseline"]["full_win70:confirmation"]["win_rate"]-args.rollback_drop_tolerance)
                if confirmed["win_rate"] < confirm_floor:
                    restore_path = state["artifact_identities"]["target_best"]["path"]
                    restored, restored_optimizer, _ = load_v6_checkpoint(restore_path, args.device, restore_rng=False)
                    model.load_state_dict(restored.state_dict())
                    optimizer.load_state_dict(restored_optimizer.state_dict())
                    state["rollback_count"] += 1
                    append_record(log, dict(record_type="target_rollback", global_step=step, stage=stage.name,
                        reason="confirmed_target_below_baseline", confirmed_evaluation=confirmed,
                        restored_checkpoint=restore_path, rollback_count=state["rollback_count"]))
                    last_target = state["best_target_evaluation"]
                    last_stage = None
                    close_collectors()
                    save(args.latest_checkpoint)
                    if state["rollback_count"] >= args.max_rollbacks:
                        status = "rollback_limit"
                        break
                    if step-state["stage_start_step"] >= stage.maximum_steps:
                        status = "stage_blocked"
                        break
                    build_collectors()
                    summary("rollback completed; target result refers to the restored checkpoint's evaluation")
                    continue
            # Never replace a confirmed target-best with an unconfirmed dev spike.
            if is_better(last_target, state["best_target_evaluation"]):
                confirmed = evaluate_pair("full_win70", "confirmation", "confirmation_target_best")
                previous = state.get("best_target_confirmation", state["baseline"]["full_win70:confirmation"])
                if (confirmed["win_rate"], confirmed["mean_model_score_diff"]) >= (previous["win_rate"], previous["mean_model_score_diff"]):
                    state["best_target_evaluation"], state["best_target_confirmation"] = last_target, confirmed
                    save_best("target_best", args.target_best_checkpoint)
            if is_better(last_stage, state["best_stage_evaluation"]):
                state["best_stage_evaluation"] = last_stage
                save_best("stage_best", args.stage_best_checkpoint)
            if last_target["wins"] >= 9 and last_target["episodes"] == 10:
                confirmed = evaluate_pair("full_win70", "confirmation", "confirmation_target")
                if confirmed["win_rate"] >= .85 and all(confirmed["by_model_side"][s]["win_rate"] >= .7 for s in ("A","B")):
                    state["target_qualified"] = True
                    state["target_confirmation"] = confirmed
                    save(args.target_checkpoint)
                    identify("target", args.target_checkpoint)
                    save(args.latest_checkpoint)
                    finish_selected_target()
                    break
            baseline_stage = state["baseline"][stage.evaluation_opponent+":dev"]
            passed = gate_passed(stage, last_stage, baseline_stage, stage_steps=step-state["stage_start_step"])
            if passed:
                confirmed = evaluate_pair(stage.evaluation_opponent, "confirmation", "confirmation_stage_"+stage.name)
                passed = gate_passed(stage, confirmed, state["baseline"][stage.evaluation_opponent+":confirmation"],
                                     stage_steps=step-state["stage_start_step"])
            if passed and state["stage_index"]+1 < len(stages):
                snapshot = args.snapshot_dir/f"stage_{state['stage_index']}_step_{step}.pt"
                save(snapshot)
                state["snapshots"].append(str(snapshot.resolve()))
                identify(f"snapshot_{state['stage_index']}", snapshot)
                close_collectors()
                state["stage_index"] += 1
                state["stage_start_step"] = step
                state["best_stage_evaluation"] = None
                append_record(log, dict(record_type="stage_transition", global_step=step, previous_stage=stage.name,
                    stage=stages[state["stage_index"]].name, confirmed_evaluation=confirmed))
                build_collectors()
            elif step-state["stage_start_step"] >= stage.maximum_steps:
                status = "stage_blocked"
                append_record(log, dict(record_type="stage_blocked", global_step=step, stage=stage.name,
                    reason="maximum_stage_budget_without_confirmed_gate", stage_evaluation=last_stage))
                break
            save(args.latest_checkpoint)
            summary()
        if status == "running":
            status = "interrupted" if interrupted[0] else "budget_exhausted"
        close_collectors()
        save(args.latest_checkpoint)
        summary()
        print(f"status={status} step={step}", flush=True)
        return 0 if status in ("target_reached", "budget_exhausted", "interrupted") else 3
    except Exception as exc:
        status = "failed"
        append_record(log, dict(record_type="failure", global_step=step, error=repr(exc)))
        summary(repr(exc))
        # Keep the last atomically saved finite model; never overwrite it with a
        # potentially non-finite or partly updated model from an exception path.
        raise
    finally:
        close_collectors()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    args = build_parser().parse_args()
    checked = preflight(args)
    if args.check:
        print(json.dumps({"status":"preflight_passed", "training_started":False,
            "config":checked[3], "source_sha256":checked[2]["source_sha256"]}, indent=2))
        return 0
    args.log_dir.mkdir(parents=True, exist_ok=True)
    # Process-owned advisory lock survives neither exit nor SIGKILL; stale PID
    # files cannot permanently block resume, and reused PIDs are not killed.
    with ExitStack() as stack:
        lock = stack.enter_context((args.log_dir/"run.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another v6 trainer already owns this log directory")
        for output in sorted((args.latest_checkpoint, args.target_best_checkpoint,
                              args.stage_best_checkpoint, args.target_checkpoint)):
            output.parent.mkdir(parents=True, exist_ok=True)
            output_lock = stack.enter_context(output.with_suffix(output.suffix+".lock").open("a"))
            try:
                fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"another trainer owns checkpoint output {output}")
        # Re-check outputs after acquiring the lock to close launch races.
        checked = preflight(args)
        (args.log_dir/"training.pid").write_text(str(os.getpid())+"\n")
        try:
            return run(args, checked)
        finally:
            (args.log_dir/"training.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
