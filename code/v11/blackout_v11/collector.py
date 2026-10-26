"""Canonical public observations, one frozen policy, one natural full episode."""
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .io import atomic_json
from .rewards import OBJECTIVE, potential, terminal_outcome
from .trajectory import EpisodeWriter, canonical_names, pack
from .behavior import SituationRecorder, decision_metadata


def observation_hash(obs):
    h = hashlib.sha256()
    for name in sorted(obs):
        h.update(name.encode())
        for key in ("vector", "graphic"):
            h.update(obs[name][key].tobytes())
    return h.hexdigest()


class Behavior:
    def __init__(self):
        self.steps = self.pickups = self.deliveries = self.respawns = self.moving = self.stuck = 0
        self.score_gain = 0.

    def observe(self, before, after, terminal):
        self.steps += 1
        if terminal:
            return
        a, b = before[0, :90].reshape(10, 9), after[0, :90].reshape(10, 9)
        own = a[:, 2] > 0
        movement = np.linalg.norm(b[:, :2]-a[:, :2], axis=-1)
        was, now = a[:, 3:].argmax(-1), b[:, 3:].argmax(-1)
        self.pickups += int(np.count_nonzero(own & (was == 0) & (now == 1)))
        score = float(after[0, 93]-before[0, 93])
        self.deliveries += int(score > 0 and np.any(own & (was == 1) & (now == 0)))
        self.respawns += int(np.count_nonzero(own & (movement > .2)))
        self.moving += int(np.any(movement[own] > 1e-5))
        self.stuck += int(np.all(movement[own] < 1e-5))
        self.score_gain += max(0., score)

    def report(self):
        return {"steps": self.steps, "pickup_proxy": self.pickups, "delivery_proxy": self.deliveries,
                "respawn_proxy": self.respawns, "moving_step_fraction": self.moving/max(1, self.steps),
                "stuck_team_step_fraction": self.stuck/max(1, self.steps),
                "visible_positive_score_delta": self.score_gain, "event_truth": "observation proxies"}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def collect_episode(policy, critic, opponent, task, env_factory, heartbeat=lambda **_: None, stop=lambda: False):
    del critic  # Frozen predecision critic targets are batched after full collection.
    cfg, side = task["config"], int(task["side"])
    started = time.monotonic()
    env, writer = None, None
    steps, last_heartbeat = 0, started
    behavior = Behavior()
    recorder = SituationRecorder.for_task(task)
    timings = {"actor": 0., "opponent": 0., "env_step": 0., "storage": 0., "reset": 0.}
    windows = []
    own_names, other_names = canonical_names(side), canonical_names(1-side)
    expected = set(own_names+other_names)
    device = next(policy.parameters()).device
    state = policy.initial_state(1, device=device)
    teacher = None
    rng = np.random.default_rng(task["action_seed"]+89)
    if task.get("teacher"):
        from .opponents import make_opponent
        teacher = make_opponent(task["teacher_spec"], task["action_seed"]+97)
    try:
        heartbeat(phase="environment_start", steps=0)
        env = env_factory()
        obs, _ = env.reset(seed=task["requested_seed"])
        if recorder is not None:
            recorder.start(obs)
        timings["reset"] = time.monotonic()-started
        if set(obs) != expected:
            raise ValueError("incomplete initial team observations")
        vectors, graphic = pack(obs, own_names)
        if not (graphic[1:] > 0).any():
            raise ValueError("graphic has no visible game content")
        initial_hash = observation_hash(obs)
        from .evaluation import initial_map_hash
        map_hash = initial_map_hash(obs)
        totals = {name: 0. for name in expected}
        writer = EpisodeWriter(task["trajectory"], cfg.get("max_episode_steps", 22000), cfg.get("trajectory_chunk", 64))
        previous_time = float(vectors[0, 95])
        for step in range(cfg.get("max_episode_steps", 22000)):
            if stop():
                raise InterruptedError("stopped; incomplete episode excluded")
            if step % 128 == 0 and shutil.disk_usage(Path(task["trajectory"])).free < cfg.get("disk_stop_gib", 25)*1024**3:
                raise RuntimeError("trajectory disk reserve reached")
            now = time.monotonic()
            if step == 0 or now-last_heartbeat >= 5:
                heartbeat(phase="collect", steps=step, decisions=writer.length, game_time_left=previous_time, **timings)
                last_heartbeat = now
                if task.get("audit_windows", True):
                    from .windows import visible_windows
                    audit = visible_windows()
                    if len(windows) < 32:
                        windows.append(audit)
                    if not audit["verified"] or audit["onscreen_owned_windows"]:
                        raise RuntimeError("background window audit failed: " + str(audit))
            mark = time.monotonic()
            teacher_actions = None
            if teacher is not None:
                teacher_actions = teacher.act({n: obs[n] for n in own_names}, own_names)
            with torch.inference_mode():
                v = torch.from_numpy(vectors[None]).to(device)
                g = torch.from_numpy(graphic[None]).to(device)
                result = policy.step(v, g, state, deterministic=bool(task.get("deterministic", False)))
                state = result["state"]
                selected = result["action"][0].detach().cpu().numpy()
                if selected.shape != (5, 2) or not np.isfinite(selected).all() or np.abs(selected).max() > 1.00001:
                    raise ValueError("invalid actor output")
                is_event = bool(result["event"][0].item())
                labels = None
                if is_event and teacher_actions is not None:
                    from .teachers import label_choices
                    target = torch.as_tensor(np.stack([teacher_actions[n] for n in own_names])[None], device=device)
                    labels = label_choices(policy, v, g, result["decision_state"], target, result.get("base_action"))
            actor_seconds = time.monotonic()-mark
            timings["actor"] += actor_seconds
            if is_event:
                mark = time.monotonic()
                writer.add(step=step, vectors=vectors, graphic=graphic, decision_state=result["decision_state"],
                           choices=result["choices"], log_prob=float(result["log_prob"][0].item()),
                           potential=potential(vectors), teacher_choices=labels, on_policy=teacher is None)
                timings["storage"] += time.monotonic()-mark
            actions = dict(zip(own_names, selected))
            teacher_forced = False
            if teacher_actions is not None and rng.random() < float(task.get("teacher_mix", 1.)):
                teacher_forced = True
                actions = teacher_actions
                executed = torch.as_tensor(np.stack([actions[n] for n in own_names])[None],
                                           device=state["previous_action"].device, dtype=torch.float32)
                # Explicit state uses canonical coordinates; teacher outputs are
                # physical movements. Record the action that actually ran.
                state["previous_action"] = executed.flip(-1) if side else executed
            mark = time.monotonic()
            actions.update(opponent.act({n: obs[n] for n in other_names}, other_names))
            opponent_seconds = time.monotonic()-mark
            timings["opponent"] += opponent_seconds
            mark = time.monotonic()
            next_obs, raw, terms, truncs, infos = env.step(actions)
            timings["env_step"] += time.monotonic()-mark
            steps = step+1
            if recorder is not None:
                recorder.observe(step, obs, next_obs, actions, raw, terms, truncs, infos,
                    decision=decision_metadata(policy, side, selected, result=result, teacher_forced=teacher_forced),
                    inference_seconds={side: actor_seconds, 1-side: opponent_seconds})
            if set(raw) != expected or set(terms) != expected or set(truncs) != expected:
                raise ValueError("incomplete ten-agent transition")
            if len({bool(terms[n]) for n in expected}) != 1 or any(truncs.values()):
                raise ValueError("partial termination or truncation is not a training episode")
            if not all(np.isfinite(float(x)) for x in raw.values()):
                raise ValueError("nonfinite raw reward")
            for name in expected:
                totals[name] += float(raw[name])
            done = all(terms.values())
            if set(next_obs) == expected:
                nv, ng = pack(next_obs, own_names)
            elif done:
                nv, ng = vectors, graphic
            else:
                raise ValueError("incomplete next observations")
            if not done and float(nv[0, 95]) > previous_time+1e-3:
                raise ValueError("time reset without terminal")
            if not done:
                previous_time = float(nv[0, 95])
            behavior.observe(vectors, nv, done)
            if done:
                if env.agents:
                    raise ValueError("terminal flags with active agents")
                team_totals = [sum(totals[n] for n in canonical_names(s)) for s in (0, 1)]
                outcome = terminal_outcome(team_totals, side)
                metadata = dict(natural_terminal=True, truncated=False, watchdog=False, objective=OBJECTIVE,
                    policy_sha256=task["policy_sha256"], episode_id=task["id"], env_steps=steps,
                    side=side, requested_seed=task["requested_seed"], action_seed=task["action_seed"],
                    initial_observation_hash=initial_hash, initial_map_hash=map_hash,
                    applied_map_seed_verified=False, raw_team_totals=team_totals, runner_reward_totals=team_totals,
                    runner_winner=None if team_totals[0] == team_totals[1] else int(team_totals[1] > team_totals[0]),
                    opponent=task["opponent"], family=task.get("family", task["opponent"].get("family", "unknown")),
                    score_before_terminal=vectors[0, 93:95].tolist(),
                    score_after_terminal=nv[0, 93:95].tolist() if set(next_obs) == expected else None,
                    terminal_info_raw=_json_safe(infos), official_winner=None, official_winner_available=False,
                    behavior=behavior.report(), wall_seconds=time.monotonic()-started, timings=timings,
                    window_audits=windows, teacher=task.get("teacher_spec"), teacher_mix=task.get("teacher_mix"),
                    on_policy=teacher is None,
                    behavior_log=recorder.close(complete=True) if recorder is not None else {"enabled": False})
                return writer.commit(outcome, cfg.get("gamma", 1.), cfg.get("gae_lambda", .9975), metadata)
            obs, vectors, graphic = next_obs, nv, ng
        raise RuntimeError("episode step watchdog; partial episode excluded")
    except BaseException as exc:
        behavior_log = recorder.close(error=exc) if recorder is not None else {"enabled": False}
        atomic_json(task["invalid_path"], {"valid": False, "error": str(exc), "type": type(exc).__name__,
                    "steps": steps, "decisions": 0 if writer is None else writer.length, "behavior": behavior.report(),
                    "wall_seconds": time.monotonic()-started, "policy_sha256": task["policy_sha256"],
                    "episode_id": task["id"], "no_optimizer_commit": True, "behavior_log": behavior_log})
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()
