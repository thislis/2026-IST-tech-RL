"""A fresh supplied environment per complete episode, with staging before PPO."""
import hashlib
import time
import numpy as np
import torch
from .diagnostics import Behavior
from .io import atomic_json
from .rewards import OBJECTIVE, potential, terminal_outcome
from .trajectory import EpisodeWriter, pack


def observation_hash(obs):
    h = hashlib.sha256()
    for name in sorted(obs):
        h.update(name.encode())
        for key in ("vector", "graphic"):
            h.update(obs[name][key].tobytes())
    return h.hexdigest()


def collect_episode(policy, critic, opponent, task, env_factory, heartbeat=lambda **_: None, stop=lambda: False):
    cfg, side = task["config"], task["side"]
    start = time.monotonic()
    env = None
    writer = None
    behavior = Behavior()
    generator = torch.Generator().manual_seed(task["action_seed"])
    timings = {"inference": 0., "opponent": 0., "env_step": 0., "storage": 0., "reset": 0.}
    last_heartbeat = start
    own_names = []
    window_reports = []
    try:
        heartbeat(phase="environment_start", steps=0)
        env = env_factory()
        obs, _ = env.reset(seed=task["requested_seed"])
        timings["reset"] = time.monotonic()-start
        expected = {f"unit_{i}" for i in range(10)}
        if set(obs) != expected:
            raise ValueError("incomplete initial observations")
        own_names = [n for n in obs if int(n.split("_")[1])//5 == side]
        other_names = [n for n in obs if n not in own_names]
        initial_hash = observation_hash({n: obs[n] for n in own_names})
        totals = {n: 0. for n in expected}
        vectors, graphic = pack(obs, own_names)
        if not (graphic[1:] > 0).any():
            raise ValueError("graphic has no visible game content")
        writer = EpisodeWriter(task["trajectory"], cfg["max_episode_steps"])
        previous_time = float(vectors[0, 95])
        for step in range(cfg["max_episode_steps"]):
            if stop():
                raise InterruptedError("stop requested; uncommitted episode discarded")
            now = time.monotonic()
            if step == 0 or now-last_heartbeat >= 5:
                heartbeat(phase="collect", steps=step, game_time_left=previous_time, **timings)
                last_heartbeat = now
                if task.get("audit_windows"):
                    from .windows import visible_windows
                    audit = visible_windows()
                    if len(window_reports) < 32: window_reports.append(audit)
                    if not audit["verified"] or audit["onscreen_owned_windows"]:
                        raise RuntimeError("rendered background window audit failed: " + str(audit))
            mark = time.monotonic()
            with torch.inference_mode():
                vt, gt = torch.from_numpy(vectors), torch.from_numpy(graphic[None])
                logs = policy.team_distribution(vt[None], gt)[0][0]
                indices = policy.choose(logs, generator)
                selected = policy.directions[indices].numpy()
                old_log = logs.gather(-1, indices[:, None]).squeeze(-1).numpy()
                value = float(critic(vt[None])[0])
            timings["inference"] += time.monotonic()-mark
            actions = dict(zip(own_names, selected))
            mark = time.monotonic()
            actions.update(opponent.act({n: obs[n] for n in other_names}, other_names))
            timings["opponent"] += time.monotonic()-mark
            mark = time.monotonic()
            next_obs, raw, terms, truncs, infos = env.step(actions)
            timings["env_step"] += time.monotonic()-mark
            if set(raw) != expected or set(terms) != expected or set(truncs) != expected:
                raise ValueError("incomplete ten-agent transition")
            if len({bool(terms[n]) for n in expected}) != 1 or any(truncs.values()):
                raise ValueError("partial termination/unexpected truncation")
            if not all(np.isfinite(float(x)) for x in raw.values()):
                raise ValueError("nonfinite raw reward")
            for n in expected: totals[n] += float(raw[n])
            done = all(terms.values())
            if set(next_obs) == expected:
                nv, ng = pack(next_obs, own_names)
            elif done:
                nv, ng = vectors, graphic
            else:
                raise ValueError("incomplete next observation")
            if not done and float(nv[0, 95]) > previous_time+1e-3:
                raise ValueError("time reset without terminal")
            if not done: previous_time = float(nv[0, 95])
            aux = behavior.observe(vectors, nv, indices.numpy(), done, logs.argmax(-1).numpy())
            mark = time.monotonic()
            writer.add(vectors=vectors, graphic=graphic, actions=indices.numpy(), log_probs=old_log,
                       values=value, potentials=potential(vectors), aux_targets=aux, aux_valid=float(not done))
            timings["storage"] += time.monotonic()-mark
            if done:
                if env.agents:
                    raise ValueError("terminal flags with active agents")
                # Exactly the original runner's per-agent accumulation and team sum.
                team_totals = [sum(totals[n] for n in {f"unit_{j}" for j in range(i*5, i*5+5)}) for i in (0, 1)]
                metadata = dict(natural_terminal=True, objective=OBJECTIVE, policy_sha256=task["policy_sha256"],
                    episode_id=task["id"], side=side, requested_seed=task["requested_seed"],
                    applied_map_seed_verified=False, initial_observation_sha256=initial_hash,
                    raw_team_totals=team_totals, opponent=task["opponent"],
                    last_visible_score=vectors[0, 93:95].tolist(),
                    terminal_observed_score=nv[0, 93:95].tolist() if set(next_obs) == expected else None,
                    wrapper_terminal_proxy={n: infos.get(n, {}).get("winner") for n in expected},
                    behavior=behavior.report(), wall_seconds=time.monotonic()-start, timings=timings,
                    window_audits=window_reports, engine_winner_verified=False)
                return writer.commit(terminal_outcome(team_totals, side), cfg["gamma"], cfg["gae_lambda"], metadata)
            obs, vectors, graphic = next_obs, nv, ng
        raise RuntimeError("22,000-step watchdog; entire episode invalid, not a draw")
    except BaseException as exc:
        atomic_json(task["invalid_path"], {"valid": False, "error": str(exc), "type": type(exc).__name__,
                    "steps": 0 if writer is None else writer.length, "behavior": behavior.report(),
                    "wall_seconds": time.monotonic()-start, "policy_sha256": task["policy_sha256"],
                    "episode_id": task["id"], "no_optimizer_commit": True})
        raise
    finally:
        if env is not None: env.close()
