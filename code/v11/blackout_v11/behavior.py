"""Primitive-transition evidence and observation-conditioned behavior summaries.

Trace vectors/actions use the league's lossless XOR format. Graphic snapshots
are evidence samples, not a complete graphic replay. All event interpretations
are explicitly observation proxies rather than inferred policy intentions.
"""
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from agent_league.trace import Behavior as LeagueBehavior, Trace, NAMES
from .io import ROOT, atomic_json, read_json

SCHEMA = "blackout.v11.behavior.v1"
DEFAULTS = dict(chunk_steps=512, snapshot_every=1000, stop_epsilon=1e-6,
                movement_epsilon=1e-5, teleport_distance=.2, near_base_distance=.12,
                direction_change_cos=.70710678, enemy_near_distance=.15)


def plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def decision_metadata(policy, side, planned_actions, *, result=None, teacher_forced=False):
    """Read existing v11 state without changing policy execution or sampling."""
    if result is not None:
        state = result.get("state")
        event = bool(result["event"][0].item())
    else:
        model = getattr(policy, "_net", getattr(policy, "policy", policy))
        state = getattr(model, "_runtime_state", None)
        if not state:
            return None
        goal = state.get("goal_age")
        residual = state.get("residual_remaining")
        event = bool((goal == 1).any().item()) or bool((residual == getattr(model, "residual_period", 4)-1).any().item())
    data = dict(side=int(side), event=event, teacher_forced=bool(teacher_forced),
                planned_actions=plain(planned_actions), coordinate_system="goals/state canonical; planned actions physical")
    if event and state:
        for source, target in (("goal", "selected_goal"), ("role", "selected_role"),
                               ("intent", "selected_intent"), ("correction", "selected_residual"),
                               ("direct", "selected_direct"), ("goal_age", "goal_age")):
            if source in state:
                data[target] = plain(state[source][0])
        if result is not None:
            data["head_choices"] = {k: plain(v[0]) for k, v in result.get("choices", {}).items()}
    return data


class BoundedTrace(Trace):
    """League-compatible transition chunks with a hard retained-file budget."""
    def __init__(self, path, cfg, limit):
        if limit < 16384:
            raise ValueError("behavior trace limit must be at least 16 KiB")
        super().__init__(path, cfg)
        self.events.close()
        self.limit = int(limit)
        self.index_reserve = min(65536, limit//8)
        self.final_reserve = min(1024**2, limit//4)
        self.used = (self.path/"events.jsonl.gz").stat().st_size
        self.capped = False
        self.dropped_transitions = self.dropped_events = self.dropped_snapshots = 0
        self.pending_events = []
        self.pending_event_bytes = 0
        self.closed = False

    def _fits(self, size, final=False):
        reserve = self.index_reserve+(0 if final else self.final_reserve)
        return self.used+size <= self.limit-reserve

    def event(self, row):
        if self.capped:
            self.dropped_events += 1
            return
        encoded = json.dumps(plain(row), ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n"
        self.pending_events.append(encoded)
        self.pending_event_bytes += len(encoded.encode())
        if self.pending_event_bytes >= 256*1024:
            self.flush_events()

    def flush_events(self):
        if not self.pending_events:
            return
        compressed = gzip.compress("".join(self.pending_events).encode(), mtime=0)
        if self._fits(len(compressed)):
            with (self.path/"events.jsonl.gz").open("ab") as stream:
                stream.write(compressed)
            self.used += len(compressed)
            self.event_count += len(self.pending_events)
        else:
            self.capped = True
            self.dropped_events += len(self.pending_events)
        self.pending_events.clear()
        self.pending_event_bytes = 0

    def append(self, **row):
        if self.capped:
            self.dropped_transitions += 1
            return
        super().append(**row)

    def flush(self):
        if not self.rows:
            self.flush_events()
            return
        count = len(self.rows)
        super().flush()
        item = self.chunks[-1]
        path = self.path/item["path"]
        size = path.stat().st_size
        if self._fits(size):
            self.used += size
        else:
            path.unlink()
            self.chunks.pop()
            self.count -= count
            self.dropped_transitions += count
            self.capped = True
        self.flush_events()

    def snapshot(self, obs, step, final=False):
        if not obs or (self.capped and not final):
            self.dropped_snapshots += 1
            return
        name = f"map-{step:06d}.npz"
        if any(item["path"] == name for item in self.snapshots):
            return
        super().snapshot(obs, step)
        path = self.path/name
        size = path.stat().st_size
        if self._fits(size, final=final):
            self.used += size
        else:
            path.unlink()
            self.snapshots.pop()
            self.dropped_snapshots += 1
            self.capped = True

    def close(self, complete=False):
        if self.closed:
            return
        self.flush()
        super().close(complete=complete and not self.capped)
        index = read_json(self.path/"index.json")
        index.update(behavior_schema=SCHEMA, episode_complete=bool(complete), trace_capped=self.capped,
                     retained_byte_limit=self.limit, dropped_transitions=self.dropped_transitions,
                     dropped_events=self.dropped_events, dropped_snapshots=self.dropped_snapshots,
                     summary_continues_after_cap=True,
                     entity_identity="v11 supplied runner: public unit order equals observed absolute entity order")
        atomic_json(self.path/"index.json", index)
        self.used += (self.path/"index.json").stat().st_size
        self.closed = True


def _vectors(obs):
    present = np.array([n in obs and "vector" in obs[n] for n in NAMES], bool)
    values = np.full((10, 96), np.nan, np.float32)
    for i, name in enumerate(NAMES):
        if present[i]:
            values[i] = np.asarray(obs[name]["vector"], np.float32)
    return values, present


def _map_counts(obs):
    counts = np.full((2, 11), np.nan, np.float32)
    for side in (0, 1):
        item = obs.get(f"unit_{side*5}")
        if item is not None and "graphic" in item:
            counts[side] = np.asarray(item["graphic"]).sum((0, 1))
    return counts


class SituationRecorder:
    def __init__(self, path, cfg=None, metadata=None):
        cfg = cfg or {}
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
        self.cfg["chunk_steps"] = min(512, max(1, int(cfg.get("behavior_chunk_steps", 512))))
        self.cfg["snapshot_every"] = max(1, int(cfg.get("behavior_snapshot_every", 1000)))
        limit = int(cfg.get("behavior_trace_max_bytes", cfg.get("behavior_trace_mib", 64)*1024**2))
        self.trace = BoundedTrace(self.path/"trace", self.cfg, limit)
        self.league = LeagueBehavior(self.cfg)
        self.metadata = plain(metadata or {})
        self.steps = self.invalid_transitions = self.decision_events = self.teacher_forced_steps = 0
        self.cumulative = np.zeros(10, np.float64)
        self.cumulative_complete = np.ones(10, bool)
        self.situations = {"side0": {}, "side1": {}}
        self.previous_action = None
        self.last_obs, self.last_observation_step = None, 0
        self.examples, self.started, self.closed = [], time.time(), None

    @classmethod
    def for_task(cls, task):
        cfg = task.get("config", {})
        if not cfg.get("behavior_logging", True):
            return None
        if task.get("trajectory"):
            attempt = Path(task["trajectory"]).resolve().parent
        else:
            attempt = Path(task.get("attempt_directory", ROOT/"logs/v11/behavior"/task.get("id", "unbound"))).resolve()
        if cfg.get("behavior_output_root"):
            key = hashlib.sha256(str(attempt).encode()).hexdigest()[:24]
            path = Path(cfg["behavior_output_root"])/"episodes"/key
        else:
            path = attempt/"behavior"
        opponent = task.get("opponent", {})
        metadata = {"episode_id": task.get("id"), "task_id": task.get("id"), "kind": task.get("kind"),
                    "arm": task.get("arm"), "side": task.get("side"), "requested_seed": task.get("requested_seed"),
                    "action_seed": task.get("action_seed"), "opponent_id": opponent.get("id", opponent.get("kind")),
                    "opponent_family": opponent.get("family", task.get("family")),
                    "teacher": task.get("teacher_spec"), "attempt_source": str(attempt),
                    "task_source": str(attempt/"task.json"), "split": task.get("split")}
        return cls(path, cfg, metadata)

    def start(self, obs):
        self.last_obs = obs
        self.trace.snapshot(obs, 0)

    def _example(self, event):
        kind = event.get("kind")
        maximum = 8 if kind == "policy_decision" else 12 if kind == "situation_proxy" else 4
        same = [i for i, item in enumerate(self.examples) if item.get("kind") == kind]
        if len(same) < maximum and len(self.examples) < 24:
            self.examples.append(plain(event))
        elif same and kind != "policy_decision":
            # Preserve the first examples and the newest observation/terminal.
            self.examples[same[-1]] = plain(event)

    def _situations(self, before, after, after_present, actions, terminal, obs):
        for side in (0, 1):
            base = side*5
            if not np.isfinite(before[base]).all():
                continue
            entities = before[base, :90].reshape(10, 9)
            own = np.flatnonzero(entities[:, 2] > 0)
            if len(own) != 5:
                continue
            positions = entities[own, :2]
            enemies = entities[entities[:, 2] < 0, :2]
            close_enemy = (np.linalg.norm(positions[:, None]-enemies[None], axis=-1).min(-1) < self.cfg["enemy_near_distance"]
                           if len(enemies) else np.zeros(5, bool))
            item = obs.get(f"unit_{base}", {})
            storage = np.argwhere(np.asarray(item["graphic"])[..., 2] > 0) if "graphic" in item else np.empty((0, 2))
            if len(storage):
                storage_xy = np.stack(((storage[:, 1]+.5)/96, 1-(storage[:, 0]+.5)/96), -1)
                near_storage = np.linalg.norm(positions[:, None]-storage_xy[None], axis=-1).min(-1) < self.cfg["near_base_distance"]
            else:
                near_storage = np.zeros(5, bool)
            holding = entities[own, 3:].argmax(-1)
            classes = before[base:base+5, 90:93].argmax(-1)
            act = actions[base:base+5]
            norms = np.linalg.norm(act, axis=-1)
            stopped = norms <= self.cfg["stop_epsilon"]
            turned = np.zeros(5, bool)
            if self.previous_action is not None:
                previous = self.previous_action[base:base+5]
                pn = np.linalg.norm(previous, axis=-1)
                cosine = (act*previous).sum(-1)/np.maximum(norms*pn, 1e-12)
                turned = (pn > self.cfg["stop_epsilon"]) & ~stopped & (cosine < self.cfg["direction_change_cos"])
            observed = bool(after_present[base:base+5].all() and
                            np.isfinite(after[base:base+5]).all() and not terminal)
            moved = pickup = delivery = respawn = np.zeros(5, bool)
            if observed:
                nxt = after[base, :90].reshape(10, 9)
                distance = np.linalg.norm(nxt[own, :2]-positions, axis=-1)
                moved = distance > self.cfg["movement_epsilon"]
                held_next = nxt[own, 3:].argmax(-1)
                pickup = (holding == 0) & (held_next == 1)
                respawn = distance > self.cfg["teleport_distance"]
                same_class = (after[base:base+5, 90:93].argmax(-1) == classes)
                delivery = (holding == 1) & (held_next == 0) & near_storage & same_class & ~respawn
            delta = before[base, 93]-before[base, 94]
            score = "winning" if delta > 0 else "losing" if delta < 0 else "tied"
            clock = before[base, 95]
            phase = "early" if clock > 2/3 else "mid" if clock > 1/3 else "late"
            for unit in range(5):
                categories = ["score:"+score, "phase:"+phase,
                    "holding:"+("carrying" if holding[unit] else "empty"),
                    "enemy:"+("near" if close_enemy[unit] else "far"),
                    "storage:"+("near" if near_storage[unit] else "far"), "class:"+str(int(classes[unit]))]
                categories.append("joint:"+"|".join(x.split(":", 1)[1] for x in categories))
                values = dict(unit_steps=1, stop=int(stopped[unit]), command_move=int(not stopped[unit]),
                              turn=int(turned[unit]), movement_observed_steps=int(observed),
                              observed_move=int(moved[unit]), pickup_proxy=int(pickup[unit]),
                              delivery_proxy=int(delivery[unit]), respawn_proxy=int(respawn[unit]))
                for category in categories:
                    row = self.situations[f"side{side}"].setdefault(category, {k: 0 for k in values})
                    for key, value in values.items():
                        row[key] += value
            if pickup.any() or delivery.any() or respawn.any():
                event = dict(step=self.steps, side=side, kind="situation_proxy", pickup=int(pickup.sum()),
                             delivery=int(delivery.sum()), respawn=int(respawn.sum()), score=score, phase=phase)
                self.trace.event(event)
                self._example(event)

    def observe(self, step, obs, next_obs, actions, rewards, terms, truncs, infos=None,
                decision=None, inference_seconds=None):
        if self.closed is not None:
            raise RuntimeError("behavior recorder already closed")
        before, before_present = _vectors(obs)
        after, after_present = _vectors(next_obs)
        action = np.full((10, 2), np.nan, np.float32)
        reward = np.full(10, np.nan, np.float64)
        reward_present = np.array([n in rewards for n in NAMES], bool)
        terminated = np.array([bool(terms.get(n, False)) for n in NAMES])
        truncated = np.array([bool(truncs.get(n, False)) for n in NAMES])
        for i, name in enumerate(NAMES):
            if name in actions:
                action[i] = actions[name]
            if reward_present[i]:
                reward[i] = float(rewards[name])
                if np.isfinite(reward[i]):
                    self.cumulative[i] += reward[i]
            self.cumulative_complete[i] &= bool(reward_present[i] and np.isfinite(reward[i]))
        terminal = bool(terminated.all())
        valid = (set(terms) == set(NAMES) == set(truncs) == set(actions) == set(rewards)
                 and before_present.all() and np.isfinite(before).all() and np.isfinite(action).all()
                 and np.isfinite(reward).all() and len(set(terminated.tolist())) == 1
                 and not truncated.any() and (after_present.all() or terminal)
                 and np.isfinite(after[after_present]).all())
        side_totals = self.cumulative.reshape(2, 5).sum(-1)
        latency = np.full(2, np.nan, np.float64)
        if inference_seconds is not None:
            if isinstance(inference_seconds, dict):
                for key, value in inference_seconds.items():
                    latency[int(key)] = float(value)
            else:
                latency[:] = np.asarray(inference_seconds, np.float64)
        planned = np.full((10, 2), np.nan, np.float32)
        decisions, forced = np.zeros(2, bool), np.zeros(2, bool)
        items = [] if decision is None else decision if isinstance(decision, list) else [decision]
        for item in items:
            if item is None:
                continue
            item = plain(item)
            side = int(item["side"])
            decisions[side] = bool(item.get("event"))
            forced[side] = bool(item.get("teacher_forced"))
            if "planned_actions" in item:
                planned[side*5:side*5+5] = np.asarray(item["planned_actions"], np.float32)
            if decisions[side] or forced[side]:
                item["executed_actions"] = plain(action[side*5:side*5+5])
                event = dict(step=int(step), kind="policy_decision" if decisions[side] else "teacher_override", data=item)
                self.trace.event(event)
                if decisions[side]:
                    self._example(event)
        self.decision_events += int(decisions.sum())
        self.teacher_forced_steps += int(forced.sum())
        changed = np.full(2, -1, np.int64)
        for side in (0, 1):
            name = f"unit_{side*5}"
            if name in obs and name in next_obs and "graphic" in obs[name] and "graphic" in next_obs[name]:
                changed[side] = np.count_nonzero(np.asarray(obs[name]["graphic"]) != np.asarray(next_obs[name]["graphic"]))
        self.trace.append(step=int(step), wall_time=time.time(), game_time=float(before[0, 95]),
            before=before, before_present=before_present, actions=action, planned_actions=planned,
            teacher_forced=forced, v11_decision=decisions, rewards=reward, reward_present=reward_present,
            after=after, after_present=bool(after_present.all()), after_present_mask=after_present,
            terminated=terminated, truncated=truncated, cumulative_reward=side_totals,
            terminated_present=np.array([n in terms for n in NAMES]), truncated_present=np.array([n in truncs for n in NAMES]),
            cumulative_agent_reward=self.cumulative.copy(), cumulative_reward_complete=self.cumulative_complete.copy(),
            inference_seconds=latency, map_channel_counts=_map_counts(obs),
            map_channel_counts_after=_map_counts(next_obs), graphic_changed_values=changed, transition_valid=bool(valid))
        if before_present.all() and np.isfinite(action).all():
            self.league.observe(step, before, after if after_present.all() else None, action, side_totals,
                                self.trace, terminal=terminal, obs=obs)
            self._situations(before, after, after_present, action, terminal, obs)
        if terminal:
            event = dict(step=int(step), kind="terminal", terminated=terminated.tolist(), truncated=truncated.tolist(),
                         after_present=after_present.tolist(), infos=plain(infos or {}))
            self.trace.event(event)
            self._example(event)
        self.steps += 1
        self.invalid_transitions += int(not valid)
        self.previous_action = action.copy()
        self.last_obs = next_obs if after_present.all() else obs
        self.last_observation_step = step+1 if after_present.all() else step
        if (step+1) % self.cfg["snapshot_every"] == 0:
            self.trace.snapshot(self.last_obs, self.last_observation_step)

    def close(self, complete=False, error=None):
        if self.closed is not None:
            return self.closed
        if self.last_obs:
            self.trace.snapshot(self.last_obs, self.last_observation_step, final=True)
        self.trace.close(complete=complete)
        report = dict(schema=SCHEMA, metadata=self.metadata, steps=self.steps, complete=bool(complete),
            error=None if error is None else str(error), invalid_transitions=self.invalid_transitions,
            cumulative_reward=self.cumulative.reshape(2, 5).sum(-1).tolist(),
            cumulative_reward_complete=self.cumulative_complete.tolist(), league=self.league.report(),
            situations=self.situations, decision_events=self.decision_events,
            teacher_forced_team_steps=self.teacher_forced_steps, examples=self.examples,
            started_at=self.started, closed_at=time.time(), thresholds=self.cfg,
            situation_definition=dict(score="pre-action observed score, not reward winner",
                phase="remaining time: early > 2/3, mid > 1/3, late otherwise",
                holding="observed item category; carrying includes battery or special item",
                movement="command_move is commanded nonzero action; observed_move is position change",
                classes={"0": "worker", "1": "hunter", "2": "carrier"},
                joint_order=["score", "phase", "holding", "enemy", "storage", "class"],
                events="pickup/delivery/respawn are observation proxies; terminal changes excluded"),
            trace=dict(path=str(self.trace.path), steps=self.trace.count, events=self.trace.event_count,
                capped=self.trace.capped, bytes=self.trace.used, byte_limit=self.trace.limit,
                dropped_transitions=self.trace.dropped_transitions, dropped_events=self.trace.dropped_events,
                dropped_snapshots=self.trace.dropped_snapshots, complete=bool(complete and not self.trace.capped),
                graphic="initial/final and periodic two-team snapshots, not full graphic replay"))
        atomic_json(self.path/"summary.json", report)
        self.closed = dict(enabled=True, schema=SCHEMA, summary_path=str(self.path/"summary.json"),
            trace_path=str(self.trace.path), steps=self.steps, trace_steps=self.trace.count,
            trace_capped=self.trace.capped, complete=bool(complete))
        return self.closed
