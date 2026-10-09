"""Frozen opponents. Only their public observations and action outputs are used."""

from project_paths import project_root, project_path

from collections import deque
import numpy as np
import torch
from .io import ROOT, file_hash
from .policy import MyPolicy


class NeuralOpponent:
    def __init__(self, path):
        self.policy = MyPolicy().eval()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.policy.load_state_dict(payload["policy_state"])

    @torch.inference_mode()
    def act(self, obs, names=None):
        names = list(obs) if names is None else list(names)
        vectors = torch.from_numpy(np.stack([obs[n]["vector"] for n in names]))
        graphic = torch.from_numpy(np.stack([obs[n]["graphic"] for n in names])).permute(0, 3, 1, 2)
        actions = self.policy(vectors, graphic).numpy()
        return dict(zip(names, actions))


class StrategyOpponent:
    """Observation-only scripted stress tasks, not claimed as learned strategies.

    Historical/scripted training opponents receive normal env agent names; the
    submitted V9 actor does not. Geometry is decoded only from visible pixels.
    """
    def __init__(self, kind, seed=0):
        self.kind, self.rng = kind, np.random.default_rng(seed)
        self.previous = None
        self.cached = None

    def act(self, obs, names=None):
        names = list(obs) if names is None else list(names)
        if self.kind in ("noop", "random"):
            return {n: (np.zeros(2, np.float32) if self.kind == "noop" else
                        np.asarray(self.rng.uniform(-1, 1, 2), np.float32)) for n in names}
        first = obs[names[0]]
        entities = first["vector"][:90].reshape(10, 9)
        graphic = first["graphic"]
        # The graphic's vertical axis is reversed relative to world Y.
        tiled = graphic.reshape(24, 4, 24, 4, 11).sum((1, 3))[::-1]
        blocked = tiled[..., 1] > 8
        own = np.argwhere(tiled[..., 2] > 0)
        enemy = np.argwhere(tiled[..., 3] > 0)
        batteries = np.argwhere(tiled[..., 6] > 0)
        result = {}
        for n in names:
            index = int(n.split("_")[1])
            position = entities[index, :2]
            cell = tuple(np.clip(np.floor(position[::-1]*24), 0, 23).astype(int))
            holding = int(entities[index, 3:].argmax())
            klass = int(obs[n]["vector"][90:93].argmax())
            targets = own if holding else batteries
            if self.kind == "raider" and not holding and len(enemy):
                targets = enemy
            if self.kind == "rush":
                if klass != 1:
                    targets = np.asarray([[11, 11], [12, 12]])
                else:
                    positions = entities[entities[:, 2] < 0, :2]
                    targets = np.clip(np.floor(positions[:, ::-1]*24), 0, 23).astype(int)
            if self.kind == "defender" and index % 5 >= 2:
                targets = np.asarray([[11, 11]]) if klass != 1 else own
                foes = entities[entities[:, 2] < 0, :2]
                close = foes[np.linalg.norm(foes-position, axis=1) < .2]
                if klass == 1 and len(close):
                    targets = np.clip(np.floor(close[:, ::-1]*24), 0, 23).astype(int)
            if not len(targets):
                targets = np.asarray([[11, 11]])
            target = min((tuple(x) for x in targets), key=lambda x: abs(x[0]-cell[0])+abs(x[1]-cell[1]))
            step = self.next_step(cell, target, blocked)
            destination = (np.asarray(step[::-1], np.float32)+.5)/24
            direction = destination-position
            norm = np.linalg.norm(direction)
            result[n] = (direction/max(norm, 1e-6)).astype(np.float32) if norm > .008 else np.zeros(2, np.float32)
        return result

    @staticmethod
    def next_step(start, goal, blocked):
        queue, previous = deque([start]), {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                while previous[current] is not None and previous[current] != start:
                    current = previous[current]
                return current
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nxt = current[0]+dy, current[1]+dx
                if 0 <= nxt[0] < 24 and 0 <= nxt[1] < 24 and nxt not in previous and not blocked[nxt]:
                    previous[nxt] = current
                    queue.append(nxt)
        return start


class HistoricalOpponent:
    def __init__(self, kind):
        self.kind, self.policy = kind, None

    def act(self, obs, names=None):
        names = sorted(list(obs) if names is None else list(names), key=lambda n: int(n.split("_")[1]))
        if self.policy is None:
            team = int(names[0].split("_")[1]) // 5
            if self.kind == "target":
                from blackout_rl.policy import DeterministicCheckpointPolicy
                self.policy = DeterministicCheckpointPolicy(project_path('artifacts/checkpoints/win_70_vs_scripted.pt', root=ROOT), team=team)
            else:
                from blackout_rl.frozen_opponent import FrozenScriptedOpponent
                self.policy = FrozenScriptedOpponent(team)
            self.policy.reset()
        return self.policy.act(obs, names)


def make_opponent(spec, seed=0):
    kind = spec["kind"]
    if kind == "history":
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("history snapshot changed")
        return NeuralOpponent(spec["path"])
    if kind in ("target", "scripted"):
        return HistoricalOpponent(kind)
    if kind not in ("noop", "random", "collector", "rush", "raider", "defender"):
        raise ValueError("unknown opponent: " + kind)
    return StrategyOpponent(kind, seed)
