"""Lossless, bounded event replay; only natural complete episodes are committed."""
from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, read_json
from .rewards import shape_episode


def canonical_names(side):
    return [f"unit_{i}" for i in range(side*5, side*5+5)]


def pack(obs, names):
    if len(names) != 5 or names != sorted(names, key=lambda n: int(n.split('_')[-1])):
        raise ValueError("a canonical complete team is required")
    vectors = np.stack([obs[n]["vector"] for n in names])
    graphics = [obs[n]["graphic"] for n in names]
    if vectors.shape != (5, 96) or vectors.dtype != np.float32:
        raise ValueError("vector contract changed")
    if any(g.shape != (96, 96, 11) or g.dtype != np.float32 for g in graphics):
        raise ValueError("graphic contract changed")
    if not np.isfinite(vectors).all() or any(not np.isfinite(g).all() for g in graphics):
        raise ValueError("nonfinite observations")
    if not all(np.array_equal(g, graphics[0]) for g in graphics[1:]):
        raise ValueError("team maps differ; cannot share scene")
    if not all(np.array_equal(v[:90], vectors[0, :90]) and np.array_equal(v[93:], vectors[0, 93:]) for v in vectors[1:]):
        raise ValueError("team vectors do not share entity/context observation")
    return vectors.copy(), np.moveaxis(graphics[0], -1, 0).copy()


def tensor_record(mapping):
    result = {}
    for key, value in mapping.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("state/choice must contain tensors: " + key)
        if value.shape[0] != 1:
            raise ValueError("collector owns one team stream")
        result[key] = value.detach().cpu().numpy()[0].copy()
    return result


class EpisodeWriter:
    def __init__(self, path, capacity=22000, chunk_size=64):
        self.path, self.capacity = Path(path), int(capacity)
        self.path.mkdir(parents=True, exist_ok=False)
        self.chunk_size, self.length = int(chunk_size), 0
        self.pending, self.chunks, self.steps, self.potentials = [], [], [], []

    def add(self, *, step, vectors, graphic, decision_state, choices, log_prob,
            potential, teacher_choices=None, on_policy=True):
        if self.length >= self.capacity or (self.steps and step <= self.steps[-1]):
            raise ValueError("event capacity/order violated")
        row = {"vectors": np.asarray(vectors, np.float32).copy(),
               "log_probs": np.asarray(log_prob, np.float32),
               "on_policy": np.asarray(on_policy, bool)}
        row.update({"state__"+k: v for k, v in tensor_record(decision_state).items()})
        row.update({"choice__"+k: v for k, v in tensor_record(choices).items()})
        if teacher_choices:
            row.update({"teacher__"+k: v for k, v in tensor_record(teacher_choices).items()})
        for key, value in row.items():
            if not np.isfinite(value).all():
                raise ValueError("nonfinite trajectory field: " + key)
        row["graphic"] = np.asarray(graphic, np.float32).copy()
        self.pending.append(row)
        self.steps.append(int(step))
        self.potentials.append(float(potential))
        self.length += 1
        if len(self.pending) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        keys = set(self.pending[0]) - {"graphic"}
        if any(set(row)-{"graphic"} != keys for row in self.pending):
            raise ValueError("event schema changed within episode")
        arrays = {k: np.stack([r[k] for r in self.pending]) for k in sorted(keys)}
        maps = np.stack([r["graphic"] for r in self.pending])
        binary = np.logical_or(maps == 0, maps == 1).all((1, 2, 3)) & (maps.sum(1) == 1).all((1, 2))
        arrays["maps"] = maps.argmax(1).astype(np.uint8)
        arrays["float_indices"] = np.flatnonzero(~binary)
        arrays["float_graphics"] = maps[~binary]
        name = f"events-{len(self.chunks):05d}.npz"
        temp = self.path / (name+".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path/name)
        self.chunks.append({"file": name, "start": self.length-len(self.pending), "length": len(self.pending)})
        self.pending.clear()

    def commit(self, outcome, gamma, lam, metadata, env_steps=None):
        if not self.length or not metadata.get("natural_terminal"):
            raise ValueError("incomplete episode cannot be committed")
        env_steps = int(metadata.get("env_steps", env_steps or 0))
        if env_steps <= self.steps[-1] or self.steps[0] != 0:
            raise ValueError("event coverage must begin at zero and end at natural terminal")
        self.flush()
        durations = np.diff(np.r_[self.steps, env_steps]).astype(np.int64)
        rewards = shape_episode(self.potentials, durations, outcome, gamma)
        for key, value in (("durations", durations), ("rewards", rewards), ("event_steps", self.steps)):
            np.save(self.path/(key+".npy"), np.asarray(value), allow_pickle=False)
        record = dict(metadata, length=self.length, decisions=self.length, env_steps=env_steps,
                      outcome=float(outcome), valid=True, chunks=self.chunks, format="v11-events-v1",
                      train_return=float(np.dot(gamma ** np.asarray(self.steps), rewards)),
                      gamma=float(gamma), gae_lambda=float(lam), compressed_lossless=True)
        atomic_json(self.path/"validated.json", record)
        return record


class Episode:
    def __init__(self, path, validated=False):
        self.path = Path(path)
        self.meta = read_json(self.path/("validated.json" if validated else "committed.json"))
        if not self.meta.get("valid") or not self.meta.get("natural_terminal"):
            raise ValueError("not a complete valid episode")
        self.length, self.env_steps = self.meta["length"], self.meta["env_steps"]
        self.chunks = self.meta["chunks"]
        self.cache = OrderedDict()
        keys = ["durations", "rewards", "event_steps"] + ([] if validated else ["values", "advantages", "returns"])
        self.arrays = {k: np.load(self.path/(k+".npy"), mmap_mode="r", allow_pickle=False) for k in keys}

    def _chunk(self, index):
        if index not in self.cache:
            with np.load(self.path/self.chunks[index]["file"], allow_pickle=False) as data:
                self.cache[index] = {k: data[k] for k in data.files}
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def read(self, indices, device="cpu"):
        indices = np.asarray(indices, np.int64)
        if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= self.length:
            raise ValueError("invalid event indices")
        rows = []
        starts = np.asarray([c["start"] for c in self.chunks])
        for index in indices:
            ci = int(np.searchsorted(starts, index, side="right")-1)
            chunk = self._chunk(ci)
            local = int(index-starts[ci])
            row = {k: a[local] for k, a in chunk.items() if k not in ("maps", "float_indices", "float_graphics")}
            fallbacks = np.flatnonzero(chunk["float_indices"] == local)
            row["graphic"] = (chunk["float_graphics"][int(fallbacks[0])] if len(fallbacks) else
                              np.eye(11, dtype=np.float32)[chunk["maps"][local]].transpose(2, 0, 1))
            rows.append(row)
        packed = {k: torch.from_numpy(np.stack([r[k] for r in rows])).to(device) for k in rows[0]}
        result = {k: v for k, v in packed.items() if "__" not in k}
        for prefix, key in (("state__", "state"), ("choice__", "choices"), ("teacher__", "teacher_choices")):
            result[key] = {k[len(prefix):]: v for k, v in packed.items() if k.startswith(prefix)}
        for key, array in self.arrays.items():
            result[key] = torch.from_numpy(np.array(array[indices])).to(device)
        return result


class WaveBatch:
    def __init__(self, paths):
        self.episodes = [Episode(p) for p in paths]
        if not self.episodes:
            raise ValueError("empty wave")
        if len({e.meta["policy_sha256"] for e in self.episodes}) != 1:
            raise ValueError("mixed behavior snapshots")
        if len({e.meta["critic_sha256"] for e in self.episodes}) != 1:
            raise ValueError("mixed frozen target critics")
        self.env_steps = sum(e.env_steps for e in self.episodes)
        self.advantages = np.concatenate([e.arrays["advantages"] for e in self.episodes]).copy()
        self.returns = np.concatenate([e.arrays["returns"] for e in self.episodes]).copy()

    def __len__(self):
        return sum(e.length for e in self.episodes)

    def sequences(self, burn_in=32, length=64):
        return [(ep, max(0, start-burn_in), start, min(ep.length, start+length))
                for ep in self.episodes for start in range(0, ep.length, length)]
