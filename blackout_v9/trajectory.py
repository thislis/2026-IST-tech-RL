"""Disk-backed complete-episode staging. Partial episodes are never PPO batches."""
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, read_json
from .rewards import gae_episode, shape_episode

FIELDS = {"vectors": ((5, 96), "float32"), "actions": ((5,), "int64"),
          "log_probs": ((5,), "float32"), "values": ((), "float32"),
          "potentials": ((), "float64"), "maps": ((96, 96), "uint8"),
          "aux_targets": ((3,), "float32"), "aux_valid": ((), "float32")}


def pack(obs, names):
    if len(names) != 5:
        raise ValueError("a complete team is required")
    vectors = np.stack([obs[n]["vector"] for n in names])
    graphics = [obs[n]["graphic"] for n in names]
    if vectors.shape != (5, 96) or vectors.dtype != np.float32:
        raise ValueError("vector contract changed")
    if any(g.shape != (96, 96, 11) or g.dtype != np.float32 for g in graphics):
        raise ValueError("graphic contract changed")
    if not np.isfinite(vectors).all() or any(not np.isfinite(g).all() for g in graphics):
        raise ValueError("nonfinite observation")
    if not all(np.array_equal(g, graphics[0]) for g in graphics[1:]):
        raise ValueError("team graphic no longer shared; compression contract must be revised")
    if not all(np.array_equal(v[:90], vectors[0, :90]) and np.array_equal(v[93:], vectors[0, 93:]) for v in vectors[1:]):
        raise ValueError("team scene no longer shared")
    return vectors.copy(), np.moveaxis(graphics[0], -1, 0).copy()


class EpisodeWriter:
    def __init__(self, path, capacity):
        self.path, self.capacity, self.length = Path(path), capacity, 0
        self.path.mkdir(parents=True, exist_ok=False)
        self.arrays = {key: np.lib.format.open_memmap(self.path / (key + ".npy"), mode="w+", dtype=dtype,
                       shape=(capacity, *shape)) for key, (shape, dtype) in FIELDS.items()}
        self.fallback = []

    def add(self, *, vectors, graphic, actions, log_probs, values, potentials, aux_targets, aux_valid=1.):
        if self.length >= self.capacity:
            raise RuntimeError("episode step watchdog")
        i = self.length
        binary = np.logical_or(graphic == 0, graphic == 1).all() and (graphic.sum(0) == 1).all()
        if binary:
            self.arrays["maps"][i] = graphic.argmax(0).astype(np.uint8)
        else:
            # No lossy argmax fallback: preserve the complete original graphic.
            folder = self.path / "float_graphics"
            folder.mkdir(exist_ok=True)
            np.save(folder / f"{i}.npy", graphic, allow_pickle=False)
            self.fallback.append(i)
            self.arrays["maps"][i] = 0
        for key, value in (("vectors", vectors), ("actions", actions), ("log_probs", log_probs),
                           ("values", values), ("potentials", potentials), ("aux_targets", aux_targets), ("aux_valid", aux_valid)):
            if not np.isfinite(value).all():
                raise ValueError("nonfinite trajectory field: " + key)
            self.arrays[key][i] = value
        self.length += 1

    def commit(self, outcome, gamma, lam, metadata):
        if not self.length or not metadata.get("natural_terminal"):
            raise ValueError("cannot commit incomplete/nonterminal episode")
        for array in self.arrays.values():
            array.flush()
        rewards = shape_episode(self.arrays["potentials"][:self.length], outcome, gamma)
        advantages, returns = gae_episode(rewards, self.arrays["values"][:self.length], gamma, lam)
        for key, value in (("rewards", rewards), ("advantages", advantages), ("returns", returns)):
            np.save(self.path / (key + ".npy"), value, allow_pickle=False)
        record = dict(metadata, length=self.length, outcome=outcome, train_return=float(rewards.sum(dtype=np.float64)),
                      compressed=True, float_graphics=self.fallback, valid=True)
        atomic_json(self.path / "committed.json", record)
        return record


class Episode:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = read_json(self.path / "committed.json")
        if not self.meta["valid"] or not self.meta["natural_terminal"]:
            raise ValueError("uncommitted episode")
        self.length = self.meta["length"]
        self.arrays = {k: np.load(self.path / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                       for k in list(FIELDS) + ["rewards", "advantages", "returns"]}
        self.fallback = set(self.meta["float_graphics"])

    def graphic(self, indices):
        ids = np.asarray(self.arrays["maps"][indices])
        graphic = np.eye(11, dtype=np.float32)[ids].transpose(0, 3, 1, 2)
        for local, index in enumerate(indices):
            if int(index) in self.fallback:
                graphic[local] = np.load(self.path / "float_graphics" / f"{index}.npy", allow_pickle=False)
        return graphic


class WaveBatch:
    def __init__(self, paths):
        self.episodes = [Episode(p) for p in paths]
        if not self.episodes:
            raise ValueError("empty valid wave")
        versions = {e.meta["policy_sha256"] for e in self.episodes}
        if len(versions) != 1:
            raise ValueError("mixed behavior policy versions")
        self.offsets = np.cumsum([0] + [e.length for e in self.episodes])
        self.advantages = np.concatenate([e.arrays["advantages"] for e in self.episodes]).copy()
        self.returns = np.concatenate([e.arrays["returns"] for e in self.episodes]).copy()

    def __len__(self):
        return int(self.offsets[-1])

    def minibatch(self, indices, device):
        indices = np.asarray(indices)
        fields = ("vectors", "actions", "log_probs", "values", "aux_targets", "aux_valid", "returns")
        result = {k: np.empty((len(indices), *self.episodes[0].arrays[k].shape[1:]),
                             dtype=self.episodes[0].arrays[k].dtype) for k in fields}
        graphics = np.empty((len(indices), 11, 96, 96), np.float32)
        which = np.searchsorted(self.offsets[1:], indices, side="right")
        for i, episode in enumerate(self.episodes):
            rows = np.flatnonzero(which == i)
            if not len(rows):
                continue
            local = indices[rows] - self.offsets[i]
            for key in fields:
                result[key][rows] = episode.arrays[key][local]
            graphics[rows] = episode.graphic(local)
        result["graphic"] = graphics
        return {k: torch.from_numpy(v).to(device) for k, v in result.items()}
