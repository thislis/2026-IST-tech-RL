"""Small frozen-history pool with family floors and within-history PFSP."""
import numpy as np
from .io import atomic_torch, cpu_state, file_hash


class League:
    def __init__(self, state=None, capacity=8):
        self.entries = [] if state is None else state
        self.capacity = capacity

    def sample(self, rng, history=True, pilot=False):
        if pilot:
            return {"kind": str(rng.choice(["noop", "random", "collector"], p=[.45, .25, .3]))}
        group = rng.choice(["easy", "strategy", "target", "history"], p=[.2, .3, .2, .3])
        if group == "history" and history and self.entries:
            difficulty = np.asarray([.05 + (1-(e["wins"]+1)/(e["games"]+2))**2 for e in self.entries])
            entry = self.entries[int(rng.choice(len(self.entries), p=difficulty/difficulty.sum()))]
            return {"kind": "history", "path": entry["path"], "sha256": entry["sha256"]}
        if group == "target":
            return {"kind": "target"}
        if group == "easy":
            return {"kind": str(rng.choice(["noop", "random"]))}
        return {"kind": str(rng.choice(["collector", "rush", "raider", "defender"]))}

    def observe(self, spec, outcome):
        if spec["kind"] != "history":
            return
        for entry in self.entries:
            if entry["sha256"] == spec["sha256"]:
                entry["games"] += 1
                entry["wins"] += (outcome+1)/2

    def snapshot(self, policy, directory, step, descriptor):
        from pathlib import Path
        import time
        path = Path(directory) / f"step-{step}-{time.time_ns()}.pt"
        atomic_torch(path, {"policy_state": cpu_state(policy)})
        entry = {"path": str(path), "sha256": file_hash(path), "step": step,
                 "wins": 0., "games": 0, "descriptor": list(map(float, descriptor))}
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            # Keep newest and a spread of observed descriptors (not just age).
            descriptors = np.asarray([e["descriptor"] for e in self.entries])
            selected = [len(self.entries)-1]
            while len(selected) < self.capacity:
                distance = np.min(((descriptors[:, None]-descriptors[selected][None])**2).sum(-1), axis=1)
                distance[selected] = -1
                selected.append(int(distance.argmax()))
            self.entries = [self.entries[i] for i in sorted(selected)]
        return entry
