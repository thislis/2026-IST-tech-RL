"""Episode allocation by measured family/side step deficits, never episode %s."""
import copy
from pathlib import Path
from .io import digest, read_json

FAMILY_WEIGHTS = {"planner": .4, "scripted": .3, "self_play": .2, "direct": .1}


class DeficitScheduler:
    def __init__(self, catalog, state=None, weights=None):
        self.catalog = catalog
        self.weights = dict(weights or FAMILY_WEIGHTS)
        if set(self.weights) != set(FAMILY_WEIGHTS) or abs(sum(self.weights.values())-1) > 1e-7:
            raise ValueError("family weights must define normalized planner/scripted/self_play/direct")
        self.data = copy.deepcopy(state) if state else {
            "families": {f: {"env_steps": 0, "episodes": 0, "assigned": 0} for f in self.weights},
            "sides": {"0": 0, "1": 0}, "side_assigned": {"0": 0, "1": 0}, "serial": 0}

    def next(self, seed, self_play=None, reserve=None):
        # Reservations prevent all jobs in one wave receiving the same deficit.
        reserve = {} if reserve is None else reserve
        families = self.data["families"]
        total = sum(x["env_steps"] for x in families.values()) + sum(reserve.values())
        horizon = max(1000., total + 5000.)
        family = max(self.weights, key=lambda f: (
            self.weights[f]*horizon - families[f]["env_steps"] - reserve.get(f, 0),
            -families[f]["episodes"], -families[f]["assigned"]))
        candidates = list(self.catalog.get(family, []))
        if family == "self_play" and self_play:
            candidates = list(self_play)
        if not candidates:
            raise ValueError("no registered opponents for family " + family)
        index = families[family]["assigned"] % len(candidates)
        opponent = dict(candidates[index], family=family)
        side = min((0, 1), key=lambda s: (self.data["sides"][str(s)], self.data["side_assigned"][str(s)]))
        serial = self.data["serial"]
        self.data["serial"] += 1
        families[family]["assigned"] += 1
        self.data["side_assigned"][str(side)] += 1
        mean = families[family]["env_steps"] / max(1, families[family]["episodes"])
        reserve[family] = reserve.get(family, 0) + max(1000., mean or 10000.)
        splits = read_json(Path(__file__).parents[1]/"configs/splits.json")
        bounds = splits["train"]
        map_seed = bounds[0] + int(digest([splits["seed"], "train", int(seed), serial])[:12], 16) % (bounds[1]-bounds[0]+1)
        return {"side": side, "opponent": opponent, "requested_seed": map_seed,
                "action_seed": int(seed)*1000003 + serial*17 + 101, "family": family}

    def observe(self, family, side, env_steps):
        if env_steps < 1:
            raise ValueError("only completed episodes enter scheduler exposure")
        self.data["families"][family]["env_steps"] += int(env_steps)
        self.data["families"][family]["episodes"] += 1
        self.data["sides"][str(side)] += int(env_steps)

    def state_dict(self):
        return copy.deepcopy(self.data)

    def report(self):
        result = self.state_dict()
        total = sum(x["env_steps"] for x in result["families"].values())
        for family, row in result["families"].items():
            row["target_fraction"] = self.weights[family]
            row["actual_fraction"] = row["env_steps"] / max(1, total)
        result["total_env_steps"] = total
        return result
