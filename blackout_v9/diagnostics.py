"""Visible trajectory diagnostics; no inferred event is reported as engine truth."""
import numpy as np


class Behavior:
    def __init__(self):
        self.steps = self.delivery_proxy = self.pickup_proxy = self.raid_proxy = 0
        self.moving = self.same_actions = self.respawn_proxy = 0
        self.score_gain = 0.
        self.actions = np.zeros(9, dtype=np.int64)
        self.greedy_actions = np.zeros(9, dtype=np.int64)
        self.greedy_same = 0

    def observe(self, before, after, actions, terminal=False, greedy=None):
        self.steps += 1
        self.actions += np.bincount(actions, minlength=9)
        self.same_actions += int(np.all(actions == actions[0]))
        if greedy is not None:
            self.greedy_actions += np.bincount(greedy, minlength=9)
            self.greedy_same += int(np.all(greedy == greedy[0]))
        if terminal:
            return np.zeros(3, np.float32)
        a, b = before[0, :90].reshape(10, 9), after[0, :90].reshape(10, 9)
        allies = a[:, 2] > 0
        movement = np.linalg.norm(b[:, :2]-a[:, :2], axis=1)
        self.moving += int(np.any(movement[allies] > 1e-5))
        self.respawn_proxy += int(np.count_nonzero(movement > .2))
        was, now = a[:, 3:].argmax(-1), b[:, 3:].argmax(-1)
        pickup = int(np.count_nonzero(allies & (was == 0) & (now == 1)))
        score_delta = float(after[0, 93]-before[0, 93])
        other_delta = float(after[0, 94]-before[0, 94])
        self.pickup_proxy += pickup
        self.delivery_proxy += int(score_delta > 0 and np.any(allies & (was == 1) & (now == 0)))
        self.raid_proxy += int(other_delta < 0 and pickup > 0)
        self.score_gain += max(0., score_delta)
        return np.asarray([score_delta, float(np.mean(was[allies] != now[allies])), float(movement[allies].mean())], np.float32)

    def report(self):
        return {"steps": self.steps, "moving_step_fraction": self.moving/max(1, self.steps),
                "same_action_fraction": self.same_actions/max(1, self.steps),
                "action_histogram": self.actions.tolist(), "pickup_proxy": self.pickup_proxy,
                "greedy_action_histogram": self.greedy_actions.tolist(),
                "greedy_same_action_fraction": self.greedy_same/max(1, self.steps),
                "delivery_proxy": self.delivery_proxy, "raid_proxy": self.raid_proxy,
                "respawn_proxy": self.respawn_proxy, "visible_positive_score_delta": self.score_gain,
                "event_truth": "observation proxies only; terminal reset excluded"}
