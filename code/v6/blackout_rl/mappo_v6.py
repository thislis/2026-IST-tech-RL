"""V6 inference: one explicit team decision, real planner context, shared semantics.

Only observation-derived information enters the actor. The team coordinator
chooses KEEP or one (slot, correction) from a 41-way categorical distribution.
This is a coordinated team actor, not five independent decentralized samples.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batching import stack_observations, team_agents
from .ippo_model import IPPOActorCritic
from .navigation import PathNotFound
from .representation import absorption_phase_features
from .scripted_fsm import ScriptedTeamController, HUNTER_SHRINE_CELLS, carrier_shrine_cell
from .team_state import Role

SCHEMA_VERSION = "blackout.mappo_planner_residual.v6"
GUARDRAIL_MODE = "planner_joint_residual_v6"
ROLES = (Role.WORKER, Role.WORKER, Role.WORKER, Role.GUARD, Role.GUARD)
CONTEXT_SIZE = 49
CENTRAL_SIZE = 123 + 11 * 4 * 4


def correction_actions(planner: np.ndarray) -> np.ndarray:
    """Eight corrections around the *continuous* planner vector.

    A stopped planner exposes all eight compass directions (no duplicate stop).
    A moving planner exposes +/-45, +/-90, stop, reverse, +/-135 degrees.
    """
    actions = np.zeros((5, 8, 2), dtype=np.float32)
    angles = (45, -45, 90, -90, None, 180, 135, -135)
    for row, direction in enumerate(planner):
        norm = float(np.linalg.norm(direction))
        for column, degrees in enumerate(angles):
            if norm < 1e-6:
                radians = column * math.pi / 4
                actions[row, column] = (math.cos(radians), math.sin(radians))
            elif degrees is not None:
                radians = math.radians(degrees)
                x, y = direction / norm
                actions[row, column] = (x * math.cos(radians) - y * math.sin(radians),
                                        x * math.sin(radians) + y * math.cos(radians))
    return actions


def joint_distribution(logits: torch.Tensor, valid: torch.Tensor,
                       exploration: float = 0.0) -> torch.distributions.Categorical:
    """Use this exact distribution for sampling AND PPO log-prob evaluation.

    exploration is explicit probability mass over legal corrections. It is
    frozen per rollout and stored in the buffer, so the importance ratio uses
    the actual behavior policy. Greedy inference uses the same 41-way logits.
    """
    if not math.isfinite(exploration) or not 0 <= exploration < 1:
        raise ValueError("exploration must be finite in [0,1)")
    if logits.shape != valid.shape or logits.shape[-1] != 41 or valid.dtype != torch.bool:
        raise ValueError("expected matching (...,41) logits and boolean mask")
    if not bool(valid[..., 0].all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("KEEP must be valid and all logits must be finite")
    probabilities = logits.masked_fill(~valid, -torch.inf).softmax(-1)
    explore = valid.to(logits.dtype).clone()
    explore[..., 0] = 0
    count = explore.sum(-1, keepdim=True)
    explore /= count.clamp_min(1)
    floor = exploration * (count > 0).to(logits.dtype)
    return torch.distributions.Categorical(probs=(1 - floor) * probabilities + floor * explore)


class V6Model(nn.Module):
    def __init__(self, actor: IPPOActorCritic, initial_override_probability: float = 0.10):
        super().__init__()
        if not 0 < initial_override_probability < 0.5:
            raise ValueError("initial override probability must be in (0,0.5)")
        if actor.model_config["recurrent_version"] != "none":
            raise ValueError("v6 requires a non-recurrent neural encoder")
        self.actor_model = actor
        self.actor_model.requires_grad_(False)
        hidden = int(actor.model_config["hidden_dim"])
        self.residual = nn.Sequential(nn.Linear(hidden + CONTEXT_SIZE, hidden), nn.Tanh(), nn.Linear(hidden, 8))
        self.critic = nn.Sequential(nn.Linear(CENTRAL_SIZE, 256), nn.Tanh(),
                                    nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        final = self.residual[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(initial_override_probability / (40 * (1-initial_override_probability))))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-2] != 5:
            raise ValueError("team features must retain all five slots")
        corrections = self.residual(features).flatten(-2)
        return torch.cat((torch.zeros_like(corrections[..., :1]), corrections), dim=-1)


class PlannerFeatures:
    """State shared by collection, evaluation, and the standalone export."""
    def __init__(self, team: int):
        self.team = team
        self.agents = team_agents(team)
        self.planner = ScriptedTeamController(team, roles=ROLES, chase_radius_cells=48)
        self.reset()

    def reset(self):
        self.planner.reset()
        self.previous_positions = None
        self.stall = np.zeros(5, dtype=np.float32)
        self.failures = 0

    def prepare(self, observations: Mapping, model: V6Model):
        device = next(model.parameters()).device
        batch = stack_observations(observations, agents=self.agents)
        vector = torch.as_tensor(batch.vectors, device=device)
        graphic = torch.as_tensor(batch.graphics_chw, device=device)
        slots = torch.arange(5, device=device)
        positions = batch.vectors[0, :90].reshape(10, 9)[:, :2]
        own = positions[self.team*5:self.team*5+5]
        displacement = np.zeros_like(own) if self.previous_positions is None else own-self.previous_positions
        self.stall = np.where(np.linalg.norm(displacement, axis=1) < 1e-4, self.stall+1, 0)
        self.previous_positions = own.copy()
        try:
            action_dict = self.planner.act(observations, self.agents)
            path_ok = True
        except PathNotFound:
            self.failures += 1
            self.planner.reset()
            action_dict = {a: np.zeros(2, dtype=np.float32) for a in self.agents}
            path_ok = False
        planner = np.stack([action_dict[a] for a in self.agents]).astype(np.float32)
        alternatives = (correction_actions(planner[:, ::-1])[:, :, ::-1].copy()
                        if self.team else correction_actions(planner))
        context = []
        valid = np.ones((5, 8), dtype=bool)
        for row, agent in enumerate(self.agents):
            block = batch.vectors[row, :90].reshape(10, 9)
            self_block = block[self.team*5+row]
            target = self.planner.targets.get(agent) if path_ok else None
            delta = np.zeros(2, dtype=np.float32) if target is None else np.asarray(((target.x+.5)/24, (target.y+.5)/24))-own[row]
            enemy_delta = positions[(1-self.team)*5:(1-self.team)*5+5]-own[row]
            nearest = enemy_delta[np.linalg.norm(enemy_delta, axis=1).argmin()]
            relative = np.concatenate((positions[self.team*5:self.team*5+5],
                                       positions[(1-self.team)*5:(1-self.team)*5+5]))-own[row]
            # The game mirrors sides over y=x, not by rotating 180 degrees.
            orient = lambda x: np.asarray(x)[..., ::-1] if self.team else np.asarray(x)
            phase = absorption_phase_features(vector[row:row+1, -1]).cpu().numpy()[0]
            context.append(np.concatenate((
                orient(planner[row]), orient(delta),
                [float(path_ok and bool(self.planner.paths.get(agent))), float(target is not None)],
                np.eye(3)[0 if row < 3 else 1], batch.vectors[row, 90:93], self_block[3:9],
                phase, batch.vectors[row, 93:95], [float(self.team)], orient(displacement[row]),
                [min(float(self.stall[row])/30, 1)], orient(nearest), [np.linalg.norm(nearest)],
                orient(relative).reshape(-1))))
            # Predict only a short swept move; this is a geometry filter, not a
            # claim to reproduce all Unity contacts. KEEP always stays available.
            walls = batch.graphics_chw[row, 1]
            forbidden = set(HUNTER_SHRINE_CELLS) | {carrier_shrine_cell(self.team)} if row < 3 else {carrier_shrine_cell(self.team)}
            for column, direction in enumerate(alternatives[row]):
                if np.allclose(direction, planner[row], atol=1e-5):
                    valid[row, column] = False
                    continue
                endpoint = own[row] + direction * (9 * .02 / 24)
                for fraction in (.5, 1.0):
                    point = own[row] + fraction * (endpoint-own[row])
                    # Unit radius margin (wall pixels are top-down).
                    for offset in ((0,0), (.014,0), (-.014,0), (0,.014), (0,-.014)):
                        x, y = point + offset
                        if not (0 <= x < 1 and 0 <= y < 1):
                            valid[row, column] = False
                            continue
                        py, px = min(int((1-y)*walls.shape[0]), walls.shape[0]-1), int(x*walls.shape[1])
                        if walls[py, px] > .5:
                            valid[row, column] = False
                    cell = (int(point[0]*24), int(point[1]*24))
                    if any(cell == (c.x, c.y) for c in forbidden):
                        valid[row, column] = False
        context_tensor = torch.as_tensor(np.asarray(context, dtype=np.float32), device=device)
        with torch.no_grad():
            latent = model.actor_model.encode(vector, graphic, slots)
        features = torch.cat((latent, context_tensor), dim=-1)
        mask = torch.as_tensor(np.concatenate(([True], valid.reshape(-1))), device=device)
        return features, mask, planner, alternatives


def apply_joint_action(index: int, planner: np.ndarray, alternatives: np.ndarray) -> np.ndarray:
    if not 0 <= index < 41:
        raise ValueError("joint action must be in [0,40]")
    result = planner.copy()
    if index:
        row, column = divmod(index-1, 8)
        result[row] = alternatives[row, column]
    return result


class V6Policy:
    def __init__(self, model: V6Model, *, team: int, seed: int = 0):
        self.model = model.eval()
        self.team, self.seed = team, seed
        self.context = PlannerFeatures(team)
        self.last_time = None
        self.last_input = None
        self.last_result = None

    def reset(self):
        self.context.reset()
        self.last_time = self.last_input = self.last_result = None

    def act(self, observations: Mapping, agents):
        expected = team_agents(self.team)
        if set(agents) != set(expected):
            raise ValueError("v6 policy requires exactly its five named agents")
        time_left = float(observations[expected[0]]["vector"][-1])
        if self.last_time is not None and time_left > self.last_time + 1e-6:
            self.reset()
        # Stateless-call determinism: repeated identical input must not advance FSM.
        current = tuple((observations[a]["vector"], observations[a]["graphic"]) for a in expected)
        if self.last_input is not None and all(np.array_equal(x, y) for xs, ys in zip(current, self.last_input) for x, y in zip(xs, ys)):
            return {a: self.last_result[i].copy() for i, a in enumerate(expected)}
        with torch.inference_mode():
            features, mask, planner, alternatives = self.context.prepare(observations, self.model)
            logits = self.model.logits(features)
            index = int(joint_distribution(logits, mask).probs.argmax())
            actions = apply_joint_action(index, planner, alternatives)
        self.last_input = tuple(tuple(x.copy() for x in pair) for pair in current)
        self.last_time, self.last_result = time_left, actions
        return {a: actions[i].copy() for i, a in enumerate(expected)}


def model_from_payload(payload: Mapping, device: str = "cpu") -> V6Model:
    if payload.get("v6_schema") != SCHEMA_VERSION:
        raise ValueError("checkpoint has no compatible v6 state")
    config = dict(payload["model_config"])
    config.pop("n_actions", None)
    with torch.random.fork_rng(devices=[]):
        model = V6Model(IPPOActorCritic(**config)).to(device)
    model.load_state_dict(payload["v6_model_state"], strict=True)
    return model.eval()
