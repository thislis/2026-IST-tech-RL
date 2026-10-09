"""Same legacy features without the legacy mask that v8 always discarded.

Copied from the pinned mappo_v6.PlannerFeatures.prepare; planner, numeric
feature ordering and frozen encoder computation are unchanged. Only the
learner factory selects this class; frozen opponent code is untouched.
"""
from __future__ import annotations
import numpy as np
import torch
from blackout_rl.batching import stack_observations
from blackout_rl.mappo_v6 import correction_actions
from blackout_rl.navigation import PathNotFound
from blackout_rl.representation import absorption_phase_features
from .planner_adapter import PlannerAdapter
from .contracts import ContractError


class FastPlannerAdapter(PlannerAdapter):
    def prepare(self, observations, model):
        failures = self.failures
        result = self._features(observations, model)
        if self.failures != failures:
            raise ContractError('unisolated planner failure; invalidate attempt')
        self.teacher_valid = {a: a not in self.planner.unreachable for a in self.agents}
        return result

    def _features(self, observations: Mapping, model: V6Model):
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
        context_tensor = torch.as_tensor(np.asarray(context, dtype=np.float32), device=device)
        with torch.no_grad():
            latent = model.actor_model.encode(vector, graphic, slots)
        features = torch.cat((latent, context_tensor), dim=-1)
        mask = None  # Deliberately unused by v8; its reasoned mask is built once in the factory.
        return features, mask, planner, alternatives
