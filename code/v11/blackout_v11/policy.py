"""Standalone v11 team policy (torch and the standard library only).

The supplied run_match/load_checkpoint path calls each team in canonical unit
order. A policy instance owns one such five-row team stream. Runtime memory is
explicit in ``step`` and transient in the two-input competition wrapper.
"""
import math
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('P', 'R', 'G', 'F', 'D')
INTENTS = ('collect', 'deliver', 'special', 'intercept', 'defend', 'raid', 'continue')
CANDIDATES = 31
INF = 100000.0


@torch.jit.script
def _relax(walk: torch.Tensor, goals: torch.Tensor, neighbours: torch.Tensor,
           edge_ok: torch.Tensor, costs: torch.Tensor, max_iterations: int = 575):
    """Exact positive-cost 24x24 grid distances; ONNX exports a bounded Loop."""
    size = walk.size(0)
    vertices = torch.arange(576, device=walk.device).unsqueeze(0)
    distance = torch.where(vertices == goals.unsqueeze(1), torch.zeros_like(walk, dtype=torch.float32),
                           torch.full_like(walk, 100000.0, dtype=torch.float32))
    distance = torch.where(walk, distance, torch.full_like(distance, 100000.0))
    iteration = 0
    changed = True
    while iteration < max_iterations and changed:
        nearby = distance[:, neighbours.reshape(-1)].reshape(size, 576, 8)
        candidate = torch.where(edge_ok, nearby + costs, torch.full_like(nearby, 100000.0))
        updated = torch.minimum(distance, torch.min(candidate, dim=-1)[0])
        changed = bool(torch.any(updated != distance))
        distance = updated
        iteration += 1
    return distance


def _where(condition, yes, no):
    # ORT's ARM CPU provider has no bool-valued Where kernel; boolean logic is
    # equivalent and portable, while numeric tensors retain regular Where.
    if yes.dtype == torch.bool and no.dtype == torch.bool:
        return (condition & yes) | (~condition & no)
    return torch.where(condition, yes, no)


def move_state(state, device):
    return {k: v.to(device) for k, v in state.items()}


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


def _take(values, index):
    return values.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def _xy(cell):
    return torch.stack((cell.remainder(24), torch.div(cell, 24, rounding_mode='floor')), -1).float().add(.5).div(24)


def _cell(position):
    p = (position * 24).long().clamp(0, 23)
    return p[..., 1] * 24 + p[..., 0]


def own_entities(vector):
    entities = vector[:, 0, :90].reshape(-1, 10, 9)
    # Positive entities are sorted by their original absolute unit index.
    order = torch.arange(10, device=vector.device)[None].expand(len(vector), -1)
    selected = _where(entities[..., 2] > 0, order, order + 10).argsort(-1)[:, :5]
    return entities.gather(1, selected[..., None].expand(-1, -1, 9))



def canonical_observation(vector, graphic):
    """Map physical B to A by the game's y=x reflection, keeping unit order."""
    swap = vector[:, 0, 2] < 0
    entities = vector[..., :90].reshape(vector.shape[0], 5, 10, 9)
    position = _where(swap[:, None, None, None], entities[..., :2].flip(-1), entities[..., :2])
    entities = torch.cat((position, entities[..., 2:]), -1)
    normalized = torch.cat((entities.reshape(vector.shape[0], 5, 90), vector[..., 90:]), -1)
    graphic = _where(swap[:, None, None, None], graphic.transpose(-2, -1).flip((-2, -1)), graphic)
    return normalized, graphic, swap

STATE_KEYS = ('unit_hidden', 'team_hidden', 'goal', 'intent', 'role', 'goal_age',
              'residual_remaining', 'correction', 'direct', 'previous_action',
              'previous_position', 'previous_holding', 'previous_class',
              'previous_time', 'side', 'step_count', 'stuck_count', 'waypoint',
              'blocked_cell', 'blocked_ttl', 'failed_store', 'failed_storage', 'failed_ttl',
              'delivery_wait', 'walls', 'ally_storage', 'enemy_storage',
              'enemy_base', 'item_memory', 'item_age', 'nav_goal', 'nav_walk', 'nav_distance')


class TensorPlanner(nn.Module):
    """World-y-up semantic decoding and eight-neighbour tensor path planning."""
    def __init__(self):
        super().__init__()
        grid = torch.arange(576)
        x, y = grid % 24, grid // 24
        offsets = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1],
                                [1, 1], [-1, 1], [-1, -1], [1, -1]])
        nx, ny = x[:, None] + offsets[:, 0], y[:, None] + offsets[:, 1]
        inside = (nx >= 0) & (nx < 24) & (ny >= 0) & (ny < 24)
        neighbours = (ny.clamp(0, 23) * 24 + nx.clamp(0, 23)).long()
        horizontal = y[:, None] * 24 + nx.clamp(0, 23)
        vertical = ny.clamp(0, 23) * 24 + x[:, None]
        shrine = ((x == 11) | (x == 12)) & ((y == 11) | (y == 12))
        for name, value in [('neighbours', neighbours), ('inside', inside),
                            ('horizontal', horizontal), ('vertical', vertical),
                            ('diagonal', (offsets[:, 0] != 0) & (offsets[:, 1] != 0)),
                            ('costs', offsets.float().square().sum(-1).sqrt()),
                            ('hunter_shrine', shrine), ('cell_xy', _xy(grid))]:
            self.register_buffer(name, value)

    def edges(self, walk):
        w = walk.reshape(-1, 576)
        neighbour = w[:, self.neighbours.reshape(-1)].reshape(-1, 576, 8)
        horizontal = w[:, self.horizontal.reshape(-1)].reshape(-1, 576, 8)
        vertical = w[:, self.vertical.reshape(-1)].reshape(-1, 576, 8)
        return (w[..., None] & neighbour & self.inside[None] &
                (~self.diagonal[None, None] | (horizontal & vertical)))

    def distance_fields(self, walkable, goal_cells, max_iterations=575):
        """Shape [...,24,24] or [...,576] -> [...,576]; unreachable=100000."""
        shape = goal_cells.shape
        walk = walkable.reshape(-1, 576)
        return _relax(walk, goal_cells.reshape(-1), self.neighbours,
                      self.edges(walk), self.costs, max_iterations).reshape(*shape, 576)

    @staticmethod
    def decode(graphic):
        # Rendering is top-down; all path arrays below use world [y,x].
        return F.max_pool2d(graphic, 4, 4).flip(-2).flatten(-2) > .5

    def walkability(self, state, role, classes, current, holding=None):
        n = role.shape[0]
        walk = ~(state['walls'] | state['enemy_base'])[:, None].expand(-1, 5, -1)
        carrier_cell = role.new_full((n,), 20 * 24 + 1)
        grid = torch.arange(576, device=walk.device)[None, None]
        collector = classes == 0
        carrying = torch.zeros_like(collector) if holding is None else holding > 0
        forbid_hunter = collector & ((role != 1) | carrying)
        forbid_carrier = collector & ((role != 2) | carrying)
        walk = walk & ~(forbid_hunter[..., None] & self.hunter_shrine[None, None])
        walk = walk & ~(forbid_carrier[..., None] & (grid == carrier_cell[:, None, None]))
        blocked = (grid == state['blocked_cell'][..., None]) & (state['blocked_ttl'][..., None] > 0)
        walk = walk & ~blocked
        # A unit already on a now-forbidden tile must be able to leave it.
        walk = walk | ((grid == current[..., None]) & ~state['walls'][:, None])
        return walk

    def candidates(self, vector, semantic, state, own, classes):
        n = len(vector)
        current = _cell(own[..., :2])
        holding = own[..., 3:].argmax(-1)
        role_walks = torch.stack([self.walkability(state, torch.full_like(classes, role), classes, current, holding) for role in range(3)], 2)
        role_distances = self.distance_fields(role_walks, current[...,None].expand(-1,-1,3))
        grid = torch.arange(576, device=vector.device)[None, None]
        entities = vector[:, 0, :90].reshape(n, 10, 9)
        enemy = (entities[..., 2] < 0)
        enemy_cell = _cell(entities[..., :2])
        enemy_tiles = ((grid[:, 0, :, None] == enemy_cell[:, None, :]) & enemy[:, None]).any(-1)
        battery = (semantic[:, 6] | (state['item_memory'] & (state['item_age'] < 12)))
        battery = battery & ~state['ally_storage'] & ~state['enemy_storage'] & ~self.hunter_shrine
        special = semantic[:, 7:11].any(1)
        storage = state['ally_storage'][:, None].expand(-1, 5, -1)
        storage = storage & ~(state['failed_storage'] & (state['failed_ttl'][..., None] > 0))
        holding = own[..., 3:].argmax(-1)
        cells_by_role, intents_by_role, masks_by_role, priors_by_role = [], [], [], []
        for role in range(3):
            walk, distances = role_walks[:,:,role], role_distances[:,:,role]
            class_collector = classes == 0
            carrying = holding > 0
            carrier_cell = current.new_full((n,), 20 * 24 + 1)
            transform = (self.hunter_shrine[None, None].expand(n, 5, -1) if role == 1 else
                         (grid == carrier_cell[:, None, None]).expand(-1, 5, -1))
            target_special = special[:, None].expand(-1, 5, -1)
            if role in (1, 2):
                target_special = _where(class_collector[..., None], transform, target_special)
            reached_defend = (self.cell_xy[None,None] - own[:,:,None,:2]).square().sum(-1) < .024 ** 2
            defend = storage & ~reached_defend
            sets = [battery[:, None].expand(-1, 5, -1), storage, target_special,
                    enemy_tiles[:, None].expand(-1, 5, -1), defend,
                    (state['enemy_storage'] & semantic[:, 6])[:, None].expand(-1, 5, -1)]
            can_collect = (classes != 1) & ~carrying if role != 1 else torch.zeros_like(carrying)
            can_deliver = (classes != 1) & carrying
            can_special = (classes != 1) & ~carrying
            can_intercept = classes == 1
            can_defend = torch.ones_like(carrying)
            can_raid = (classes != 1) & ~carrying if role != 1 else torch.zeros_like(carrying)
            enabled = [can_collect, can_deliver, can_special, can_intercept, can_defend, can_raid]
            all_cells, all_intents, all_mask, all_prior = [], [], [], []
            for intent, target in enumerate(sets):
                valid = target & walk & (distances < INF / 2) & enabled[intent][..., None]
                # Keep guards near the threatened edge of a storage rather than its base corner.
                cost = distances
                if intent == 4:
                    central = (self.cell_xy - .5).square().sum(-1) * 6
                    cost = distances * .25 + central
                rank = _where(valid, cost, torch.full_like(cost, INF))
                _, cells = rank.topk(5, dim=-1, largest=False, sorted=True)
                legal = valid.gather(-1, cells)
                path = distances.gather(-1, cells)
                preference = [8., 20., 1., 7., -2., 0.][intent]
                prior = preference - .20 * path
                if intent == 2 and role in (1, 2):
                    prior = prior + class_collector.float()[..., None] * 18
                if intent == 4 and role == 1:
                    prior = prior + (classes == 1).float()[..., None] * 4
                if intent == 3:
                    prior = prior - (path > 12).float() * 12
                all_cells.append(cells)
                all_intents.append(torch.full_like(cells, intent))
                all_mask.append(legal)
                all_prior.append(prior)
            continuation = state['goal'].clamp(0, 575)
            continuation_valid = state['goal'] >= 0
            old_intent = state['intent'].clamp(0, 5)
            still_target = torch.zeros_like(continuation_valid)
            for intent, target in enumerate(sets):
                still_target = still_target | ((old_intent == intent) & _take(target, continuation) & enabled[intent])
            continuation_valid = continuation_valid & still_target & (_take(distances, continuation) < INF / 2)
            continue_prior = vector.new_tensor([8., 20., 1., 7., -2., 0.])[old_intent] - .2 * _take(distances, continuation) + .75
            if role in (1, 2):
                continue_prior = continue_prior + ((old_intent == 2) & class_collector).float() * 18
            all_cells.append(continuation[..., None])
            all_intents.append(old_intent[..., None])
            all_mask.append(continuation_valid[..., None])
            all_prior.append(continue_prior[..., None])
            cells = torch.cat(all_cells, -1)
            intents = torch.cat(all_intents, -1)
            legal = torch.cat(all_mask, -1)
            prior = torch.cat(all_prior, -1)
            # Empty candidate sets have one explicit STOP/current-cell fallback.
            empty = ~legal.any(-1)
            fallback_cell = _where(empty, current, cells[..., -1])
            cells = torch.cat((cells[..., :-1], fallback_cell[..., None]), -1)
            fallback_intent = _where(empty, torch.full_like(intents[..., -1], 6), intents[..., -1])
            intents = torch.cat((intents[..., :-1], fallback_intent[..., None]), -1)
            legal = torch.cat((legal[..., :-1], (legal[..., -1] | empty)[..., None]), -1)
            prior = torch.cat((prior[..., :-1], _where(empty, torch.zeros_like(continue_prior), prior[..., -1])[..., None]), -1)
            cells_by_role.append(cells); intents_by_role.append(intents)
            masks_by_role.append(legal); priors_by_role.append(prior)
        return dict(candidate_cells=torch.stack(cells_by_role, 2),
                    candidate_intents=torch.stack(intents_by_role, 2),
                    candidate_legal=torch.stack(masks_by_role, 2),
                    candidate_prior=torch.stack(priors_by_role, 2))

    def navigate(self, state, own, classes, force_compute=False):
        current = _cell(own[..., :2])
        walk = self.walkability(state, state['role'], classes, current, own[..., 3:].argmax(-1))
        goal = state['goal'].clamp(0, 575)
        changed = (state['nav_goal'] != goal) | (state['nav_walk'] != walk).any(-1)
        if force_compute or bool(changed.any()):
            fresh = self.distance_fields(walk, goal)
            distances = _where(changed[..., None], fresh, state['nav_distance'])
        else:
            distances = state['nav_distance']
        n = len(own)
        flat_current = current.reshape(-1)
        neighbour = self.neighbours[flat_current].reshape(n, 5, 8)
        distance = distances.gather(-1, neighbour)
        edge = self.edges(walk).reshape(n, 5, 576, 8).gather(2, current[..., None, None].expand(-1, -1, 1, 8)).squeeze(2)
        scores = _where(edge, distance + self.costs, torch.full_like(distance, INF))
        best = scores.argmin(-1)
        next_cell = _take(neighbour, best)
        reachable = (scores.amin(-1) < INF / 2) | (current == goal)
        next_cell = _where(current == goal, goal, next_cell)
        old_waypoint = state['waypoint'].clamp(0, 575)
        at_waypoint = (_xy(old_waypoint) - own[..., :2]).square().sum(-1).sqrt() < .012
        invalid_waypoint = ~_take(walk, old_waypoint)
        choose = changed | at_waypoint | (state['waypoint'] < 0) | invalid_waypoint
        waypoint = _where(choose, next_cell, old_waypoint)
        delta = _xy(waypoint) - own[..., :2]
        length = delta.square().sum(-1, keepdim=True).sqrt()
        action = delta / length.clamp_min(1e-8)
        action = _where(((length[..., 0] > .012) & reachable)[..., None], action, torch.zeros_like(action))
        state['nav_goal'], state['nav_walk'], state['nav_distance'] = goal, walk, distances
        state['waypoint'] = waypoint
        return action, state


class Attention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, query, context):
        b, count, width = query.shape
        q = self.q(query).reshape(b, count, self.heads, width // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, -1)
        k = k.reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) * (width // self.heads) ** -.5).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, count, width))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.attn = Attention()
        self.ffnorm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, x):
        q = self.norm(x)
        x = x + self.attn(q, q)
        return x + self.ff(self.ffnorm(x))


class MyPolicy(nn.Module):
    def __init__(self, vector_size=96, n_channels=11):
        super().__init__()
        if (vector_size, n_channels) != (96, 11):
            raise ValueError('v11 requires vector_size=96, n_channels=11')
        self.planner = TensorPlanner()
        self.patch = nn.Linear(176, 64)
        self.xy = nn.Linear(2, 64, bias=False)
        self.entity = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 64))
        self.context = nn.Linear(5, 64)
        self.types = nn.Parameter(torch.randn(3, 64) * .02)
        self.latents = nn.Parameter(torch.randn(32, 64) * .02)
        self.input_norm = nn.LayerNorm(64)
        self.cross = Attention()
        self.blocks = nn.Sequential(Block(), Block())
        self.query = nn.Linear(31, 64)
        self.readout = Attention()
        self.unit_gru = nn.GRUCell(128, 64)
        self.team_gru = nn.GRUCell(64, 64)
        self.feature_norm = nn.LayerNorm(64)
        self.role_head = nn.Linear(64, 3)
        self.candidate_key = nn.Sequential(nn.Linear(13, 64), nn.Tanh(), nn.Linear(64, 64))
        self.goal_query = nn.Linear(64, 64)
        self.gate_head = nn.Linear(64, 2)
        self.correction_head = nn.Linear(64, 8)
        self.keep_head = nn.Linear(64, 1)
        self.direct_head = nn.Linear(64, 9)
        d = math.sqrt(.5)
        self.register_buffer('directions', torch.tensor([[0, 0], [1, 0], [d, d], [0, 1], [-d, d], [-1, 0], [-d, -d], [0, -1], [d, -d]], dtype=torch.float32))
        self.register_buffer('settings', torch.tensor([3, 0, 1, 25, 4, 1], dtype=torch.int64))
        self.register_buffer('patch_xy', self.planner.cell_xy.reshape(24, 24, 2).flip(0).reshape(576, 2))
        for head in (self.role_head, self.goal_query, self.gate_head, self.correction_head, self.keep_head, self.direct_head):
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)
        self.gate_head.bias.data[0] = 4.
        self.keep_head.bias.data[0] = 6.
        self.arm, self.decoder, self.max_interventions = 'F', 'greedy', 1
        self.goal_period, self.residual_period, self.use_memory = 25, 4, True
        self.planner_device = None
        self._runtime_state = None

    def configure(self, arm='F', decoder='sample', max_interventions=1, goal_period=25, residual_period=4, use_memory=True):
        if arm not in ARMS or decoder not in ('sample', 'greedy'):
            raise ValueError('unknown arm or decoder')
        if not 1 <= max_interventions <= 5 or min(goal_period, residual_period) < 1:
            raise ValueError('invalid intervention/decision schedule')
        self.settings.copy_(torch.tensor([ARMS.index(arm), int(decoder == 'sample'), max_interventions,
                                         goal_period, residual_period, int(use_memory)], device=self.settings.device))
        self._read_settings()
        self.reset()
        return self

    def _read_settings(self):
        arm, sample, cap, goal, residual, memory = self.settings.detach().cpu().tolist()
        self.arm, self.decoder = ARMS[arm], ('sample' if sample else 'greedy')
        self.max_interventions, self.goal_period, self.residual_period = cap, goal, residual
        self.use_memory = bool(memory)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._read_settings()
        self.reset()
        return result

    def configure_devices(self, planner_device=None):
        self.planner_device = None if planner_device is None else str(planner_device)
        self.planner.to(next(self.parameters()).device if planner_device is None else planner_device)
        self.reset()
        return self

    def set_devices(self, actor_device, planner_device=None):
        self.to(actor_device)
        return self.configure_devices(planner_device)

    def reset(self):
        self._runtime_state = None

    def initial_state(self, batch=1, device=None):
        device = next(self.parameters()).device if device is None else device
        f = lambda *shape: torch.zeros((batch, *shape), dtype=torch.float32, device=device)
        i = lambda *shape: torch.zeros((batch, *shape), dtype=torch.long, device=device)
        b = lambda *shape: torch.zeros((batch, *shape), dtype=torch.bool, device=device)
        state = dict(unit_hidden=f(5, 64), team_hidden=f(64), goal=i(5)-1, intent=i(5),
                     role=torch.tensor([0, 0, 0, 1, 1], device=device)[None].expand(batch, -1).clone(),
                     goal_age=i(5)+self.goal_period, residual_remaining=i(5), correction=i(5)-1,
                     direct=i(5), previous_action=f(5, 2), previous_position=f(5, 2),
                     previous_holding=i(5), previous_class=i(5), previous_time=f()-1,
                     side=i()-1, step_count=i(), stuck_count=i(5), waypoint=i(5)-1,
                     blocked_cell=i(5)-1, blocked_ttl=i(5), failed_store=i(5)-1, failed_storage=b(5,576), failed_ttl=i(5),
                     delivery_wait=i(5), walls=b(576), ally_storage=b(576), enemy_storage=b(576),
                     enemy_base=b(576), item_memory=b(576), item_age=i(576),
                     nav_goal=i(5)-1, nav_walk=b(5, 576), nav_distance=f(5, 576)+INF)
        return {k: state[k] for k in STATE_KEYS}

    def _prepare(self, vector, graphic, state, force_compute=False):
        n = len(vector)
        current_time = vector[:, 0, 95]
        side = (vector[:, 0, 2] < 0).long()
        reset = (state['side'] != side) | (current_time > state['previous_time'] + 1e-4) | (state['step_count'] == 0)
        initial = self.initial_state(n, vector.device)
        state = {k: _where(reset.reshape(n, *([1] * (v.ndim - 1))), initial[k], v) for k, v in state.items()}
        own = own_entities(vector)
        classes = vector[..., 90:93].argmax(-1)
        holding = own[..., 3:].argmax(-1)
        displacement = (own[..., :2] - state['previous_position']).square().sum(-1).sqrt()
        respawn = (~reset[:, None]) & ((displacement > .15) | ((classes == 0) & (state['previous_class'] != 0)))
        unit_reset = reset[:, None] | respawn
        semantic = self.planner.decode(graphic)
        state['walls'] = state['walls'] | semantic[:, 1]
        state['ally_storage'] = state['ally_storage'] | semantic[:, 2]
        state['enemy_storage'] = state['enemy_storage'] | semantic[:, 3]
        # Protected bases are the border-corner storage components. Derive their
        # orientation from the initial enemy spawn, never from hidden game state.
        entities = vector[:, 0, :90].reshape(n, 10, 9)
        enemy_mask = entities[..., 2] < 0
        enemy_position = (entities[..., :2] * enemy_mask[..., None]).sum(1) / enemy_mask.sum(1, keepdim=True).clamp_min(1)
        enemy_corner = (enemy_position > .5).float()
        corner_region = (self.planner.cell_xy[None] - enemy_corner[:, None]).abs().amax(-1) < (3. / 24.)
        state['enemy_base'] = _where(reset[:, None], corner_region, state['enemy_base'])
        unseen = ~semantic[:, 6]
        state['item_age'] = _where(unseen, state['item_age'] + 1, torch.zeros_like(state['item_age']))
        state['item_memory'] = semantic[:, 6] | (state['item_memory'] & (state['item_age'] < 12))
        commanded = state['previous_action'].square().sum(-1) > .01
        stationary = (displacement < 1e-4) & commanded & ~unit_reset
        state['stuck_count'] = _where(stationary, state['stuck_count'] + 1, torch.zeros_like(state['stuck_count']))
        stuck = state['stuck_count'] >= 30
        state['blocked_ttl'] = (state['blocked_ttl'] - 1).clamp_min(0)
        state['blocked_cell'] = _where(stuck, state['waypoint'], state['blocked_cell'])
        state['blocked_ttl'] = _where(stuck, torch.full_like(state['blocked_ttl'], 30), state['blocked_ttl'])
        near_goal = (_xy(state['goal'].clamp(0, 575)) - own[..., :2]).square().sum(-1).sqrt() < .024
        delivery_wait = near_goal & (state['intent'] == 1) & (holding > 0)
        state['delivery_wait'] = _where(delivery_wait, state['delivery_wait'] + 1, torch.zeros_like(state['delivery_wait']))
        failed = state['delivery_wait'] >= 25
        state['failed_ttl'] = (state['failed_ttl'] - 1).clamp_min(0)
        state['failed_store'] = _where(failed, state['goal'], state['failed_store'])
        state['failed_ttl'] = _where(failed, torch.full_like(state['failed_ttl'], 200), state['failed_ttl'])
        holding_change = holding != state['previous_holding']
        clear_failed = holding_change | (state['failed_ttl'] <= 0)
        state['failed_storage'] = state['failed_storage'] & ~clear_failed[..., None]
        if force_compute or bool(failed.any()):
            storage_walk = state['ally_storage'][:,None].expand(-1,5,-1)
            connected = self.planner.distance_fields(storage_walk, state['goal'].clamp(0,575)) < INF / 2
            state['failed_storage'] = state['failed_storage'] | (connected & failed[...,None])
        class_change = classes != state['previous_class']
        vanished = ((state['intent'] == 0) & (state['goal'] >= 0) & (holding == 0) & ~_take(state['item_memory'][:, None].expand(-1, 5, -1), state['goal'].clamp(0, 575)))
        arrived = near_goal & ((state['intent'] == 3) | (state['intent'] == 4))
        interrupt = holding_change | class_change | vanished | stuck | failed | respawn | arrived
        goal_event = (state['goal_age'] >= self.goal_period) | (state['goal'] < 0) | interrupt | reset[:, None]
        residual_event = (state['residual_remaining'] <= 0) | interrupt | reset[:, None]
        event = goal_event.any(-1) | residual_event.any(-1)
        state['unit_hidden'] = _where(unit_reset[..., None], torch.zeros_like(state['unit_hidden']), state['unit_hidden'])
        state['role'] = _where(classes == 1, torch.ones_like(classes), state['role'])
        state['role'] = _where(classes == 2, torch.full_like(classes, 2), state['role'])
        state['goal'] = _where(respawn, torch.full_like(state['goal'], -1), state['goal'])
        state['correction'] = _where(residual_event, torch.full_like(state['correction'], -1), state['correction'])
        if force_compute or bool(event.any()):
            candidates = self.planner.candidates(vector, semantic, state, own, classes)
        else:
            candidates = {}
        role_legal = torch.stack(((classes == 0), (classes != 2) & (holding == 0), (classes != 1) & (holding == 0)), -1)
        role_legal = role_legal | F.one_hot(state['role'], 3).bool()
        decision = dict(unit_hidden=state['unit_hidden'], team_hidden=state['team_hidden'],
                        unit_reset=unit_reset, team_reset=reset, previous_goal=state['goal'],
                        previous_action=state['previous_action'], role=state['role'], intent=state['intent'],
                        goal_event=goal_event, residual_event=residual_event, role_legal=role_legal,
                        correction_before=state['correction'], direct_before=state['direct'], side_swap=(side == 1),
                        **candidates)
        return state, decision, event, own, classes, holding, side

    def scene(self, vector, graphic):
        n = len(vector)
        patches = graphic.reshape(n, 11, 24, 4, 24, 4).permute(0, 2, 4, 1, 3, 5).reshape(n, 576, 176)
        maps = self.patch(patches) + self.xy(self.patch_xy) + self.types[0]
        entities = self.entity(vector[:, 0, :90].reshape(n, 10, 9)) + self.types[1]
        elapsed = 420 * (1 - vector[:, 0, 95])
        phase = elapsed.remainder(20) / 20
        context = torch.cat((vector[:, 0, 93:96], torch.sin(phase[:, None] * 2 * math.pi), torch.cos(phase[:, None] * 2 * math.pi)), -1)
        token = self.context(context)[:, None] + self.types[2]
        tokens = self.input_norm(torch.cat((maps, entities, token), 1))
        latent = self.latents[None].expand(n, -1, -1)
        return self.blocks(latent + self.cross(latent, tokens)), context

    @staticmethod
    def _choose(logits, legal, active, forced, deterministic):
        legal = legal.bool()
        # A single legal action is deterministic with exactly zero log-prob.
        logs = logits.masked_fill(~legal, -1e9).log_softmax(-1)
        if forced is None:
            choice = logs.argmax(-1) if deterministic else torch.multinomial(logs.exp().reshape(-1, logs.shape[-1]), 1).reshape(logs.shape[:-1])
        else:
            choice = forced.long()
        log_prob = _take(logs, choice) * active.float()
        entropy = -(logs.exp() * logs).sum(-1) * active.float()
        return choice, log_prob, entropy

    def evaluate_decision(self, vector, graphic, state, choices=None, deterministic=False, canonicalized=False):
        """Differentiable event replay; planner/FSM are not re-executed.

        Replace unit_hidden/team_hidden in the recorded decision_state with the
        current recurrent prefix, retaining unit_reset/team_reset masks.
        """
        n = len(vector)
        if not canonicalized:
            vector, graphic, _ = canonical_observation(vector, graphic)
        own = own_entities(vector)
        latent, context = self.scene(vector, graphic)
        unit_before = _where(state['unit_reset'][..., None], torch.zeros_like(state['unit_hidden']), state['unit_hidden'])
        team_before = _where(state['team_reset'][..., None], torch.zeros_like(state['team_hidden']), state['team_hidden'])
        if not self.use_memory:
            unit_before, team_before = torch.zeros_like(unit_before), torch.zeros_like(team_before)
        query_input = torch.cat((own, vector[..., 90:93], _xy(state['previous_goal'].clamp(0, 575)),
                                 state['previous_action'], F.one_hot(state['intent'].clamp(0, 6), 7).float(),
                                 F.one_hot(state['role'], 3).float(), context[:, None].expand(-1, 5, -1)), -1)
        query = self.query(query_input)
        attended = query + self.readout(query, latent)
        unit = self.unit_gru(torch.cat((attended, team_before[:, None].expand(-1, 5, -1)), -1).reshape(-1, 128), unit_before.reshape(-1, 64)).reshape(n, 5, 64)
        team = self.team_gru(unit.mean(1), team_before)
        event = state['goal_event'].any(-1) | state['residual_event'].any(-1)
        unit = _where(event[:, None, None], unit, unit_before)
        team = _where(event[:, None], team, team_before)
        features = self.feature_norm(unit + team[:, None])
        forced = {} if choices is None else choices
        zero = vector.new_zeros(n)
        log_prob, entropy = zero, zero
        chosen_roles, chosen_goals, chosen_intents = [], [], []
        role_choices, goal_choices = [], []
        # Initial reservations for held goals count against active assignments.
        reserved = []
        role_logits = self.role_head(features) + F.one_hot(state['role'], 3).float() * 4.
        goal_active = state['goal_event'] if self.arm in ('G', 'F') else torch.zeros_like(state['goal_event'])
        for slot in range(5):
            active = goal_active[:, slot]
            role_mask = state['role_legal'][:, slot].clone()
            other_carrier = ((state['role'] == 2) & (torch.arange(5, device=vector.device)[None] != slot)).any(-1)
            if chosen_roles:
                other_carrier = other_carrier | (torch.stack(chosen_roles, 1) == 2).any(-1)
            role_mask[:, 2] = role_mask[:, 2] & (~other_carrier | (state['role'][:, slot] == 2))
            role_mask = _where(active[:, None], role_mask, F.one_hot(state['role'][:, slot], 3).bool())
            role, lp, en = self._choose(role_logits[:, slot], role_mask, active,
                                        None if choices is None else forced['role'][:, slot], deterministic)
            log_prob, entropy = log_prob + lp, entropy + en
            chosen_roles.append(role); role_choices.append(role)
            gather = role[:, None, None].expand(-1, 1, CANDIDATES)
            cells = state['candidate_cells'][:, slot].gather(1, gather).squeeze(1)
            intents = state['candidate_intents'][:, slot].gather(1, gather).squeeze(1)
            legal = state['candidate_legal'][:, slot].gather(1, gather).squeeze(1)
            prior = state['candidate_prior'][:, slot].gather(1, gather).squeeze(1)
            target_xy = _xy(cells)
            distance = (target_xy - own[:, slot, None, :2]).square().sum(-1).sqrt()
            candidate_features = torch.cat((target_xy, F.one_hot(intents.clamp(0, 6), 7).float(), distance[..., None],
                                            F.one_hot(role, 3).float()[:, None].expand(-1, CANDIDATES, -1)), -1)
            logits = (self.candidate_key(candidate_features) * self.goal_query(features[:, slot])[:, None]).sum(-1) / 8 + prior
            # Capacity-one collection/special goals; two delivery slots per tile.
            counts = torch.zeros_like(cells)
            for before in range(slot):
                counts = counts + (cells == chosen_goals[before][:, None]).long()
            for later in range(slot + 1, 5):
                held = ~state['goal_event'][:, later]
                counts = counts + ((cells == state['previous_goal'][:, later, None]) & held[:, None]).long()
            capacity = _where((intents == 0) | (intents == 2), torch.ones_like(cells), torch.full_like(cells, 2))
            legal = legal & (counts < capacity)
            # Reservation exhaustion produces an explicit current-cell fallback;
            # it must never revive an invalid old goal or a reserved item.
            exhausted = ~legal.any(-1)
            cells = torch.cat((cells[:,:-1], _where(exhausted, _cell(own[:,slot,:2]), cells[:,-1])[:,None]), -1)
            intents = torch.cat((intents[:,:-1], _where(exhausted, torch.full_like(intents[:,-1],6), intents[:,-1])[:,None]), -1)
            legal = torch.cat((legal[:, :-1], (legal[:, -1] | exhausted)[:, None]), -1)
            reconsider = state['goal_event'][:, slot]
            held_index = torch.full((n,), CANDIDATES - 1, dtype=torch.long, device=vector.device)
            legal = _where(reconsider[:, None], legal, F.one_hot(held_index, CANDIDATES).bool())
            goal, lp, en = self._choose(logits if self.arm in ('G', 'F') else prior, legal,
                                       active, None if choices is None else forced['goal'][:, slot],
                                       deterministic or self.arm not in ('G', 'F'))
            log_prob, entropy = log_prob + lp, entropy + en
            cell = _take(cells, goal)
            intent = _take(intents, goal)
            chosen_goals.append(_where(reconsider, cell, state['previous_goal'][:, slot]))
            chosen_intents.append(_where(reconsider, intent, state['intent'][:, slot]))
            goal_choices.append(goal)
        correction = state['correction_before']
        residual = torch.zeros(n, dtype=torch.long, device=vector.device)
        gates = torch.zeros((n, 5), dtype=torch.long, device=vector.device)
        corrections = correction.clamp_min(0)
        direct = state['direct_before']
        res_active = state['residual_event']
        if self.arm == 'R':
            active = res_active.any(-1)
            logits = torch.cat((self.keep_head(team), self.correction_head(features).reshape(n, 40)), -1)
            legal = torch.cat((torch.ones((n, 1), dtype=torch.bool, device=vector.device), res_active[..., None].expand(-1, -1, 8).reshape(n, 40)), -1)
            residual, lp, en = self._choose(logits, legal, active, forced.get('residual'), deterministic)
            log_prob, entropy = log_prob + lp, entropy + en
            slot_index, corr_index = ((residual - 1).clamp_min(0) // 8), ((residual - 1).clamp_min(0) % 8)
            slots = torch.arange(5, device=vector.device)[None]
            selected = (slots == slot_index[:, None]) & (residual[:, None] > 0) & active[:, None]
            correction = _where(active[:, None], torch.full_like(correction, -1), correction)
            correction = _where(selected, corr_index[:, None], correction)
            gates, corrections = selected.long(), correction.clamp_min(0)
        elif self.arm == 'F':
            used = ((correction >= 0) & ~res_active).sum(-1)
            gate_rows, correction_rows, executed = [], [], []
            for slot in range(5):
                active = res_active[:, slot]
                legal = torch.stack((torch.ones_like(active), used < self.max_interventions), -1)
                legal = _where(active[:, None], legal, F.one_hot((correction[:, slot] >= 0).long(), 2).bool())
                gate, lp, en = self._choose(self.gate_head(features[:, slot]), legal, active,
                                           None if choices is None else forced['gate'][:, slot], deterministic)
                log_prob, entropy = log_prob + lp, entropy + en
                corr_active = active & (gate == 1)
                corr_legal = torch.ones((n, 8), dtype=torch.bool, device=vector.device)
                corr, lp, en = self._choose(self.correction_head(features[:, slot]), corr_legal, corr_active,
                                           None if choices is None else forced['correction'][:, slot], deterministic)
                log_prob, entropy = log_prob + lp, entropy + en
                choice = _where(active, _where(gate == 1, corr, torch.full_like(corr, -1)), correction[:, slot])
                used = used + (active & (gate == 1)).long()
                gate_rows.append(gate); correction_rows.append(corr); executed.append(choice)
            gates, corrections, correction = torch.stack(gate_rows, 1), torch.stack(correction_rows, 1), torch.stack(executed, 1)
        elif self.arm == 'D':
            logits = self.direct_head(features)
            legal = torch.ones_like(logits, dtype=torch.bool)
            chosen, lp, en = self._choose(logits, legal, res_active, forced.get('direct'), deterministic)
            direct = _where(res_active, chosen, direct)
            log_prob, entropy = log_prob + lp.sum(-1), entropy + en.sum(-1)
        else:
            correction = torch.full_like(correction, -1)
        result_choices = dict(role=torch.stack(role_choices, 1), goal=torch.stack(goal_choices, 1),
                              gate=gates, correction=corrections, residual=residual, direct=direct)
        return dict(log_prob=log_prob, entropy=entropy, choices=result_choices, unit_hidden=unit,
                    team_hidden=team, selected_goal=torch.stack(chosen_goals, 1),
                    selected_intent=torch.stack(chosen_intents, 1), selected_role=torch.stack(chosen_roles, 1),
                    selected_correction=correction, selected_direct=direct, features=features)

    def _correct(self, base, correction):
        angles = base.new_tensor([45., -45., 90., -90., 135., -135., 180., 0.]) * (math.pi / 180.)
        angle = angles[correction.clamp_min(0)]
        c, s = angle.cos(), angle.sin()
        rotated = torch.stack((base[..., 0] * c - base[..., 1] * s, base[..., 0] * s + base[..., 1] * c), -1)
        rotated = _where((correction == 7)[..., None], torch.zeros_like(rotated), rotated)
        absolute = self.directions[correction.clamp_min(0) + 1]
        rotated = _where((base.square().sum(-1) < 1e-8)[..., None], absolute, rotated)
        return _where((correction >= 0)[..., None], rotated, base).clamp(-1, 1)

    def step(self, vector, graphic, state, deterministic=False, choices=None, force_compute=False):
        if vector.ndim != 3 or vector.shape[1:] != (5, 96) or graphic.shape != (len(vector), 11, 96, 96):
            raise ValueError('step requires [N,5,96] and verified shared [N,11,96,96]')
        actor_device = next(self.parameters()).device
        planner_device = actor_device if self.planner_device is None else torch.device(self.planner_device)
        canonical_vector, canonical_graphic, side_swap = canonical_observation(vector, graphic)
        pv, pg = canonical_vector.to(planner_device), canonical_graphic.to(planner_device)
        state = move_state(state, planner_device)
        state, decision, event, own, classes, holding, side = self._prepare(pv, pg, state, force_compute)
        decision_actor = move_state(decision, actor_device)
        result = None
        if force_compute or bool(event.any()):
            result = self.evaluate_decision(canonical_vector.to(actor_device), canonical_graphic.to(actor_device), decision_actor,
                                            None if choices is None else move_state(choices, actor_device), deterministic, canonicalized=True)
            for key, value in [('unit_hidden', 'unit_hidden'), ('team_hidden', 'team_hidden'),
                               ('goal', 'selected_goal'), ('intent', 'selected_intent'), ('role', 'selected_role'),
                               ('correction', 'selected_correction'), ('direct', 'selected_direct')]:
                state[key] = result[value].to(planner_device)
        base, state = self.planner.navigate(state, own, classes, force_compute)
        base = base.to(actor_device)
        action = self.directions[state['direct'].to(actor_device)] if self.arm == 'D' else self._correct(base, state['correction'].to(actor_device))
        state['goal_age'] = _where(decision['goal_event'], torch.zeros_like(state['goal_age']), state['goal_age']) + 1
        state['residual_remaining'] = _where(decision['residual_event'], torch.full_like(state['residual_remaining'], self.residual_period), state['residual_remaining']) - 1
        state['previous_action'] = action.to(planner_device)
        state['previous_position'] = own[..., :2]
        state['previous_holding'], state['previous_class'] = holding, classes
        state['previous_time'], state['side'] = pv[:, 0, 95], side
        state['step_count'] = state['step_count'] + 1
        state['stuck_count'] = _where(state['stuck_count'] >= 30, torch.zeros_like(state['stuck_count']), state['stuck_count'])
        zeros = vector.new_zeros(len(vector)).to(actor_device)
        action = _where(side_swap.to(actor_device)[:, None, None], action.flip(-1), action)
        base = _where(side_swap.to(actor_device)[:, None, None], base.flip(-1), base)
        return dict(action=action.float(), state=state, event=event.to(actor_device),
                    log_prob=zeros if result is None else result['log_prob'],
                    entropy=zeros if result is None else result['entropy'],
                    choices={} if result is None else result['choices'], decision_state=decision_actor,
                    base_action=base, features=None if result is None else result['features'])

    def forward(self, vector, graphic):
        if vector.ndim != 2 or vector.shape[1] != 96 or graphic.shape != (len(vector), 11, 96, 96):
            raise ValueError('expected float32[B,96], float32[B,11,96,96]')
        if vector.dtype != torch.float32 or graphic.dtype != torch.float32:
            raise TypeError('float32 observations required')
        if len(vector) == 0:
            return vector.new_empty((0, 2))
        if len(vector) != 5:
            raise ValueError('v11 team controller requires the supplied runner canonical five-row team batch')
        if not torch.equal(graphic, graphic[:1].expand_as(graphic)):
            raise ValueError('team graphics differ; common-scene compression is unsafe')
        if not torch.equal(vector[:, :90], vector[:1, :90].expand(5, -1)):
            raise ValueError('rows do not share one team scene')
        if self._runtime_state is None:
            self._runtime_state = self.initial_state(1, vector.device)
        result = self.step(vector[None], graphic[:1], self._runtime_state, deterministic=self.decoder == 'greedy')
        self._runtime_state = detach_state(result['state'])
        return result['action'][0].to(vector.device)


class ExportCore(nn.Module):
    """Development ONNX boundary with explicit state; not an extra public I/O."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, vector, graphic, *state_values):
        state = dict(zip(STATE_KEYS, state_values))
        result = self.policy.step(vector, graphic, state, deterministic=True, force_compute=True)
        return (result['action'], *(result['state'][key] for key in STATE_KEYS))


def fixture(batch=1, device='cpu'):
    """Small meaningful scene for checks and device measurements, not game data."""
    vector = torch.zeros((batch, 5, 96), dtype=torch.float32, device=device)
    entities = vector[:, :, :90].reshape(batch, 5, 10, 9)
    position = torch.tensor([[.08, .9]] * 5 + [[.9, .08]] * 5, device=device)
    entities[..., :2] = position
    entities[..., 2] = torch.tensor([1.] * 5 + [-1.] * 5, device=device)
    entities[..., 3] = 1
    vector[..., 90] = 1
    vector[..., 95] = 1
    graphic = torch.zeros((batch, 11, 96, 96), dtype=torch.float32, device=device)
    graphic[:, 0] = 1
    def tile(channel, x, y):
        graphic[:, 0, (23-y)*4:(24-y)*4, x*4:(x+1)*4] = 0
        graphic[:, channel, (23-y)*4:(24-y)*4, x*4:(x+1)*4] = 1
    for x, y in [(4, 20), (6, 16), (16, 6)]: tile(2, x, y)
    for x, y in [(20, 4), (16, 6)]: tile(3, x, y)
    for x, y in [(7, 17), (9, 15), (14, 17), (8, 10), (16, 12)]: tile(6, x, y)
    for y in range(3, 16):
        if y != 10: tile(1, 5, y)
    return vector, graphic
