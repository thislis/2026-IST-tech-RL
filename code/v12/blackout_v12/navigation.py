"""Exact eight-neighbour Dijkstra fields and bounded CPU LRU path cache.

    Diagonal corner cutting is forbidden. Dijkstra has the same shortest-path
    result as positive-cost min-plus relaxation, without 575 full-grid sweeps.
"""
from collections import OrderedDict, deque
import heapq
import math
import numpy as np

INF = 1e6
OFFSETS = ((1, 0, 1.), (0, 1, 1.), (-1, 0, 1.), (0, -1, 1.),
           (1, 1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
           (-1, -1, math.sqrt(2)), (1, -1, math.sqrt(2)))
HUNTER_SHRINE = (11*24+11, 11*24+12, 12*24+11, 12*24+12)
CARRIER_SHRINE = 20*24+1


def cells(position):
    xy = np.clip(np.asarray(position)*24, 0, 23).astype(np.int64)
    return xy[..., 1]*24+xy[..., 0]


def center(cell):
    return np.array([cell % 24 + .5, cell // 24 + .5], dtype=np.float32)/24


def components(mask):
    remaining = set(np.flatnonzero(np.asarray(mask).reshape(-1)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, found = deque([start]), [start]
        while queue:
            cell = queue.popleft()
            x, y = cell % 24, cell // 24
            for dx, dy, _ in OFFSETS[:4]:
                nx, ny = x+dx, y+dy
                other = ny*24+nx
                if 0 <= nx < 24 and 0 <= ny < 24 and other in remaining:
                    remaining.remove(other)
                    queue.append(other)
                    found.append(other)
        groups.append(tuple(sorted(found)))
    return tuple(groups)


def neighbours(cell, walk):
    x, y = cell % 24, cell // 24
    for dx, dy, cost in OFFSETS:
        nx, ny = x+dx, y+dy
        if not (0 <= nx < 24 and 0 <= ny < 24 and walk[ny, nx]):
            continue
        if dx and dy and not (walk[y, nx] and walk[ny, x]):
            continue
        yield ny*24+nx, cost


def distance_field(walk, sources):
    """Return shortest distance and deterministic source owner for each cell."""
    walk = np.asarray(walk, dtype=bool).reshape(24, 24)
    distance = np.full(576, INF, np.float32)
    owner = np.full(576, -1, np.int16)
    queue = []
    for source in sorted(set(int(s) for s in sources)):
        if 0 <= source < 576 and walk.reshape(-1)[source]:
            distance[source], owner[source] = 0., source
            heapq.heappush(queue, (0., source, source))
    while queue:
        cost, root, cell = heapq.heappop(queue)
        if cost > float(distance[cell])+1e-5:
            continue
        for other, edge in neighbours(cell, walk):
            proposed = cost+edge
            if proposed < float(distance[other])-1e-5 or (abs(proposed-float(distance[other])) <= 1e-5 and root < owner[other]):
                distance[other], owner[other] = proposed, root
                heapq.heappush(queue, (proposed, root, other))
    return distance, owner


class Navigation:
    def __init__(self, cache_fields=32):
        self.limit = int(cache_fields)
        if self.limit<1:
            raise ValueError('cache_fields must be positive')
        self.cache = OrderedDict()
        self.solves = self.hits = 0

    def reset(self):
        self.cache.clear()
        self.solves = self.hits = 0

    def field(self, walk, sources, profile=()):
        # Complete walk mask includes map changes, shrine intent and local
        # blocks. Profile includes observed class/size-item and cost version.
        key = (tuple(profile), np.packbits(walk).tobytes(), tuple(sorted(set(map(int, sources)))))
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        result = distance_field(walk, key[-1])
        self.solves += 1
        self.cache[key] = result
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return result

    @property
    def state_bytes(self):
        return sum(a.nbytes+b.nbytes+len(k[1]) for k, (a,b) in self.cache.items())

    def walkability(self, state, slot, class_id, holding, target_kind=None, allow_loaded_transform=False):
        walk = ~(state.walls | state.enemy_base)
        walk = walk.copy()
        if class_id == 0:
            if target_kind != 'transform-Hunter' or (holding and not allow_loaded_transform):
                walk.reshape(-1)[list(HUNTER_SHRINE)] = False
            if target_kind != 'transform-Carrier' or (holding and not allow_loaded_transform):
                walk.reshape(-1)[CARRIER_SHRINE] = False
        for cell, ttl in state.units[slot].blocked.items():
            if ttl > 0:
                walk.reshape(-1)[cell] = False
        current = int(cells(state.positions[slot]))
        if not state.walls.reshape(-1)[current]:
            walk.reshape(-1)[current] = True
        return walk

    def action(self, position, goal, walk, profile=()):
        if goal < 0:
            return np.zeros(2, np.float32), -1
        current = int(cells(position))
        field, _ = self.field(walk, [goal], profile)
        if field[current] >= INF/2:
            return np.zeros(2, np.float32), -1
        if current == goal:
            waypoint = goal
        else:
            options = [(float(field[n])+cost, n) for n, cost in neighbours(current, walk)]
            waypoint = min(options, default=(INF, -1))[1]
        if waypoint < 0:
            return np.zeros(2, np.float32), -1
        delta = center(waypoint)-position
        norm = float(np.linalg.norm(delta))
        if norm < .006:
            return np.zeros(2, np.float32), waypoint
        return (delta/max(norm, 1e-9)).astype(np.float32), waypoint

    def evade(self, position, enemies, walk):
        """Short fixed safety action; never follows a released task."""
        current = int(cells(position))
        if not len(enemies):
            return np.zeros(2, np.float32), current
        # Staying is preferable to a step that moves closer to the threat.
        options = [(float(np.linalg.norm(position-enemies,axis=-1).min()),current)]
        options += [(float(np.linalg.norm(center(n)-enemies, axis=-1).min()), n)
                   for n, _ in neighbours(current, walk)]
        best = max(options, default=(0., current))[1]
        if best==current:
            return np.zeros(2,np.float32),current
        delta = center(best)-position
        norm = float(np.linalg.norm(delta))
        return (delta/max(norm, 1e-9)).astype(np.float32), best
