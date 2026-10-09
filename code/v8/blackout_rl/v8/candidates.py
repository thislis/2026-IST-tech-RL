import numpy as np
import torch
from blackout_rl.scripted_fsm import HUNTER_SHRINE_CELLS, carrier_shrine_cell
from .contracts import ContractError

STABLE_IDS = tuple(['KEEP'] + [f'slot{s}:correction{d}' for s in range(5) for d in range(8)])

def build_mask(observations, team, planner, alternatives):
    reasons = [[] for _ in range(40)]
    for slot in range(5):
        obs = observations[f'unit_{team*5+slot}']
        own = obs['vector'][:90].reshape(10,9)[team*5+slot,:2]
        walls = obs['graphic'][:,:,1]
        forbidden = ({(c.x,c.y) for c in HUNTER_SHRINE_CELLS} if slot < 3 else set()) | {(carrier_shrine_cell(team).x,carrier_shrine_cell(team).y)}
        for column, direction in enumerate(alternatives[slot]):
            reason = reasons[slot*8+column]
            if np.allclose(direction, planner[slot], atol=1e-5):
                reason.append('duplicate_candidate:planner')
            endpoint = own + direction * (9 * .02 / 24)
            for fraction in (.5, 1.):
                point = own + fraction*(endpoint-own)
                if tuple((point*24).astype(int)) in forbidden:
                    reason.append('heuristic_preference:role_shrine')
                for offset in ((0,0),(.014,0),(-.014,0),(0,.014),(0,-.014)):
                    x,y = point + offset
                    if not (0<=x<1 and 0<=y<1):
                        reason.append('geometry_estimate:boundary_margin')
                    elif walls[min(int((1-y)*len(walls)),len(walls)-1),int(x*walls.shape[1])] > .5:
                        reason.append('geometry_estimate:wall_margin')
            reasons[slot*8+column] = sorted(set(reason))
    return torch.tensor([not r for r in reasons],dtype=torch.bool), reasons

def execute(index, planner, alternatives, valid):
    if not 0 <= index <= 40 or index and not bool(valid[index-1]):
        raise ContractError('invalid proposal, cannot silently overwrite action')
    result = planner.copy()
    if index:
        slot, direction = divmod(index-1,8)
        result[slot] = alternatives[slot,direction]
    if result.shape != (5,2) or not np.isfinite(result).all() or np.abs(result).max()>1.000001:
        raise ContractError('invalid actuator output')
    return result, 'identity_executor_v1'
