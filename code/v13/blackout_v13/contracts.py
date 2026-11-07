"""Fixed public and recurrent contracts; semantic IDs are independent of shortlist order."""
SCHEMA = 'blackout.v13.tensor.1'
HISTORY_SIZE = 64
TOKEN_SIZE = 128
TASK_SIZE = 24
KINDS = ('collect-trip', 'deliver', 'special-trip', 'intercept', 'defend',
         'raid-trip', 'transform-Hunter', 'transform-Carrier', 'hold', 'evade')
LAGS = (0, 4, 8, 16, 32, 64)
# task: source,destination,kind,phase,age,last_event,time,holding,class,x,y,
# valid,version,action_x,action_y,step,goal,speed,segment,distance,eta,enemy_id,side,stalled_seconds
STATE_SHAPES = ((5, 64, 128), (5, 64), (5, 24), (5, 2, 576))


def initial_state(device='cpu'):
    import torch
    return tuple(torch.zeros(s, device=device) for s in STATE_SHAPES)


def validate_observation(vector, graphic):
    import torch
    if vector.shape != (5, 96) or graphic.shape != (5, 11, 96, 96):
        raise ValueError('expected canonical team vector[5,96], graphic[5,11,96,96]')
    if vector.dtype != torch.float32 or graphic.dtype != torch.float32:
        raise TypeError('public inputs must be float32')
    if not torch.isfinite(vector).all() or not torch.isfinite(graphic).all():
        raise ValueError('nonfinite observation')
