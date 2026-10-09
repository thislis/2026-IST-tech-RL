from .contracts import ContractError

DECODERS = ('joint_argmax_v1', 'gate_then_conditional_argmax_v1', 'sampled_v1')

def decode(distribution, decoder, threshold=.5, generator=None):
    if decoder not in DECODERS or not 0 <= threshold <= 1:
        raise ContractError('unknown decoder or invalid threshold')
    if decoder == 'sampled_v1':
        return distribution.sample(generator)
    if decoder == 'joint_argmax_v1':
        return distribution.probs.argmax(-1)  # stable ID, KEEP wins ties
    correction = distribution.probs[..., 1:].argmax(-1) + 1
    return correction * (distribution.q > threshold).long()
