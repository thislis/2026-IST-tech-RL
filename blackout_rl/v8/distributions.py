"""One normalized joint distribution over KEEP plus forty stable corrections."""
import torch
from torch.nn import functional as F
from .contracts import ContractError

class JointPolicy:
    def __init__(self, log_probs):
        self.log_probs = log_probs
        self.probs = log_probs.exp()
        self.q = self.probs[..., 1:].sum(-1)

    def log_prob(self, action):
        return self.log_probs.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self):
        safe = self.log_probs.masked_fill(~torch.isfinite(self.log_probs), 0.)
        return -(self.probs * safe).sum(-1)

    def sample(self, generator=None):
        shape = self.probs.shape[:-1]
        return torch.multinomial(self.probs.reshape(-1, 41), 1, generator=generator).reshape(shape)

    @property
    def conditional(self):
        return self.probs[..., 1:] / self.q.unsqueeze(-1).clamp_min(torch.finfo(self.probs.dtype).tiny)


def _validate(corrections, valid):
    if corrections.shape != valid.shape or corrections.shape[-1] != 40 or valid.dtype != torch.bool:
        raise ContractError('40 correction logits and boolean mask required')
    if not bool(torch.isfinite(corrections).all()):
        raise ContractError('nonfinite correction logits')


def gated(gate, corrections, valid):
    _validate(corrections, valid)
    if gate.shape != corrections.shape[:-1] or not bool(torch.isfinite(gate).all()):
        raise ContractError('gate shape/nonfinite')
    has = valid.any(-1)
    # Avoid empty softmax (including its backward), then remove dummy mass.
    safe = valid.clone()
    safe[..., 0] |= ~has
    conditional = corrections.masked_fill(~safe, -torch.inf).log_softmax(-1)
    keep = torch.where(has, F.logsigmoid(-gate), torch.zeros_like(gate))
    correction = (F.logsigmoid(gate).unsqueeze(-1) + conditional).masked_fill(~valid, -torch.inf)
    return JointPolicy(torch.cat((keep.unsqueeze(-1), correction), -1))


def flat(keep, corrections, valid):
    _validate(corrections, valid)
    if keep.shape != corrections.shape[:-1] or not bool(torch.isfinite(keep).all()):
        raise ContractError('KEEP shape/nonfinite')
    logits = torch.cat((keep.unsqueeze(-1), corrections.masked_fill(~valid, -torch.inf)), -1)
    return JointPolicy(logits.log_softmax(-1))


def exact_factorization(logits, valid):
    """Legacy masked flat41 as gate+conditional. No decoder change is implied."""
    if logits.shape[-1] != 41:
        raise ContractError('legacy logits must have 41 entries')
    _validate(logits[..., 1:], valid)
    has = valid.any(-1)
    safe = valid.clone(); safe[..., 0] |= ~has
    log_mass = logits[..., 1:].masked_fill(~safe, -torch.inf).logsumexp(-1)
    gate = log_mass - logits[..., 0]
    return gated(gate, logits[..., 1:], valid)
