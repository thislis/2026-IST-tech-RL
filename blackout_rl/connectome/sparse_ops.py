"""W[dst,src] propagation without an N×N trainable tensor."""
import torch


def propagate(src, dst, weights, state, n):
    # Sparse mm avoids a batch×edge×channel intermediate on the full graph.
    matrix = torch.sparse_coo_tensor(torch.stack((dst, src)), weights, (n, n), check_invariants=True)
    shape = state.shape
    flat = state.reshape(-1, n, shape[-1]).permute(1, 0, 2).reshape(n, -1)
    out = torch.sparse.mm(matrix, flat)
    return out.reshape(n, -1, shape[-1]).permute(1, 0, 2).reshape(shape)
