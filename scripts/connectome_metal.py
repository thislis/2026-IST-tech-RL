"""MPS CSR matvec with retained edge order and CPU-compatible brain state/RNG.

Only the fixed W @ spikes calculation is offloaded. No dense whole-brain matrix,
graph pruning, stochastic approximation, or GPU-generated noise is introduced.
"""
import numpy as np
import torch

SHADER = r'''
#include <metal_stdlib>
using namespace metal;
kernel void csr_dot(device float* out,
                    const device int* rowptr,
                    const device int* col,
                    const device float* weights,
                    const device float* state,
                    constant uint& slots,
                    uint index [[thread_position_in_grid]]) {
    uint row = index / slots;
    uint slot = index % slots;
    float value = 0.0f;
    for (int j = rowptr[row]; j < rowptr[row + 1]; ++j) {
        value = value + weights[j] * state[col[j] * slots + slot];
    }
    out[index] = value;
}
'''


class MetalCSR:
    def __init__(self, matrix):
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS device unavailable; run from the macOS terminal with Metal access')
        self.shape = matrix.shape
        self.kernel = torch.mps.compile_shader(SHADER)
        self.rowptr = torch.tensor(matrix.indptr.astype(np.int32), device='mps')
        self.col = torch.tensor(matrix.indices.astype(np.int32), device='mps')
        self.weights = torch.tensor(matrix.data, device='mps')

    def dot(self, state):
        if state.ndim != 2 or state.shape[0] != self.shape[1] or state.dtype != np.float32:
            raise ValueError('expected float32 [neurons, slots] spike state')
        value = torch.from_numpy(np.ascontiguousarray(state)).to('mps')
        output = torch.empty((self.shape[0], state.shape[1]), device='mps', dtype=torch.float32)
        self.kernel.csr_dot(output, self.rowptr, self.col, self.weights, value, state.shape[1])
        return output.cpu().numpy()


def install():
    from blackout_rl.v7_2.dynamics import WholeBrain
    original = WholeBrain.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.matrix = MetalCSR(self.matrix)

    WholeBrain.__init__ = initialize
