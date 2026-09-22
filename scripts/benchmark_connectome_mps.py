"""Compare CPU and MPS including transfers; check exact brain trajectories."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from blackout_rl.v7_registry import load_config, check_config, make_model
    from blackout_rl.v7_2.dynamics import WholeBrain
    from scripts.connectome_metal import MetalCSR
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available()
    cfg = load_config('configs/v7/main_study/v7_1_flywire.yaml')
    graph, _ = check_config(cfg)
    torch.manual_seed(11)
    cpu = make_model(cfg, graph).actor_model.eval()
    gpu = copy.deepcopy(cpu).to('mps')
    vectors = torch.rand(5, 96)
    graphic = torch.rand(5, 11, 96, 96)
    slots = torch.arange(5)
    result = {}
    with torch.no_grad():
        def cpu_encode():
            return cpu.encode(vectors, graphic, slots)
        def gpu_encode():
            return gpu.encode(vectors.to('mps'), graphic.to('mps'), slots.to('mps')).cpu()
        for name, fn in [('cpu_encoder', cpu_encode), ('mps_encoder_with_transfers', gpu_encode)]:
            for _ in range(10):fn()
            start = time.perf_counter()
            for _ in range(100):fn()
            result[name+'_ms'] = (time.perf_counter()-start)*10
        result['encoder_max_abs_difference'] = float((cpu_encode()-gpu_encode()).abs().max())
    cfg = load_config('configs/v7/main_study/v7_2_readout_ppo.yaml')
    graph, _ = check_config(cfg)
    brain = WholeBrain(graph, seed=11)
    cpu_matrix = brain.matrix
    gpu_matrix = MetalCSR(cpu_matrix)
    spikes = (np.random.default_rng(0).random((graph.n,1)) < .1).astype(np.float32)
    for name, matrix in [('cpu_csr', cpu_matrix), ('mps_csr_with_transfers', gpu_matrix)]:
        for _ in range(5):matrix.dot(spikes)
        start = time.perf_counter()
        for _ in range(40):matrix.dot(spikes)
        result[name+'_ms'] = (time.perf_counter()-start)*25
    result['csr_max_abs_difference'] = float(np.max(np.abs(cpu_matrix.dot(spikes)-gpu_matrix.dot(spikes))))
    sensory = np.random.default_rng(1).random((128,1,len(brain.input_indices))).astype(np.float32)
    original = brain.state_dict()
    traces = []
    finals = []
    for name, matrix in [('cpu',cpu_matrix), ('mps',gpu_matrix)]:
        brain.matrix = matrix
        brain.load_state_dict(original)
        trace = []
        start = time.perf_counter()
        for frame in sensory:
            rates = brain.step(frame)
            trace.append((rates.copy(), brain.spikes.copy()))
        result[name+'_brain_tick_ms'] = (time.perf_counter()-start)*1000/len(sensory)
        traces.append(trace)
        finals.append(brain.state_dict())
    result['brain_spikes_exact'] = all(np.array_equal(a[1],b[1]) for a,b in zip(*traces))
    result['brain_rates_exact'] = all(np.array_equal(a[0],b[0]) for a,b in zip(*traces))
    result['brain_voltage_max_abs_difference'] = float(np.max(np.abs(finals[0]['voltage']-finals[1]['voltage'])))
    output = ROOT/'reports/v7/acceleration_mps_comparison.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
