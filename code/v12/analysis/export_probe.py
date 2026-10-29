"""Short synthetic ONNX event-boundary parity; never opens a game environment."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from blackout_v12.benchmark import fixture
from blackout_v12.io import ROOT, atomic_json
from blackout_v12.onnx_core import OnnxEventBackend, export_event_core
from blackout_v12.policy import MyPolicy


def run():
    torch.set_num_threads(1)
    torch.manual_seed(12026)
    native = MyPolicy(arm='J').eval()
    # Nonzero synthetic residual exercises the exported MLP, without training.
    with torch.no_grad():
        native.scorer.layers[-1].weight.copy_(torch.linspace(-.01,.01,32).reshape(1,32))
    directory = ROOT/'artifacts/submission/v12/preparation_onnx'
    report = export_event_core(native, directory, probes=6)
    exported = copy.deepcopy(native)
    backend = OnnxEventBackend(directory/'event_core.onnx')
    exported.event_backend = backend
    original_vector, original_graphic = fixture()
    clocks = [1.,1.-.02/420,1.-.04/420,1.-.6/420,1.,
              1.-.02/420,1.-.04/420,1.-.06/420,1.,1.-.02/420,1.-.04/420,1.-.6/420]
    event_count = 0
    with torch.inference_mode():
        for index, clock in enumerate(clocks):
            vector, graphic = original_vector.copy(), original_graphic.copy()
            vector[:,95] = clock
            if index >= 8:
                vector[:,:90].reshape(5,10,9)[:,:,2] *= -1
                graphic[:,[2,3]] = graphic[:,[3,2]]
            if index in (5,6):
                vector[:,3:9] = 0
                vector[:,4] = 1
            if index == 7:
                vector[:,93] = .01
            left = native.step(vector,graphic,deterministic=True)
            right = exported.step(vector,graphic,deterministic=True)
            torch.testing.assert_close(left['action'],right['action'],atol=1e-5,rtol=1e-4)
            assert left['event'] == right['event']
            event_count += int(left['event'])
            assert [(u.task_id,u.phase,u.source,u.destination) for u in native.runtime.units] == [
                (u.task_id,u.phase,u.source,u.destination) for u in exported.runtime.units]
            if index == 6:
                restored = MyPolicy(arm='J').eval()
                restored.load_state_dict(exported.state_dict())
                restored.load_runtime_state(exported.runtime_state())
                restored.event_backend = backend
                exported = restored
    assert backend.calls == event_count < len(clocks)
    report.update(host_sequence_calls=len(clocks),host_event_calls=backend.calls,
                  maintenance_backend_calls=0,host_state_transfer=True,
                  reset_side_pickup_sequence=True,synthetic_residual=True,
                  checkpoint_trained=False,unity_started=False)
    atomic_json(ROOT/'logs/v12/preparation/onnx_probe.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    run()
