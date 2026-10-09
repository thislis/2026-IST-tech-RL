"""Immutable two-file exports and isolated verification with the provided loader."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch
from .io import atomic_torch, cpu_state, file_hash
from .policy import MyPolicy


def export_policy(policy, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).with_name("policy.py"), directory / "policy.py")
    atomic_torch(directory / "checkpoint.pt", {"policy_state": cpu_state(policy)})
    return directory


def verify_export(directory, reference=None, fixture=None):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != {"policy.py", "checkpoint.pt"}:
        raise ValueError("export must contain exactly policy.py and checkpoint.pt")
    if file_hash(directory / "policy.py") != file_hash(Path(__file__).with_name("policy.py")):
        raise ValueError("export policy source differs")
    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)["policy_state"]
    if reference is not None:
        ref = cpu_state(reference)
        if state.keys() != ref.keys() or any(not torch.equal(state[k], ref[k]) for k in ref):
            raise ValueError("export weights differ")
    policy = MyPolicy().eval(); policy.load_state_dict(state)
    rng = torch.Generator().manual_seed(909)
    vector = torch.rand(10, 96, generator=rng)
    graphic = torch.nn.functional.one_hot(torch.randint(11, (10, 96, 96), generator=rng), 11).permute(0, 3, 1, 2).float()
    with torch.inference_mode():
        data = {"vector": vector, "graphic": graphic, "log_probs": policy.log_probs(vector, graphic)}
        if fixture is not None:
            fv, fg = fixture
            data["recorded"] = {"vector": fv, "graphic": fg, "log_probs": policy.log_probs(fv, fg)}
    code = r'''
import importlib.util,json,sys,time,torch
from pathlib import Path
from blackout_env import load_checkpoint
torch.set_num_threads(1)
sys.dont_write_bytecode=True
directory=Path(sys.argv[1]); data=torch.load(sys.argv[2],weights_only=True)
spec=importlib.util.spec_from_file_location('submitted_policy',directory/'policy.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
started=time.perf_counter()
loaded=load_checkpoint(module.MyPolicy,str(directory/'checkpoint.pt'),state_dict_key='policy_state',device='cpu',vector_size=96,n_channels=11)
cold=time.perf_counter()-started
p=loaded._net; v,g=data['vector'],data['graphic']
with torch.inference_mode():
 for n in (0,1,3,5,10):
  a=p(v[:n],g[:n]); assert a.shape==(n,2) and a.dtype==torch.float32
  assert torch.isfinite(a).all() and (a.abs()<=1).all()
 logs=p.log_probs(v,g); torch.testing.assert_close(logs,data['log_probs'],atol=1e-6,rtol=1e-5)
 order=torch.tensor([4,2,1,7,0,3,8,5,9,6])
 torch.testing.assert_close(p.log_probs(v[order],g[order]),logs[order],atol=2e-6,rtol=1e-5)
 torch.testing.assert_close(torch.cat([p.log_probs(v[i:i+1],g[i:i+1]) for i in range(10)]),logs,atol=2e-6,rtol=1e-5)
 torch.manual_seed(19); expected=p(v[:5],g[:5])
 obs={f'unit_{i}':dict(vector=v[i].numpy(),graphic=g[i].permute(1,2,0).numpy()) for i in range(5)}
 torch.manual_seed(19); actual=loaded.act(obs)
 for i in range(5): torch.testing.assert_close(torch.from_numpy(actual[f'unit_{i}']),expected[i],rtol=0,atol=0)
 p.train(); train_logs=p.log_probs(v,g); p.eval()
 torch.testing.assert_close(train_logs,p.log_probs(v,g),atol=1e-6,rtol=1e-5)
 if 'recorded' in data:
  row=data['recorded']; torch.testing.assert_close(p.log_probs(row['vector'],row['graphic']),row['log_probs'],atol=2e-6,rtol=1e-5)
 durations=[]
 for _ in range(12):
  start=time.perf_counter();p(v[:5],g[:5]);durations.append(time.perf_counter()-start)
 assert not any(k.split('.')[0] in ('blackout_rl','blackout_v9') for k in sys.modules)
print(json.dumps(dict(passed=True,provided_loader=True,isolated=True,batches=[0,1,3,5,10],distribution_parity=True,cold_load_seconds=cold,warm_cpu_forward_seconds=sorted(durations),live_match_executed=False)))
'''
    with tempfile.TemporaryDirectory(prefix="v9-export-check-") as tmp:
        tmp = Path(tmp)
        isolated = tmp / "submission"
        shutil.copytree(directory, isolated)
        torch.save(data, tmp / "fixture.pt")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(isolated), str(tmp / "fixture.pt")],
                                cwd=tmp, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("isolated loader check failed: " + result.stderr)
        report = json.loads(result.stdout)
    report["files"] = {p.name: file_hash(p) for p in directory.iterdir()}
    report["official_server_certified"] = False
    return report


def load_export(directory):
    import importlib.util
    from blackout_env import load_checkpoint
    directory = Path(directory)
    spec = importlib.util.spec_from_file_location("submitted_v9_policy", directory / "policy.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return load_checkpoint(module.MyPolicy, str(directory / "checkpoint.pt"), state_dict_key="policy_state",
                           device="cpu", vector_size=96, n_channels=11)
