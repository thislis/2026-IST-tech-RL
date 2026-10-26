"""Self-contained two-file submission and explicit-state ONNX development core."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
from .io import atomic_json, atomic_torch, cpu_state, file_hash


def export_policy(policy,directory):
    directory=Path(directory); directory.parent.mkdir(parents=True,exist_ok=True)
    if directory.exists(): raise FileExistsError(directory)
    with tempfile.TemporaryDirectory(prefix='.export-',dir=directory.parent) as temp:
        staging=Path(temp)/'bundle';staging.mkdir()
        shutil.copyfile(Path(__file__).with_name('policy.py'),staging/'policy.py')
        atomic_torch(staging/'checkpoint.pt',{'policy_state':cpu_state(policy)})
        os.rename(staging,directory)
    return directory


def ensure_export(policy,directory):
    directory=Path(directory)
    if not directory.exists(): return export_policy(policy,directory)
    payload=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)['policy_state']
    expected=cpu_state(policy)
    if payload.keys()!=expected.keys() or any(not torch.equal(payload[k],expected[k]) for k in payload):
        raise ValueError('immutable export checkpoint differs')
    if file_hash(directory/'policy.py')!=file_hash(Path(__file__).with_name('policy.py')):
        raise ValueError('immutable export policy source differs')
    return directory


def load_export(directory,device='cpu',planner_device=None):
    from blackout_env import load_checkpoint
    directory=Path(directory)
    spec=importlib.util.spec_from_file_location('submitted_v11_'+file_hash(directory/'checkpoint.pt')[:12],directory/'policy.py')
    module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True;spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode=previous
    model=load_checkpoint(module.MyPolicy,str(directory/'checkpoint.pt'),state_dict_key='policy_state',
                          device=device,vector_size=96,n_channels=11)
    if planner_device is not None: model._net.configure_devices(planner_device=planner_device)
    return model


def verify_export(directory,reference=None,fixture=None):
    from .policy import MyPolicy
    from .benchmark import fixture as make_fixture
    directory=Path(directory).resolve()
    if {p.name for p in directory.iterdir()}!={'policy.py','checkpoint.pt'}:
        raise ValueError('submission must contain exactly policy.py and checkpoint.pt')
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)['policy_state']
    if reference is not None:
        expected=cpu_state(reference)
        if state.keys()!=expected.keys() or any(not torch.equal(state[k],expected[k]) for k in state):
            raise ValueError('export changed model parameters/config')
    vector,graphic=make_fixture() if fixture is None else fixture
    if graphic.ndim==3: graphic=graphic[None].expand(5,-1,-1,-1)
    policy=MyPolicy().eval();policy.load_state_dict(state)
    vectors=[];outputs=[]
    # Observed-clock reset and physical team swap are exercised in one sequence.
    for i in range(12):
        v=vector.clone();v[:,95]=1-(i%6)/21000.
        if i>=6: v[:,:90].reshape(5,10,9)[:,:,2].mul_(-1)
        vectors.append(v)
        with torch.inference_mode():
            torch.manual_seed(990+i);outputs.append(policy(v,graphic).clone())
    code=r'''
import importlib.util,json,sys,time,torch
from pathlib import Path
from blackout_env import load_checkpoint
torch.set_num_threads(1);sys.dont_write_bytecode=True
d=Path(sys.argv[1]);data=torch.load(sys.argv[2],weights_only=True)
s=importlib.util.spec_from_file_location('submission',d/'policy.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
start=time.perf_counter()
p=load_checkpoint(m.MyPolicy,str(d/'checkpoint.pt'),state_dict_key='policy_state',device='cpu',vector_size=96,n_channels=11)
cold=time.perf_counter()-start
with torch.inference_mode():
 for i,(v,expected) in enumerate(zip(data['vectors'],data['outputs'])):
  side=0 if v[0,2]>0 else 1
  obs={f'unit_{side*5+j}':dict(vector=v[j].numpy(),graphic=data['graphic'][j].permute(1,2,0).numpy()) for j in range(5)}
  torch.manual_seed(990+i);out=p.act(obs)
  actual=torch.stack([torch.from_numpy(out[n]) for n in obs])
  assert actual.shape==(5,2) and actual.dtype==torch.float32
  assert torch.isfinite(actual).all() and (actual.abs()<=1).all()
  torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
assert not any(k.startswith(('blackout_v11','blackout_rl')) for k in sys.modules)
print(json.dumps(dict(passed=True,provided_loader=True,isolated=True,sequence_calls=len(data['vectors']),clock_and_side_reset=True,cold_load_seconds=cold)))
'''
    with tempfile.TemporaryDirectory(prefix='v11-export-') as temp:
        temp=Path(temp); isolated=temp/'submission';shutil.copytree(directory,isolated)
        torch.save(dict(vectors=vectors,graphic=graphic,outputs=outputs),temp/'fixture.pt')
        result=subprocess.run([sys.executable,'-I','-c',code,str(isolated),str(temp/'fixture.pt')],
                              cwd=temp,text=True,capture_output=True,timeout=180)
        if result.returncode: raise RuntimeError('isolated provided loader check failed: '+result.stderr)
        report=json.loads(result.stdout.strip().splitlines()[-1])
    report.update(files={p.name:file_hash(p) for p in directory.iterdir()},official_server_certified=False)
    return report


def export_onnx(policy,directory,sequence_length=32):
    """Development graph uses explicit state ports; never adds submission inputs."""
    import copy
    import onnx
    import onnxruntime as ort
    from .policy import ExportCore, STATE_KEYS
    from .benchmark import fixture
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=copy.deepcopy(policy).cpu().eval()
    model.configure_devices(planner_device='cpu')
    # ONNX candidate is deterministic. Sampled decoder stays a separate PyTorch policy.
    model.configure(model.arm,'greedy',max_interventions=model.max_interventions,
                    goal_period=model.goal_period,residual_period=model.residual_period,use_memory=model.use_memory)
    core=ExportCore(model).eval()
    vector,graphic=fixture();v=vector[None];g=graphic[:1]
    state=model.initial_state(1,device='cpu')
    names=['vector','graphic']+['state_'+k for k in STATE_KEYS]
    outputs=['action']+['next_'+k for k in STATE_KEYS]
    path=directory/'core.onnx'
    with torch.no_grad():
        torch.onnx.export(core,(v,g,*[state[k] for k in STATE_KEYS]),str(path),dynamo=False,
            opset_version=17,input_names=names,output_names=outputs,do_constant_folding=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    allowed={x.name for x in session.get_inputs()}
    tensor_state=[state[k] for k in STATE_KEYS]; ort_state=[s.numpy() for s in tensor_state]
    errors=[]
    with torch.inference_mode():
        for i in range(sequence_length):
            vi=v.clone();vi[:,:,95]=1-(i%max(2,sequence_length//2))/21000.
            if i>=sequence_length//2: vi[:,:,:90].reshape(1,5,10,9)[:,:,:,2].mul_(-1)
            expected=core(vi,g,*tensor_state)
            feeds=dict(zip(names,[vi.numpy(),g.numpy(),*ort_state]))
            actual=session.run(None,{k:x for k,x in feeds.items() if k in allowed})
            if len(actual)!=len(expected): raise ValueError('ONNX state outputs lost')
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b.cpu().numpy(),atol=1e-5,rtol=1e-4)
            errors.append(float(np.max(np.abs(actual[0]-expected[0].cpu().numpy()))))
            tensor_state=list(expected[1:]);ort_state=actual[1:]
    report=dict(passed=True,sha256=file_hash(path),opset=17,provider='CPUExecutionProvider',
        sequence_calls=sequence_length,max_action_error=max(errors),inputs=names,outputs=outputs,
        state_keys=list(STATE_KEYS),public_submission=False,decoder='greedy',
        runtime_requirement='caller feeds next state into the next call; no implicit ONNX persistence')
    atomic_json(directory/'manifest.json',report)
    return report
