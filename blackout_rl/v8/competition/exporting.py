"""Two-file competition artifacts, tested with the supplied loader in isolation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch
from blackout_rl.v8.checkpoints import atomic_json,fingerprints
from blackout_rl.v8.contracts import file_hash
from .policy import MyPolicy


def export_policy(payload,output,*,allow_untrained_fixture=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if payload.get('sources')!=fingerprints():raise ValueError('export source differs from training checkpoint')
    if payload.get('global_step',0)<=0 and not allow_untrained_fixture:raise ValueError('refuse untrained competition submission')
    policy=MyPolicy();policy.load_state_dict(payload['policy_state'],strict=True)
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name('policy.py'),output/'policy.py')
    torch.save({'policy_state':policy.state_dict()},output/'checkpoint.pt')
    return output


def verify_export(output,reference_state,fixture_path=None):
    output=Path(output).resolve();reference=MyPolicy().eval();reference.load_state_dict(reference_state)
    actual_state=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True)['policy_state']
    if actual_state.keys()!=reference_state.keys():raise ValueError('submission state keys differ')
    for name,value in reference_state.items():
        if not torch.equal(actual_state[name].cpu(),value.cpu()):raise ValueError('submission weights differ: '+name)
    # Synthetic legal tensors exercise the interface; no match or Unity process.
    generator=torch.Generator().manual_seed(9029)
    vector=torch.rand(5,96,generator=generator)
    vector[:,:90].reshape(5,10,9)[:,:,2]=torch.tensor([1]*5+[-1]*5)
    graphic=torch.nn.functional.one_hot(torch.randint(0,11,(5,96,96),generator=generator),11).permute(0,3,1,2).float()
    with torch.no_grad():expected=reference(vector,graphic)
    recorded=[]
    if fixture_path is not None:
        recorded=torch.load(fixture_path,map_location='cpu',weights_only=True)
        if not recorded:raise ValueError('recorded policy inputs are empty')
        with torch.no_grad():
            recorded=[dict(row,expected=reference(row['vector'],row['graphic'])) for row in recorded]
    with tempfile.TemporaryDirectory(prefix='v8-submission-check-') as tmp:
        tmp=Path(tmp);torch.save(dict(vector=vector,graphic=graphic,expected=expected,recorded=recorded),tmp/'inputs.pt')
        code=r'''
import importlib.util,sys,json,torch
from pathlib import Path
from blackout_env import load_checkpoint
sys.dont_write_bytecode=True
torch.set_num_threads(1)
directory=Path(sys.argv[1]);data=torch.load(sys.argv[2],weights_only=True)
spec=importlib.util.spec_from_file_location('submitted_policy',directory/'policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model=load_checkpoint(module.MyPolicy,str(directory/'checkpoint.pt'),state_dict_key='policy_state',device='cpu',vector_size=96,n_channels=11)
v,g=data['vector'],data['graphic'];net=model._net
assert issubclass(module.MyPolicy,torch.nn.Module)
for count in (0,1,3,5):
    a=net(v[:count],g[:count]);assert a.shape==(count,2) and a.dtype==torch.float32
    assert torch.isfinite(a).all() and (a.abs()<=1).all()
a=net(v,g);torch.testing.assert_close(a,data['expected'],rtol=0,atol=0)
order=torch.tensor([4,1,3,0,2]);torch.testing.assert_close(net(v[order],g[order]),a[order],rtol=0,atol=0)
torch.testing.assert_close(torch.cat([net(v[i:i+1],g[i:i+1]) for i in range(5)]),a,rtol=0,atol=0)
obs={f'unit_{i}':dict(vector=v[i].numpy(),graphic=g[i].permute(1,2,0).numpy()) for i in range(5)}
actual=model.act(obs)
for i in range(5):torch.testing.assert_close(torch.from_numpy(actual[f'unit_{i}']),a[i],rtol=0,atol=0)
net(torch.zeros_like(v),torch.zeros_like(g));torch.testing.assert_close(net(v,g),a,rtol=0,atol=0)
for row in data['recorded']:
    torch.testing.assert_close(net(row['vector'],row['graphic']),row['expected'],rtol=0,atol=0)
assert not any(k=='blackout_rl' or k.startswith('blackout_rl.') for k in sys.modules)
print(json.dumps(dict(passed=True,provided_loader=True,isolated=True,batches=[0,1,3,5],row_permutation=True,repeat_calls=True,device='cpu')))
'''
        run=subprocess.run([sys.executable,'-I','-c',code,str(output),str(tmp/'inputs.pt')],cwd=tmp,capture_output=True,text=True,timeout=120)
        if run.returncode:raise RuntimeError('isolated submission verification failed: '+run.stderr)
        report=json.loads(run.stdout)
    report.update(files={name:file_hash(output/name) for name in ('policy.py','checkpoint.pt')},
                  recorded_observation_batches=len(recorded),recorded_inputs_sha256=file_hash(fixture_path) if fixture_path is not None else None,
                  live_match_executed=False,external_submission_sent=False,competition_resource_limits_known=False)
    if {p.name for p in output.iterdir()}!={'policy.py','checkpoint.pt'}:raise ValueError('submission must contain exactly two files')
    return report
