"""Full observation/state graph export and multi-step parity, no Unity required."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import initial_state,SCHEMA

INPUTS=('vector','graphic','slot','state_in_history','state_in_worker','state_in_task','state_in_nav')
OUTPUTS=('action','state_out_history','state_out_worker','state_out_task','state_out_nav')


def synthetic_observation():
    v=torch.zeros(5,96);e=torch.zeros(10,9)
    e[:5,0]=.08;e[:5,1]=torch.linspace(.78,.94,5);e[:5,2]=1
    e[5:,0]=torch.linspace(.78,.94,5);e[5:,1]=.08;e[5:,2]=-1;e[:,3]=1
    v[:,:90]=e.flatten();v[:,90]=1;v[:,95]=1
    sem=torch.zeros(11,24,24)
    for y,x in ((21,1),(16,4),(12,5),(8,8),(4,12)):sem[2,y,x]=1
    for y,x in ((18,4),(15,8),(9,12),(3,16)):sem[6,y,x]=1
    g=sem.flip(-2).repeat_interleave(4,-2).repeat_interleave(4,-1)[None].repeat(5,1,1,1)
    return v,g


def export_policy(policy,directory,verify=True):
    import copy
    import onnx
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=copy.deepcopy(policy).cpu().eval()
    vector,graphic=synthetic_observation()
    example=(vector,graphic,torch.arange(5),*initial_state())
    script=torch.jit.script(model)
    path=directory/'model.onnx'
    torch.onnx.export(script,example,str(path),input_names=list(INPUTS),output_names=list(OUTPUTS),
                      opset_version=18,dynamo=False,external_data=False)
    onnx.checker.check_model(str(path))
    (directory/'meta.json').write_text(json.dumps(dict(mode='team',frame_stack=1)),encoding='utf-8')
    result=dict(schema=SCHEMA,full_graph=True,bytes=path.stat().st_size,
                tested_scope='synthetic CPU parity; official container latency remains unmeasured')
    if verify:result.update(parity(model,path))
    (directory/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def parity(model,path,providers=None,steps=8):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=3
    session=ort.InferenceSession(str(path),sess_options=options,providers=providers or ['CPUExecutionProvider'])
    names={i.name for i in session.get_inputs()}
    v,g=synthetic_observation();reference=initial_state();state=[s.numpy() for s in initial_state()]
    errors=[];times=[]
    with torch.no_grad():
        for step in range(steps):
            if step==4:
                v,g=synthetic_observation()
                entities=v[0,:90].reshape(10,9).clone()[torch.tensor([5,6,7,8,9,0,1,2,3,4])]
                entities[:,:2]=entities[:,:2].flip(-1)
                v[:,:90]=entities.flatten();g=g.transpose(-1,-2).flip((-1,-2)).contiguous()
                reference=initial_state();state=[s.numpy() for s in initial_state()]
            v[:,95]=1-step*.02/420.
            if step==2:
                # Force a later decision branch as well as the cache-hit branch.
                reference[2][:,5]=-1000;state[2][:,5]=-1000
            observed_v=v.clone();observed_g=g.clone()
            if step==5:observed_v[2]=0;observed_g[2]=0
            if step>=6:
                # Return of a missing slot plus a role/pickup event on team B.
                observed_v[0,90:93]=torch.tensor([0.,0.,1.])
                observed_v[:,5*9+3]=0;observed_v[:,5*9+4]=1
            result=model(observed_v,observed_g,torch.arange(5),*reference)
            arrays=[observed_v.numpy(),observed_g.numpy(),np.arange(5,dtype=np.int64),*state]
            feed={k:a for k,a in zip(INPUTS,arrays) if k in names}
            start=time.perf_counter();actual=session.run(list(OUTPUTS),feed);times.append(time.perf_counter()-start)
            if not np.allclose(actual[3],result[3].numpy(),atol=3e-4,rtol=3e-5):
                raise AssertionError(f'task state differs: ORT={actual[3].tolist()} Torch={result[3].tolist()}')
            for i,(expected,observed) in enumerate(zip(result,actual)):
                np.testing.assert_allclose(observed,expected.numpy(),atol=3e-4,rtol=3e-5,
                                           err_msg=f'step {step}, output {OUTPUTS[i]}')
                errors.append(float(np.max(np.abs(observed-expected.numpy()))))
            reference=result[1:];state=actual[1:]
    return dict(parity_steps=steps,max_absolute_error=max(errors),synthetic_call_seconds=times,
                actual_inputs=sorted(names),providers=session.get_providers())
