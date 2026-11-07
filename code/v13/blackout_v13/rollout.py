"""Bounded sequence preparation; store raw tokens and FP16 crops, not full frame stacks."""
import torch
from .history import History


def prepare_chunk(chunk,lags,device):
    # Reconstruct compact history on CPU, then transfer contiguous batches once.
    # Per-token CUDA copies/kernels dominate this small model otherwise.
    history=chunk['history']
    hidden=chunk['hidden'].to(device)
    sampler=History(tuple(lags))
    prefix=chunk['prefix'];rows=chunk['rows']
    all_rows=prefix+rows
    lagged=[]
    for row in all_rows:
        clear=row['reset']
        history=history.masked_fill(clear[:,None,None],0)
        token=row['token']
        lagged.append(sampler(token,history))
        history=torch.where(row['present'][:,None,None],torch.cat((token[:,None],history[:,:63]),1),history)
    return dict(lags=torch.stack(lagged).to(device),
        local=torch.stack([r['local'] for r in all_rows]).to(device,dtype=torch.float32),
        vector=torch.stack([r['vector'] for r in all_rows]).to(device),
        reset=torch.stack([r['reset'] for r in all_rows]).to(device),
        present=torch.stack([r['present'] for r in all_rows]).to(device),hidden=hidden,burn=len(prefix),rows=rows)


def advantages(rewards,values,dones,bootstrap,gamma=1.,lam=.9995,durations=None):
    n=len(rewards);out=torch.zeros(n,dtype=torch.float32);acc=0.;next_value=float(bootstrap)
    durations=[1]*n if durations is None else durations
    for j in range(n-1,-1,-1):
        live=0. if dones[j] else 1.
        delta=float(rewards[j])+live*gamma**durations[j]*next_value-float(values[j])
        acc=delta+live*(gamma*lam)**durations[j]*acc
        out[j]=acc;next_value=float(values[j])
    return out,out+torch.as_tensor(values,dtype=torch.float32)


def manager_returns(chunk,config):
    rows=chunk['rows'];indices=[i for i,r in enumerate(rows) if r['manager'] is not None]
    if not indices:return indices,torch.zeros(0),torch.zeros(0)
    rewards=[];values=[];dones=[];durations=[]
    for k,start in enumerate(indices):
        end=indices[k+1] if k+1<len(indices) else len(rows)
        segment=rows[start:end]
        rewards.append(sum(config['gamma']**j*float(r['reward']) for j,r in enumerate(segment)))
        values.append(float(rows[start]['value']));dones.append(any(r['done'] for r in segment));durations.append(end-start)
    adv,ret=advantages(rewards,values,dones,chunk['bootstrap'],config['gamma'],config['gae_lambda'],durations)
    return indices,adv,ret
