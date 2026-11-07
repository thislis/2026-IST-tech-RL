"""Separate event/sequence PPO. One actor is frozen for every rollout generation."""
import time
import torch
from .assignment import joint_log_prob
from .rollout import prepare_chunk,advantages,manager_returns


def _pad_time(value,length,left=False):
    padding=value.new_zeros((length-len(value),*value.shape[1:]))
    return torch.cat((padding,value) if left else (value,padding),0)


def worker_group(worker,sequences):
    """Batch all environments/units in CNN, temporal MLP and each recurrent tick."""
    hidden=torch.stack([s['hidden'] for s in sequences])
    burn=max(s['burn'] for s in sequences)
    with torch.no_grad():
        if burn:
            local=torch.stack([_pad_time(s['local'][:s['burn']],burn,True) for s in sequences],1)
            vector=torch.stack([_pad_time(s['vector'][:s['burn']],burn,True) for s in sequences],1)
            lagged=torch.stack([_pad_time(s['lags'][:s['burn']],burn,True) for s in sequences],1)
            reset=torch.stack([_pad_time(s['reset'][:s['burn']],burn,True) for s in sequences],1)
            present=torch.stack([_pad_time(s['present'][:s['burn']],burn,True) for s in sequences],1)
            encoded=worker.encode_lagged(local,vector,lagged)
            for t in range(burn):
                active=torch.tensor([t>=burn-s['burn'] for s in sequences],device=hidden.device)
                prior=hidden.masked_fill(reset[t,:,:,None],0)
                _,next_hidden=worker.step_encoded(encoded[t],prior)
                hidden=torch.where((active[:,None]&present[t])[:,:,None],next_hidden,hidden)
    hidden=hidden.detach()
    length=max(len(s['rows']) for s in sequences)
    local=torch.stack([_pad_time(s['local'][s['burn']:],length) for s in sequences],1)
    vector=torch.stack([_pad_time(s['vector'][s['burn']:],length) for s in sequences],1)
    lagged=torch.stack([_pad_time(s['lags'][s['burn']:],length) for s in sequences],1)
    reset=torch.stack([_pad_time(s['reset'][s['burn']:],length) for s in sequences],1)
    present=torch.stack([_pad_time(s['present'][s['burn']:],length) for s in sequences],1)
    encoded=worker.encode_lagged(local,vector,lagged);logits=[]
    for t in range(length):
        hidden=hidden.masked_fill(reset[t,:,:,None],0)
        score,updated=worker.step_encoded(encoded[t],hidden)
        hidden=torch.where(present[t,:,:,None],updated,hidden);logits.append(score)
    output=torch.stack(logits)
    return [output[:len(s['rows']),i] for i,s in enumerate(sequences)]


def manager_group(policy,sequences,device):
    features=[];legal=[];caps=[];actions=[];lagged=[];lengths=[]
    for seq in sequences:
        ids=seq['indices'];lengths.append(len(ids))
        for j in ids:
            event=seq['rows'][j]['manager']
            features.append(event['features']);legal.append(event['legal']);caps.append(event['caps']);actions.append(event['assignment'])
        if ids:lagged.append(seq['lags'][seq['burn']:][ids])
    if not features:return [None for _ in sequences]
    scores=policy.manager.from_lags(torch.stack(features).to(device),torch.cat(lagged))
    logp=joint_log_prob(scores,torch.stack(actions).to(device),torch.stack(caps).to(device),
                       torch.stack(legal).to(device),policy.temperature)
    result=[];start=0
    for size in lengths:
        result.append(logp[start:start+size] if size else None);start+=size
    return result


class Learner:
    def __init__(self,policy,critics,config,device):
        self.policy=policy.to(device);self.critics={k:v.to(device) for k,v in critics.items()}
        self.cfg=config;self.device=device;self.version=0
        self.optimizers={k:torch.optim.Adam(list(getattr(policy,k).parameters())+list(self.critics[k].parameters()),
                                           lr=config['learning_rate']) for k in ('manager','worker')}

    def update(self,chunks,phase,imitation=False):
        if not chunks:return {}
        versions={c['version'] for c in chunks}
        if versions!={self.version}:raise ValueError('mixed or stale policy versions in PPO batch')
        if {c['phase'] for c in chunks}!={phase}:raise ValueError('mixed actor stages')
        cfg=self.cfg;policy=self.policy;critic=self.critics[phase]
        for name in ('manager','worker'):
            for p in getattr(policy,name).parameters():p.requires_grad_(name==phase)
        prepared=[]
        lags=cfg.get('manager_lags',cfg['lags']) if phase=='manager' else cfg['lags']
        for chunk in chunks:
            seq=prepare_chunk(chunk,lags,self.device)
            if phase=='worker':
                rows=chunk['rows'];ids=list(range(len(rows)))
                adv,ret=advantages([r['reward'] for r in rows],[r['value'] for r in rows],
                    [r['done'] for r in rows],chunk['bootstrap'],cfg['gamma'],cfg['gae_lambda'])
            else:ids,adv,ret=manager_returns(chunk,cfg)
            seq.update(indices=ids,advantages=adv.to(self.device),returns=ret.to(self.device))
            prepared.append(seq)
        population=torch.cat([s['advantages'] for s in prepared])
        if not len(population):return dict(skipped='no decision events')
        mean=population.mean();std=population.std(unbiased=False).clamp_min(1e-6)
        for s in prepared:s['advantages']=(s['advantages']-mean)/std
        started=time.perf_counter();statistics=[];stop=False
        # Accumulate whole sequences up to the measured device batch target.
        accumulation=max(1,int(cfg.get('sequence_minibatch',4)))
        for epoch in range(cfg['ppo_epochs']):
            order=torch.randperm(len(prepared)).tolist()
            for offset in range(0,len(order),accumulation):
                group=[prepared[j] for j in order[offset:offset+accumulation]]
                optimizer=self.optimizers[phase];optimizer.zero_grad(set_to_none=True)
                losses=[];kls=[];differentiable=[]
                worker_logits=worker_group(policy.worker,group) if phase=='worker' else []
                manager_logps=manager_group(policy,group,self.device) if phase=='manager' else []
                for seq_index,seq in enumerate(group):
                    burn=seq['burn'];rows=seq['rows'];lagged=seq['lags'][burn:]
                    if phase=='worker':
                        logits=worker_logits[seq_index]
                        distribution=torch.distributions.Categorical(logits=logits)
                        actions=torch.stack([r['action'] for r in rows]).to(self.device)
                        present=seq['present'][burn:]
                        logp=(distribution.log_prob(actions)*present).sum(-1)
                        entropy=(distribution.entropy()*present).sum()/present.sum().clamp_min(1)
                        old=torch.stack([r['old_log_prob'] for r in rows]).to(self.device)
                        if imitation:
                            mask=torch.stack([r['bc_mask'] for r in rows]).to(self.device)
                            if not mask.any():continue
                            actor_loss=-distribution.log_prob(actions)[mask].mean()
                    else:
                        ids=seq['indices']
                        if not ids:continue
                        ev=[rows[j]['manager'] for j in ids]
                        logp=manager_logps[seq_index]
                        old=torch.stack([e['old_log_prob'] for e in ev]).to(self.device)
                        # Entropy disabled for manager by default; exact second derivatives are costly.
                        entropy=logp.new_zeros(())
                        if imitation:
                            mask=torch.tensor([e['bc_valid'] for e in ev],device=self.device)
                            if not mask.any():continue
                            actor_loss=-logp[mask].mean()/5.
                    delta=logp-old
                    ratio=delta.exp()
                    kl=((ratio-1)-delta).mean()
                    if not imitation:
                        unclipped=ratio*seq['advantages']
                        clipped=ratio.clamp(1-cfg['clip'],1+cfg['clip'])*seq['advantages']
                        actor_loss=-torch.minimum(unclipped,clipped).mean()-cfg['entropy_coef']*entropy
                    if imitation:loss=actor_loss
                    else:
                        value=critic(lagged if phase=='worker' else lagged[seq['indices']])
                        value_loss=.5*(value-seq['returns']).square().mean()
                        loss=actor_loss+cfg['value_coef']*value_loss
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
                    differentiable.append(loss)
                    losses.append(float(loss.detach()));kls.append(float(kl.detach()))
                if losses:
                    torch.stack(differentiable).mean().backward()
                    params=[p for g in optimizer.param_groups for p in g['params']]
                    norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'])
                    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
                    optimizer.step()
                    statistics.append(dict(loss=sum(losses)/len(losses),kl=sum(kls)/len(kls),grad_norm=float(norm)))
                    if not imitation and statistics[-1]['kl']>cfg['target_kl']:
                        stop=True;break
            if stop:break
        self.version+=1
        return dict(version=self.version,updates=len(statistics),epochs=epoch+1,
                    wall_seconds=time.perf_counter()-started,statistics=statistics)
