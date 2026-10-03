"""Single joint-ratio PPO with module diagnostics and immutable update boundaries."""
import random
import time
import torch
from .contracts import ContractError,digest
from .checkpoints import CheckpointStore,fingerprints,atomic_json
from .telemetry import append

DEFAULT_PPO=dict(learning_rate=1e-4,epochs=4,minibatch_size=128,clip=.2,value_clip=.2,value_loss_coef=.5,entropy_coef=.003,max_grad_norm=.5,target_kl=.02)

def update(model,optimizer,batch,config,generator=None):
    if set(config)-set(DEFAULT_PPO):raise ContractError('unknown PPO setting')
    cfg=dict(DEFAULT_PPO,**config)
    if len(torch.unique(batch['behavior_version']))!=1:raise ContractError('mixed behavior versions')
    count=len(batch['action'])
    if count<1:raise ContractError('empty batch')
    advantage=batch['advantage'];advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    before={n:p.detach().clone() for n,p in params.items()}
    metrics=[];early=False;epochs_complete=0
    for epoch in range(cfg['epochs']):
        for indices in torch.randperm(count,generator=generator).split(cfg['minibatch_size']):
            dist=model.distribution(batch['features'][indices],batch['valid'][indices])
            logratio=dist.log_prob(batch['action'][indices])-batch['old_log_prob'][indices]
            ratio=logratio.exp();kl=((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):raise ContractError('nonfinite PPO ratio')
            if cfg['target_kl'] is not None and float(kl.detach())>cfg['target_kl']:
                early=True;break
            actor_loss=torch.maximum(-advantage[indices]*ratio,-advantage[indices]*ratio.clamp(1-cfg['clip'],1+cfg['clip'])).mean()
            value=model.critic(batch['central'][indices]).squeeze(-1)
            error=(value-batch['return_'][indices]).square()
            if cfg['value_clip'] is not None:
                clipped=batch['old_value'][indices]+(value-batch['old_value'][indices]).clamp(-cfg['value_clip'],cfg['value_clip'])
                error=torch.maximum(error,(clipped-batch['return_'][indices]).square())
            value_loss=.5*error.mean();entropy=dist.entropy().mean()
            loss=actor_loss+cfg['value_loss_coef']*value_loss-cfg['entropy_coef']*entropy
            if not torch.isfinite(loss):raise ContractError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norms={prefix:float(torch.sqrt(sum((p.grad.square().sum() for n,p in params.items() if n.startswith(prefix) and p.grad is not None),torch.tensor(0.)))) for prefix in ('trunk','gate','correction','critic')}
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            metrics.append(dict(policy_loss=float(actor_loss.detach()),value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),
                kl=float(kl.detach()),clip_fraction=float(((ratio-1).abs()>cfg['clip']).float().mean()),gradient_norm=float(norm),
                gradient_clipped=float(norm)>cfg['max_grad_norm'],module_gradient_norm=norms))
        if early:break
        epochs_complete+=1
    with torch.no_grad():
        prediction=model.critic(batch['central']).squeeze(-1);variance=batch['return_'].var(unbiased=False)
        ev=None if float(variance)<1e-12 else float(1-(batch['return_']-prediction).var(unbiased=False)/variance)
    delta={prefix:float(torch.sqrt(sum(((p-before[n]).square().sum() for n,p in params.items() if n.startswith(prefix)),torch.tensor(0.))).detach()) for prefix in ('trunk','gate','correction','critic')}
    by_action={}
    for name,mask in [('keep',batch['action']==0),('correction',batch['action']!=0)]:
        values=batch['advantage'][mask];by_action[name]=dict(n=len(values),mean=float(values.mean()) if len(values) else None,variance=float(values.var(unbiased=False)) if len(values) else None)
    return dict(minibatches=len(metrics),epochs_completed=epochs_complete,early_stopped=early,samples=count,
                explained_variance=ev,raw_return_variance=float(batch['return_'].var(unbiased=False)),advantages=by_action,
                module_parameter_delta=delta,minibatch_metrics=metrics)


def train(collector,directory,*,resume=False,stop_requested=None,on_checkpoint=None):
    from pathlib import Path
    import fcntl
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    config=collector.config;model=collector.policy.model
    store=CheckpointStore(directory/'checkpoints');parent=None
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['training']['ppo']['learning_rate'])
    ppo_rng=torch.Generator().manual_seed(config['training']['seed']+60000)
    source=fingerprints()
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            payload,parent=store.load()
            if payload['config_sha256']!=digest(config) or payload['sources']!=source:raise ContractError('resume source/config changed')
            # Preserve failed suffixes as evidence, then restore the committed log boundary.
            recovery=directory/'recovery';recovery.mkdir(exist_ok=True)
            for path in directory.glob('*.jsonl'):
                if path.name in ('attempts.jsonl','resume.jsonl'):continue
                offset=payload['log_offsets'].get(path.name,0)
                if path.stat().st_size<offset:raise ContractError('log shorter than checkpoint boundary')
                with path.open('rb') as f:f.seek(offset);tail=f.read()
                if tail:
                    import hashlib
                    archive=recovery/(hashlib.sha256(tail).hexdigest()+'-'+path.name)
                    if not archive.exists():archive.write_bytes(tail)
                    with path.open('r+b') as f:f.truncate(offset)
            model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer']);collector.restore(payload['collector'])
            ppo_rng.set_state(payload['ppo_rng']);torch.set_rng_state(payload['torch_rng']);random.setstate(payload['python_rng'])
        elif (directory/'resolved_config.json').exists():raise FileExistsError('run already exists')
        atomic_json(directory/'resolved_config.json',config)
        def save(reason):
            nonlocal parent
            parent=store.save(dict(config=config,config_sha256=digest(config),sources=source,model=model.state_dict(),optimizer=optimizer.state_dict(),
                collector=collector.state_dict(),global_step=collector.global_step,reason=reason,ppo_rng=ppo_rng.get_state(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                normalizer=None,scaler=None,scheduler=None,log_offsets={p.name:p.stat().st_size for p in directory.glob('*.jsonl')}),parent)
            if on_checkpoint is not None:on_checkpoint(collector.global_step,parent)
        if not resume:save('step0')
        budget=config['training']['steps'];cadence=config['training']['save_every'];next_save=(collector.global_step//cadence+1)*cadence
        if collector.global_step==budget and resume and on_checkpoint is not None:
            # A crash after the final immutable save can precede the scheduler's done record.
            on_checkpoint(collector.global_step,parent)
            collector.telemetry.close()
            atomic_json(directory/'status.json',dict(state='complete',global_step=collector.global_step,checkpoint=parent,research_improved=False))
            return parent
        if collector.global_step>=budget:raise ContractError('registered budget already complete')
        try:
            while collector.global_step<budget:
                previous_step=collector.global_step
                batch,collection=collector.collect(min(config['training']['rollout'],budget-collector.global_step))
                start=time.monotonic();metrics=update(model,optimizer,batch,config['training']['ppo'],ppo_rng)
                collector.version+=1
                append(directory/'training.jsonl',dict(global_step=collector.global_step,behavior_version=collector.version-1,collection=collection,ppo=metrics,update_seconds=time.monotonic()-start))
                stage_boundary=any(previous_step<s['until']<=collector.global_step for s in config['training']['opponent_schedule'])
                stopping=stop_requested is not None and stop_requested()
                if collector.global_step>=next_save or collector.global_step==budget or stage_boundary or stopping:
                    save('budget_end' if collector.global_step==budget else 'requested_stop' if stopping else 'stage_boundary' if stage_boundary else 'cadence')
                    next_save=(collector.global_step//cadence+1)*cadence
                atomic_json(directory/'progress.json',dict(global_step=collector.global_step,budget=budget,checkpoint=parent,
                    collection=collection,update_seconds=time.monotonic()-start))
                if stopping and collector.global_step<budget:
                    atomic_json(directory/'status.json',dict(state='paused',global_step=collector.global_step,checkpoint=parent))
                    return parent
            atomic_json(directory/'status.json',dict(state='complete',global_step=collector.global_step,checkpoint=parent,research_improved=False))
        except BaseException as exc:
            append(directory/'attempts.jsonl',dict(status='failed',exception=type(exc).__name__,detail=str(exc),global_step=collector.global_step))
            # Do not save a half-update model under a completed-boundary resume contract.
            atomic_json(directory/'status.json',dict(state='failed',last_valid_checkpoint=parent,detail=str(exc)))
            raise
        finally:collector.telemetry.close()
    return parent
