"""P2: frozen-policy episode holdout; disposable critics fit Monte Carlo returns."""
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json, atomic_torch, file_hash, read_json
from .policy import MyPolicy
from .value import Critic
from .trajectory import Episode
from .runtime import run_tasks
from .schedule import task_spec


def run_diagnostic(directory, cfg, device, stop, progress):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    report_path = directory/'result.json'
    if report_path.exists(): return read_json(report_path)
    snapshot = directory/'behavior.pt'
    with torch.random.fork_rng():
        torch.manual_seed(902); actor = MyPolicy()
    if not snapshot.exists(): atomic_torch(snapshot, {'policy_state': actor.state_dict()})
    checksum = file_hash(snapshot); tasks = []
    for i in range(cfg['diagnostic_games']):
        key = f'diagnostic-{i:03d}'
        if (directory/'episodes'/key/'trajectory/validated.json').exists(): continue
        tasks.append(dict(id=key, kind='collect', config=cfg, **task_spec(902,i),
                          policy_path=str(snapshot), policy_sha256=checksum, audit_windows=True))
    # Each retry gets a new attempt directory; validated episodes are copied atomically by rename.
    import time, shutil
    attempts = directory/'attempts'/str(time.time_ns())
    rows = run_tasks(tasks, attempts, cfg, stop, lambda **kw: progress(phase='value_diagnostic_collect', **kw)) if tasks else []
    for row in rows:
        if row['valid']:
            dest = directory/'episodes'/row['id']; dest.parent.mkdir(exist_ok=True)
            shutil.move(row['directory'], dest)
    if stop(): raise InterruptedError('diagnostic stopped')
    if any(not r['valid'] for r in rows): raise RuntimeError('invalid P2 episode; diagnostic cannot pass')
    episodes = [Episode(directory/'episodes'/f'diagnostic-{i:03d}'/'trajectory', validated=True)
                for i in range(cfg['diagnostic_games'])]
    split = cfg['diagnostic_fit_games']; selections = []; targets = []
    for ep in episodes:
        selections.append(np.unique(np.linspace(0, ep.length-1, min(ep.length,cfg['diagnostic_samples_per_episode'])).astype(int)))
        targets.append(np.cumsum(np.asarray(ep.arrays['rewards'],dtype=np.float64)[::-1])[::-1].copy().astype(np.float32))
    results = {}
    for arm in ('C0','C1','C1-blind'):
        with torch.random.fork_rng():
            torch.manual_seed(1902); critic = Critic(arm).to(device)
        fit_targets = torch.from_numpy(np.concatenate([targets[i][selections[i]] for i in range(split)])).to(device)
        critic.update_scale(fit_targets)
        opt = torch.optim.Adam(critic.parameters(), lr=cfg['ppo']['critic_lr'])
        for epoch in range(cfg['diagnostic_epochs']):
            for i, ep in enumerate(episodes[:split]):
                for begin in range(0,len(selections[i]),cfg['value_batch']):
                    if stop(): raise InterruptedError('diagnostic fit stopped')
                    ids = selections[i][begin:begin+cfg['value_batch']]
                    v = torch.from_numpy(np.array(ep.arrays['vectors'][ids])).to(device)
                    g = torch.from_numpy(np.ascontiguousarray(ep.graphic(ids))).to(device)
                    y = torch.from_numpy(targets[i][ids]).to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.smooth_l1_loss(critic.normalized(v,g),(y-critic.mean)/critic.scale)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(),.5,error_if_nonfinite=True); opt.step()
                    progress(phase='value_diagnostic_fit', arm=arm, epoch=epoch, episode=i)
        predictions = []; truth = []
        with torch.inference_mode():
            for i in range(split,len(episodes)):
                ep=episodes[i]
                for begin in range(0,len(selections[i]),cfg['value_batch']):
                    ids=selections[i][begin:begin+cfg['value_batch']]
                    v=torch.from_numpy(np.array(ep.arrays['vectors'][ids])).to(device)
                    g=torch.from_numpy(np.ascontiguousarray(ep.graphic(ids))).to(device)
                    predictions.extend(critic(v,g).cpu().tolist()); truth.extend(targets[i][ids].tolist())
        y=np.asarray(truth); error=np.asarray(predictions)-y
        results[arm]=dict(mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))),
                         explained_variance=None if np.var(y)<1e-12 else float(1-np.var(error)/np.var(y)))
    result=dict(complete=True, fit_episodes=split,holdout_episodes=len(episodes)-split,
                episode_ids=[ep.meta['episode_id'] for ep in episodes],results=results,
                target='Monte Carlo shaped return; episode-level split', weights_reused_for_training=False,
                env_steps=sum(ep.length for ep in episodes))
    atomic_json(report_path,result)
    # Release mmaps before staging cleanup; retain the diagnostic report and small tails.
    del episodes
    from .training import discard_staging
    discard_staging(directory/'episodes')
    return result
