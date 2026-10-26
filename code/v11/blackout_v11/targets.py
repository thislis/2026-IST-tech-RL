"""Frozen predecision critic targets with variable-duration GAE."""
from pathlib import Path
import time
import numpy as np
import torch
from .critic import Critic
from .io import atomic_json, file_hash, state_hash
from .rewards import gae_episode
from .trajectory import Episode


def commit_targets(paths, snapshot, expected_hash, cfg, device, progress=lambda **_: None, stop=lambda: False):
    if file_hash(snapshot) != expected_hash:
        raise ValueError("behavior snapshot changed")
    payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    critic = Critic(payload.get("critic_arm", "C1")).to(device).eval()
    critic.load_state_dict(payload["critic_state"])
    critic_hash = state_hash(payload["critic_state"])
    started, count = time.monotonic(), 0
    with torch.inference_mode():
        for path in paths:
            ep = Episode(path, validated=True)
            if ep.meta["policy_sha256"] != expected_hash or not ep.meta.get("on_policy", True):
                raise ValueError("mixed behavior or teacher-forced PPO episode")
            values = np.empty(ep.length, np.float32)
            for start in range(0, ep.length, cfg.get("value_batch", 32)):
                if stop():
                    raise InterruptedError("target construction interrupted")
                ids = np.arange(start, min(ep.length, start+cfg.get("value_batch", 32)))
                data = ep.read(ids, device)
                values[ids] = critic(data["vectors"], data["graphic"], data["state"]).cpu().numpy()
                progress(phase="targets", target_events=count+int(ids[-1])+1)
            advantages, returns = gae_episode(ep.arrays["rewards"], values, ep.arrays["durations"],
                                              cfg.get("gamma", 1.), cfg.get("gae_lambda", .9975))
            for key, array in (("values", values), ("advantages", advantages), ("returns", returns)):
                np.save(Path(path)/(key+".npy"), array, allow_pickle=False)
            atomic_json(Path(path)/"committed.json", dict(ep.meta, critic_sha256=critic_hash,
                                                        behavior_sha256=expected_hash, targets_complete=True))
            count += ep.length
    return {"events": count, "seconds": time.monotonic()-started, "critic_sha256": critic_hash,
            "device": str(device), "value_batch": cfg.get("value_batch", 32)}
