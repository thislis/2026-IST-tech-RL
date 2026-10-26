"""Frozen-teacher episodes, optional DAgger relabeling, and recurrent BC warm-up."""
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .io import ROOT, append, atomic_json, atomic_torch, cpu_state, digest, file_hash, read_json
from .opponents import training_catalog
from .policy import MyPolicy
from .runtime import run_tasks
from .trajectory import Episode
from .recurrent_ppo import _hidden, _recurrent, _slice, sequence_batch


def prepare_demonstrations(directory, cfg, registration_sha, stop=lambda: False, progress=lambda **_: None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["registration_sha256"] != registration_sha:
            raise ValueError("teacher dataset registration changed")
        if manifest.get("complete"):
            return manifest_path, manifest
    else:
        manifest = {"registration_sha256": registration_sha, "target_env_steps": cfg.get("teacher_steps", 262144),
                    "env_steps": 0, "attempted_steps": 0, "episodes": [], "complete": False,
                    "label_source": "frozen historical action projected onto legal observation-derived candidates",
                    "teacher_forcing_excluded_from_ppo": True, "format": "v11-events-v1"}
    root = ROOT/"artifacts/data/v11"/registration_sha[:20]
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root/"label_policy.pt"
    if not snapshot.exists():
        with torch.random.fork_rng():
            torch.manual_seed(20261024)
            policy = MyPolicy().eval()
            policy.configure("F", "greedy")
        atomic_torch(snapshot, {"policy_state": cpu_state(policy), "arm": "F", "decoder": "greedy"})
    snapshot_hash = file_hash(snapshot)
    pool = training_catalog()
    splits = read_json(Path(__file__).parents[1]/"configs/splits.json")
    bounds = splits["teacher"]
    teachers = pool["planner"]
    opponents = pool["scripted"]+pool["planner"]
    def ingest(rows):
        recorded = set(manifest.get("recorded_attempts", []))
        invalid = []
        for row in rows:
            if row["id"] in recorded:
                continue
            recorded.add(row["id"])
            append(directory/"episodes.jsonl", row)
            manifest["attempted_steps"] += int(row["attempt_steps"])
            path = Path(row["directory"])/"trajectory"
            if not row["valid"] or row.get("interrupted"):
                invalid.append(row)
                if path.is_dir():
                    shutil.rmtree(path)
                continue
            meta = row["result"]
            split = "validation" if int(digest(meta["episode_id"])[:8], 16) % 5 == 0 else "train"
            manifest["episodes"].append({"path": str(path), "split": split, "env_steps": meta["env_steps"],
                "events": meta["length"], "teacher": meta["teacher"], "episode_id": meta["episode_id"],
                "initial_map_hash": meta["initial_map_hash"], "side": meta["side"], "behavior": meta["behavior"]})
            manifest["env_steps"] += int(meta["env_steps"])
        manifest["recorded_attempts"] = sorted(recorded)
        manifest["complete"] = manifest["env_steps"] >= manifest["target_env_steps"]
        manifest["overrun"] = max(0, manifest["env_steps"]-manifest["target_env_steps"])
        manifest["data_directory"] = str(root)
        manifest["pending_wave"] = None
        atomic_json(manifest_path, manifest)
        return invalid
    if manifest.get("pending_wave"):
        recovered = []
        for task_path in Path(manifest["pending_wave"]).glob("*/task.json"):
            path = task_path.parent
            if (path/"attempt.json").exists():
                recovered.append(read_json(path/"attempt.json"))
            else:
                heartbeat = read_json(path/"heartbeat.json") if (path/"heartbeat.json").exists() else {}
                recovered.append(dict(id=path.name, directory=str(path), valid=False, interrupted=True,
                                      attempt_steps=heartbeat.get("steps", 0), result=None,
                                      error="uncommitted teacher episode discarded during resume"))
        ingest(recovered)
    while manifest["env_steps"] < manifest["target_env_steps"] and not stop():
        offset = int(manifest.get("attempts", 0))
        wave = root/f"wave-{offset:07d}-{time.time_ns()}"
        tasks = []
        for i in range(int(cfg["workers"])):
            serial = offset+i
            teacher = teachers[serial % len(teachers)]
            opponent = opponents[(serial//2) % len(opponents)]
            tasks.append({"id": f"teacher-{serial:07d}", "kind": "collect", "config": cfg, "arm": "F",
                "teacher": True, "teacher_spec": teacher, "teacher_mix": 1., "deterministic": True,
                "side": serial % 2, "opponent": opponent, "family": opponent.get("family", "scripted"),
                "requested_seed": bounds[0]+serial % (bounds[1]-bounds[0]+1), "action_seed": 970000+serial*17,
                "policy_path": str(snapshot), "policy_sha256": snapshot_hash, "audit_windows": True})
        manifest["attempts"] = offset+len(tasks)
        manifest["pending_wave"] = str(wave)
        atomic_json(manifest_path, manifest)
        results = run_tasks(tasks, wave, cfg, stop,
                lambda **kw: progress(phase="teacher_collection", env_steps=manifest["env_steps"],
                                      budget=manifest["target_env_steps"], **{k: v for k, v in kw.items() if k != "phase"}))
        invalid = ingest(results)
        if invalid and not stop():
            raise RuntimeError("teacher collection had invalid episodes; inspect episode ledger")
    if stop() and not manifest["complete"]:
        raise InterruptedError("teacher collection stopped before registered budget completed")
    return manifest_path, manifest


def initialize_from_demonstrations(policy, manifest, cfg, device, progress=lambda **_: None, stop=lambda: False):
    if isinstance(manifest, (tuple, list)):
        manifest = manifest[0]
    data = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    if not data.get("complete"):
        raise ValueError("teacher initialization requires complete registered dataset")
    episodes = [Episode(e["path"], validated=True) for e in data["episodes"] if e["split"] == "train"]
    if not episodes:
        raise ValueError("teacher episode split contains no training data")
    requested = dict(arm=policy.arm, decoder=policy.decoder, max_interventions=policy.max_interventions,
                     goal_period=policy.goal_period, residual_period=policy.residual_period,
                     use_memory=policy.use_memory)
    # A common F warm-up keeps G/F encoder/GRU initialization exactly matched.
    # Direct-action supervision is detached at features, so D's extra head does
    # not change that common representation or receive privileged inputs.
    policy.configure(**dict(requested, arm="F"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.get("ppo", {}).get("actor_lr", 1e-4))
    burn_in = cfg.get("ppo", {}).get("burn_in", 32)
    length = cfg.get("ppo", {}).get("sequence_length", 64)
    descriptors = [(ep, max(0, start-burn_in), start, min(ep.length, start+length))
                   for ep in episodes for start in range(0, ep.length, length)]
    rng = np.random.default_rng(771)
    losses, events = [], 0
    started = time.monotonic()
    for epoch in range(cfg.get("teacher_epochs", 2)):
        groups = {}
        for index in rng.permutation(len(descriptors)):
            desc = descriptors[int(index)]
            groups.setdefault(desc[2]-desc[1], []).append(desc)
        batch_size = int(cfg.get("resolved_minibatch_sequences", cfg.get("teacher_batch", 16)))
        for group in groups.values():
            for start in range(0, len(group), batch_size):
                if stop():
                    raise InterruptedError("teacher warm-up interrupted before initial checkpoint")
                batch, burn = sequence_batch(group[start:start+batch_size], device)
                hidden = None
                for t in range(burn):
                    frame = _slice(batch, t)
                    with torch.no_grad():
                        out = policy.evaluate_decision(frame["vectors"], frame["graphic"],
                                _recurrent(frame["state"], hidden), frame["teacher_choices"])
                    hidden = tuple(x.detach() for x in _hidden(out))
                optimizer.zero_grad(set_to_none=True)
                terms, masks = [], []
                for t in range(burn, batch["valid"].shape[1]):
                    frame = _slice(batch, t)
                    if not frame["teacher_choices"]:
                        raise ValueError("missing teacher labels")
                    state = _recurrent(frame["state"], hidden)
                    out = policy.evaluate_decision(frame["vectors"], frame["graphic"], state, frame["teacher_choices"])
                    hidden = _hidden(out)
                    mask = frame["valid"].float()
                    term = -out["log_prob"]
                    if requested["arm"] == "D":
                        direct_logs = policy.direct_head(out["features"].detach()).log_softmax(-1)
                        chosen = frame["teacher_choices"]["direct"]
                        direct_lp = direct_logs.gather(-1, chosen[..., None]).squeeze(-1)
                        term = term-(direct_lp*frame["state"]["residual_event"].float()).sum(-1)
                    terms.append(term*mask)
                    masks.append(mask)
                denom = torch.stack(masks).sum().clamp_min(1)
                loss = cfg.get("bc_coefficient", .1)*torch.stack(terms).sum()/denom
                if not torch.isfinite(loss):
                    raise ArithmeticError("nonfinite teacher loss")
                loss.backward()
                common_parameters = [p for name, p in policy.named_parameters() if not name.startswith("direct_head.")]
                norm = torch.nn.utils.clip_grad_norm_(common_parameters, cfg.get("ppo", {}).get("max_grad_norm", .5))
                if requested["arm"] == "D":
                    direct_norm = torch.nn.utils.clip_grad_norm_(policy.direct_head.parameters(),
                                                               cfg.get("ppo", {}).get("max_grad_norm", .5))
                    if not torch.isfinite(direct_norm):
                        raise ArithmeticError("nonfinite direct teacher gradient")
                if not torch.isfinite(norm):
                    raise ArithmeticError("nonfinite teacher gradient")
                optimizer.step()
                events += int(denom.detach().cpu())
                losses.append(float(loss.detach().cpu()))
                progress(phase="teacher_warmup", epoch=epoch, teacher_events=events, loss=losses[-1])
    policy.configure(**requested)
    return {"training_events": events, "training_episodes": len(episodes), "epochs": cfg.get("teacher_epochs", 2),
            "mean_loss": float(np.mean(losses)), "seconds": time.monotonic()-started,
            "validation_episodes": sum(e["split"] == "validation" for e in data["episodes"]),
            "dataset_env_steps": data["env_steps"], "label_source": data["label_source"],
            "common_encoder_initialization": "F teacher warm-up; D direct-head gradient detached from encoder",
            "performance_certified": False, "teacher_forcing_excluded_from_ppo": True}
