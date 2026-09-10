"""Atomic checkpoints, provenance, and self-contained two-file hybrid exports."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import zlib

import torch

from .mappo_v6 import SCHEMA_VERSION, GUARDRAIL_MODE, V6Model, model_from_payload
from .model_contract import SubmissionPolicy, checkpoint_payload, save_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_v6_checkpoint(path, model, optimizer, *, training, source):
    config = dict(model.actor_model.model_config)
    config.pop("n_actions", None)
    with torch.random.fork_rng(devices=[]):
        facade = SubmissionPolicy(**config)
    facade.actor_critic.load_state_dict(cpu_tree(model.actor_model.state_dict()))
    payload = checkpoint_payload(facade, global_step=training["global_step"],
                                 training_seed=training["config"]["seed"], source=source)
    payload.update(v6_schema=SCHEMA_VERSION, v6_model_state=cpu_tree(model.state_dict()),
                   mappo_v6_training=cpu_tree(training),
                   mappo_optimizer_state=cpu_tree(optimizer.state_dict()))
    payload["inference_guardrail"] = dict(version="scripted_counter_v1", mode=GUARDRAIL_MODE,
        roles=["worker"]*3+["guard"]*2, chase_radius_cells=48, max_overrides_per_step=1,
        context_version="actual_planner_history_mirrored_v6")
    payload["action_contract"]["executed_distribution"] = "joint_keep_or_slot_correction_41"
    payload["action_contract"]["inference"] = "deterministic_planner_joint_residual_v6"
    payload["torch_rng_state"] = torch.get_rng_state()
    if next(model.parameters()).device.type == "mps":
        payload["mps_rng_state"] = torch.mps.get_rng_state().cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    save_checkpoint(temporary, payload)
    temporary.replace(path)


def load_v6_checkpoint(path, device="cpu", *, restore_rng=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = model_from_payload(payload, device)
    training = payload.get("mappo_v6_training")
    if not training or training.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v6 training state missing; inference export cannot resume training")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=training["config"]["learning_rate"])
    optimizer.load_state_dict(payload["mappo_optimizer_state"])
    if restore_rng:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device == "mps" and "mps_rng_state" in payload:
            torch.mps.set_rng_state(payload["mps_rng_state"])
    return model, optimizer, payload


def source_manifest(root: Path):
    files = []
    for directory in ("blackout_rl", "scripts", "configs", "eval", "submission", "tests", "schemas"):
        files += [p for p in (root/directory).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".json")]
    files += [root/"requirements.lock", root/"game_spec.md"]
    hashes = {str(p.relative_to(root)):sha256(p) for p in sorted(files)}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"git_sha":revision, "files":hashes,
            "source_sha256":hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def archive_sources(root, destination, manifest):
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["files"]:
            archive.write(root/name, name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))


EXPORT_MODULES = ("mappo_v6", "ippo_model", "representation", "action_distribution", "batching",
                  "observation", "navigation", "semantic_map", "scripted_fsm", "team_state",
                  "strategy", "coordination")


def export_v6(checkpoint, destination, *, source_root=None):
    """Bundle audited source in policy.py; no Unity, project installation or I/O at act().

    Stateful planner and canonical-row requirements remain explicit. Packaging
    compatibility does not assert permission under an external competition rule.
    """
    root = Path(source_root or Path(__file__).resolve().parents[1])
    destination = Path(destination)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = model_from_payload(payload)
    sources = {name:(root/"blackout_rl"/(name+".py")).read_text() for name in EXPORT_MODULES}
    packed = base64.b64encode(zlib.compress(json.dumps(sources).encode())).decode()
    package = "_blackout_v6_"+hashlib.sha256(packed.encode()).hexdigest()[:12]
    config = dict(payload["model_config"])
    config.pop("n_actions", None)
    code = '''"""Standalone v6 hybrid policy. Requires torch, numpy; canonical five-row batches.
Planner memory resets on increasing time_left or explicit reset().
"""
import base64, importlib, importlib.abc, importlib.util, json, sys, types, zlib
import numpy as np
import torch
from torch import nn
_PACKAGE = PACKAGE_LITERAL
_SOURCES = json.loads(zlib.decompress(base64.b64decode(PACKED_LITERAL)))
class _Loader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_PACKAGE+".") and fullname.split(".")[-1] in _SOURCES:
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec): return None
    def exec_module(self, module):
        exec(compile(_SOURCES[module.__name__.split(".")[-1]], module.__name__, "exec"), module.__dict__)
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = []
    sys.modules[_PACKAGE] = package
    sys.meta_path.insert(0, _Loader())
_core = importlib.import_module(_PACKAGE+".mappo_v6")
_actor = importlib.import_module(_PACKAGE+".ippo_model")
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _core.V6Model(_actor.IPPOActorCritic(**CONFIG_LITERAL))
        self._policies = {}
    def reset(self):
        self._policies.clear()
    def forward(self, vector, graphic):
        if vector.shape != (5,96) or graphic.shape != (5,11,96,96):
            raise ValueError("canonical vector (5,96), graphic (5,11,96,96) required")
        if vector.dtype != torch.float32 or graphic.dtype != torch.float32:
            raise TypeError("float32 inputs required")
        if vector.device != graphic.device or vector.device != next(self.parameters()).device:
            raise ValueError("model and inputs must share device")
        if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(graphic).all()):
            raise ValueError("finite inputs required")
        team = 0 if float(vector[0,2]) > 0 else 1
        names = tuple(f"unit_{team*5+i}" for i in range(5))
        obs = {a:{"vector":vector[i].detach().cpu().numpy(),
                  "graphic":graphic[i].detach().cpu().numpy().transpose(1,2,0)} for i,a in enumerate(names)}
        if team not in self._policies:
            self._policies[team] = _core.V6Policy(self.model, team=team)
        actions = self._policies[team].act(obs, names)
        return torch.as_tensor(np.stack([actions[a] for a in names]), device=vector.device)
    def act(self, obs):
        names = sorted(obs, key=lambda name:int(name.split("_")[-1]))
        device = next(self.parameters()).device
        vector = torch.as_tensor(np.stack([obs[a]["vector"] for a in names]), device=device)
        graphic = torch.as_tensor(np.stack([obs[a]["graphic"].transpose(2,0,1) for a in names]), device=device)
        with torch.inference_mode(): actions = self(vector, graphic).cpu().numpy()
        return {a:actions[i] for i,a in enumerate(names)}
def load_policy(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = Policy().to(device)
    model.load_state_dict(payload["policy_state"], strict=True)
    return model.eval()
'''
    code = code.replace("PACKAGE_LITERAL", repr(package)).replace("PACKED_LITERAL", repr(packed)).replace("CONFIG_LITERAL", repr(config))
    if destination.exists() and any(destination.iterdir()):
        # Allow finishing a previously completed export after interruption, but
        # never replace unrelated files or a different policy at this location.
        if {p.name for p in destination.iterdir()} != {"policy.py", "checkpoint.pt"}:
            raise FileExistsError("export destination must be empty or contain this exact completed export")
        previous = torch.load(destination/"checkpoint.pt", map_location="cpu", weights_only=True)
        expected = {"model."+k:v for k,v in model.state_dict().items()}
        if ((destination/"policy.py").read_text() != code
            or previous.get("source_checkpoint_sha256") != sha256(checkpoint)
            or set(previous.get("policy_state", {})) != set(expected)
            or any(not torch.equal(previous["policy_state"][k], v) for k,v in expected.items())):
            raise FileExistsError("export destination contains different policy bytes or weights")
        return {"policy_sha256":sha256(destination/"policy.py"), "checkpoint_sha256":sha256(destination/"checkpoint.pt"),
                "source_checkpoint_sha256":sha256(checkpoint), "standalone_files":["policy.py", "checkpoint.pt"],
                "official_submission_approved":False}
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"policy.py").write_text(code, encoding="utf-8")
    torch.save({"policy_state":{"model."+k:v for k,v in model.state_dict().items()},
                "schema_version":"blackout.v6.standalone.v1", "source_checkpoint_sha256":sha256(checkpoint),
                "requirements":{"canonical_batch_order":True, "stateful_planner":True,
                                "reset":"explicit reset or increasing time_left", "dependencies":["torch", "numpy"]}},
               destination/"checkpoint.pt")
    return {"policy_sha256":sha256(destination/"policy.py"), "checkpoint_sha256":sha256(destination/"checkpoint.pt"),
            "source_checkpoint_sha256":sha256(checkpoint), "standalone_files":["policy.py", "checkpoint.pt"],
            "official_submission_approved":False}
