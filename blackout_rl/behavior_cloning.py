"""Offline behavior cloning from versioned scripted trajectory JSONL files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .action_distribution import ACTION_DIRECTIONS
from .batching import team_agents
from .ippo_model import IPPOActorCritic
from .observation import GRAPHIC_CHANNEL_NAMES, VECTOR_SIZE, parse_vector
from .trajectory import TRAJECTORY_SCHEMA_VERSION, decode_graphic


def action_vector_to_index(action: Sequence[float], *, zero_tolerance: float = 1e-6) -> int:
    """Quantize a continuous scripted action to NoOp or its nearest unit direction."""
    vector = torch.as_tensor(action, dtype=torch.float32)
    if vector.shape != (2,) or not bool(torch.all(torch.isfinite(vector))):
        raise ValueError("action must be a finite 2-vector")
    magnitude = torch.linalg.vector_norm(vector)
    if float(magnitude) <= zero_tolerance:
        return 0
    normalized = vector / magnitude
    candidates = ACTION_DIRECTIONS[1:]
    candidates = candidates / torch.linalg.vector_norm(candidates, dim=-1, keepdim=True)
    return int(torch.argmin(torch.sum((candidates - normalized) ** 2, dim=-1))) + 1


@dataclass(frozen=True)
class DemonstrationSample:
    vector: np.ndarray
    graphic_ids: np.ndarray
    slot_id: int
    action_index: int
    source: str
    step: int


class ScriptedTrajectoryDataset(Dataset):
    """Agent-row dataset decoded lazily from compact scripted trajectory records."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("at least one trajectory path is required")
        self.samples: list[DemonstrationSample] = []
        self.trajectory_metadata: list[dict] = []
        for path in paths:
            self._load_path(Path(path))
        if not self.samples:
            raise ValueError("trajectory dataset contains no step samples")

    def _load_path(self, path: Path) -> None:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) < 3 or records[0].get("record_type") != "header":
            raise ValueError(f"trajectory {path} has no valid header")
        if records[0].get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema in {path}")
        if records[-1].get("record_type") != "footer":
            raise ValueError(f"trajectory {path} has no footer")
        step_records = [record for record in records[1:-1] if record.get("record_type") == "step"]
        if int(records[-1].get("records", -1)) != len(step_records):
            raise ValueError(f"trajectory footer count mismatch in {path}")
        self.trajectory_metadata.append(
            {"path": str(path), "header": records[0], "records": len(step_records)}
        )
        for record in step_records:
            team = int(record["team"])
            agents = team_agents(team)
            graphic_ids = decode_graphic(record["observation"]["team_graphic"])
            if np.any(graphic_ids >= len(GRAPHIC_CHANNEL_NAMES)):
                raise ValueError(f"trajectory {path} contains an unknown graphic channel")
            for slot_id, agent in enumerate(agents):
                vector = np.asarray(record["observation"]["vectors"][agent], dtype=np.float32)
                parse_vector(vector)
                stored_slot = int(record["unit_state"][agent]["slot_id"])
                if stored_slot != slot_id:
                    raise ValueError(f"trajectory slot mismatch for {agent} in {path}")
                self.samples.append(
                    DemonstrationSample(
                        vector=vector,
                        graphic_ids=graphic_ids,
                        slot_id=slot_id,
                        action_index=action_vector_to_index(record["actions"][agent]),
                        source=str(path),
                        step=int(record["step"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.samples[index]
        graphic = F.one_hot(
            torch.from_numpy(sample.graphic_ids.astype(np.int64, copy=False)),
            num_classes=len(GRAPHIC_CHANNEL_NAMES),
        ).permute(2, 0, 1).to(torch.float32)
        return (
            torch.from_numpy(sample.vector),
            graphic,
            torch.tensor(sample.slot_id, dtype=torch.int64),
            torch.tensor(sample.action_index, dtype=torch.int64),
        )


@dataclass(frozen=True)
class BCMetrics:
    loss: float
    accuracy: float
    samples: int


@dataclass(frozen=True)
class BCExperimentResult:
    seed: int
    epochs: int
    batch_size: int
    train_samples: int
    held_out_samples: int
    train_sources: list[str]
    held_out_sources: list[str]
    demonstration_sample_budget: int
    downstream_environment_step_budget: dict[str, int]
    scratch: BCMetrics
    bc_pretrained: BCMetrics

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_behavior_cloning(
    model: IPPOActorCritic,
    dataset: Dataset,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> BCMetrics:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for vector, graphic, slot_id, action_index in loader:
            vector = vector.to(device)
            graphic = graphic.to(device)
            slot_id = slot_id.to(device)
            action_index = action_index.to(device)
            logits = model(vector, graphic, slot_id).action_logits
            loss_sum += float(F.cross_entropy(logits, action_index, reduction="sum"))
            correct += int((torch.argmax(logits, dim=-1) == action_index).sum())
            samples += int(action_index.numel())
    return BCMetrics(loss=loss_sum / samples, accuracy=correct / samples, samples=samples)


def behavior_clone(
    model: IPPOActorCritic,
    dataset: Dataset,
    *,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> int:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or max_grad_norm <= 0.0:
        raise ValueError("BC hyperparameters must be positive")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    samples_seen = 0
    for _ in range(epochs):
        for vector, graphic, slot_id, action_index in loader:
            logits = model(
                vector.to(device), graphic.to(device), slot_id.to(device)
            ).action_logits
            loss = F.cross_entropy(logits, action_index.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError("non-finite BC gradient norm")
            optimizer.step()
            samples_seen += int(action_index.numel())
    return samples_seen


def run_bc_warm_start_experiment(
    model: IPPOActorCritic,
    train_dataset: Dataset,
    held_out_dataset: Dataset,
    *,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    downstream_environment_steps: int = 0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[IPPOActorCritic, BCExperimentResult]:
    """Compare identical scratch/BC initializations at equal environment steps."""
    if downstream_environment_steps < 0:
        raise ValueError("downstream_environment_steps must be non-negative")
    scratch = type(model)(
        **{key: value for key, value in model.model_config.items() if key != "n_actions"}
    )
    scratch.load_state_dict(model.state_dict())
    bc_model = type(model)(
        **{key: value for key, value in model.model_config.items() if key != "n_actions"}
    )
    bc_model.load_state_dict(model.state_dict())
    scratch_metrics = evaluate_behavior_cloning(
        scratch, held_out_dataset, batch_size=batch_size, device=device
    )
    samples_seen = behavior_clone(
        bc_model,
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    bc_metrics = evaluate_behavior_cloning(
        bc_model, held_out_dataset, batch_size=batch_size, device=device
    )
    train_sources = [
        str(metadata["path"])
        for metadata in getattr(train_dataset, "trajectory_metadata", [])
    ]
    held_out_sources = [
        str(metadata["path"])
        for metadata in getattr(held_out_dataset, "trajectory_metadata", [])
    ]
    result = BCExperimentResult(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        train_samples=len(train_dataset),
        held_out_samples=len(held_out_dataset),
        train_sources=train_sources,
        held_out_sources=held_out_sources,
        demonstration_sample_budget=samples_seen,
        downstream_environment_step_budget={
            "scratch": downstream_environment_steps,
            "bc_pretrained": downstream_environment_steps,
        },
        scratch=scratch_metrics,
        bc_pretrained=bc_metrics,
    )
    return bc_model, result


def write_bc_experiment(path: str | Path, result: BCExperimentResult) -> None:
    payload = {
        "schema_version": "blackout.bc_experiment.v1",
        **result.to_dict(),
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
