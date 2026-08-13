"""Reusable device and stopping rules for long-running IPPO experiments."""

from __future__ import annotations

from dataclasses import dataclass
import io
import zlib

import torch
from torch import nn
from torch.nn import functional as F

from .ippo_model import IPPOActorCritic
from .rollout import RolloutBatch, decode_rollout_graphic


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved torch device together with an audit-friendly explanation."""

    requested: str
    resolved: str
    reason: str


@dataclass(frozen=True)
class OnlineImitationMetrics:
    """Aggregate teacher-label metrics for one rollout."""

    loss: float
    accuracy: float
    samples: int
    gradient_norm: float


class TeacherReplayBuffer:
    """CPU ring buffer with uint8 semantic IDs instead of repeated one-hot maps."""

    def __init__(self, capacity: int = 30_000, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("teacher replay capacity must be positive")
        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)
        self.size = 0
        self.position = 0
        self.vector: torch.Tensor | None = None
        self.graphic_ids: torch.Tensor | None = None
        self.slot_id: torch.Tensor | None = None
        self.action_index: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.size

    def add(self, batch: RolloutBatch) -> None:
        if batch.teacher_action_index is None:
            raise ValueError("rollout batch has no teacher action labels")
        vector = batch.vector.detach().cpu()
        graphic = batch.graphic.detach().cpu()
        graphic_ids = (
            graphic.to(torch.uint8)
            if graphic.ndim == 3
            else torch.argmax(graphic, dim=1).to(torch.uint8)
        )
        slot_id = batch.slot_id.detach().cpu()
        action_index = batch.teacher_action_index.detach().cpu()
        valid = action_index >= 0
        vector = vector[valid]
        graphic_ids = graphic_ids[valid]
        slot_id = slot_id[valid]
        action_index = action_index[valid]
        if action_index.numel() == 0:
            return
        count = int(action_index.numel())
        if count > self.capacity:
            start = count - self.capacity
            vector = vector[start:]
            graphic_ids = graphic_ids[start:]
            slot_id = slot_id[start:]
            action_index = action_index[start:]
            count = self.capacity
        if self.vector is None:
            self.vector = torch.empty(
                (self.capacity, *vector.shape[1:]), dtype=vector.dtype
            )
            self.graphic_ids = torch.empty(
                (self.capacity, *graphic_ids.shape[1:]), dtype=torch.uint8
            )
            self.slot_id = torch.empty(self.capacity, dtype=torch.int64)
            self.action_index = torch.empty(self.capacity, dtype=torch.int64)
        first = min(count, self.capacity - self.position)
        second = count - first
        destination = slice(self.position, self.position + first)
        self.vector[destination].copy_(vector[:first])
        self.graphic_ids[destination].copy_(graphic_ids[:first])
        self.slot_id[destination].copy_(slot_id[:first])
        self.action_index[destination].copy_(action_index[:first])
        if second:
            destination = slice(0, second)
            self.vector[destination].copy_(vector[first:])
            self.graphic_ids[destination].copy_(graphic_ids[first:])
            self.slot_id[destination].copy_(slot_id[first:])
            self.action_index[destination].copy_(action_index[first:])
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def state_dict(self) -> dict[str, object]:
        """Return a compact, checkpoint-safe snapshot of the replay contents."""

        state: dict[str, object] = {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "generator_state": self.generator.get_state(),
        }
        if self.size == 0:
            return state
        assert self.vector is not None
        assert self.graphic_ids is not None
        assert self.slot_id is not None
        assert self.action_index is not None
        graphic_stream = io.BytesIO()
        torch.save(self.graphic_ids[: self.size], graphic_stream)
        state.update(
            {
                "vector": self.vector[: self.size].clone(),
                "graphic_ids_zlib": zlib.compress(graphic_stream.getvalue()),
                "slot_id": self.slot_id[: self.size].clone(),
                "action_index": self.action_index[: self.size].clone(),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a replay snapshot, rejecting incompatible capacities."""

        capacity = int(state["capacity"])
        size = int(state["size"])
        position = int(state["position"])
        if capacity > self.capacity:
            raise ValueError(
                f"teacher replay checkpoint capacity {capacity} exceeds runtime {self.capacity}"
            )
        if not 0 <= size <= capacity or not 0 <= position < capacity:
            raise ValueError("invalid teacher replay size or position")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("teacher replay generator state must be a tensor")
        self.generator.set_state(generator_state.detach().cpu())
        self.size = size
        self.position = size % self.capacity
        if size == 0:
            self.vector = None
            self.graphic_ids = None
            self.slot_id = None
            self.action_index = None
            return
        vector = state["vector"]
        slot_id = state["slot_id"]
        action_index = state["action_index"]
        graphic_bytes = zlib.decompress(state["graphic_ids_zlib"])
        graphic_ids = torch.load(
            io.BytesIO(graphic_bytes), map_location="cpu", weights_only=True
        )
        values = (vector, graphic_ids, slot_id, action_index)
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("teacher replay checkpoint values must be tensors")
        vector = vector.detach().cpu()
        graphic_ids = graphic_ids.detach().cpu()
        slot_id = slot_id.detach().cpu()
        action_index = action_index.detach().cpu()
        values = (vector, graphic_ids, slot_id, action_index)
        if not all(value.shape[0] == size for value in values):
            raise ValueError("teacher replay checkpoint tensor lengths differ")
        self.vector = torch.empty((self.capacity, *vector.shape[1:]), dtype=vector.dtype)
        self.graphic_ids = torch.empty(
            (self.capacity, *graphic_ids.shape[1:]), dtype=torch.uint8
        )
        self.slot_id = torch.empty(self.capacity, dtype=torch.int64)
        self.action_index = torch.empty(self.capacity, dtype=torch.int64)
        self.vector[:size].copy_(vector)
        self.graphic_ids[:size].copy_(graphic_ids)
        self.slot_id[:size].copy_(slot_id)
        self.action_index[:size].copy_(action_index)

    def sample(
        self, batch_size: int, *, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample an empty teacher replay buffer")
        if batch_size <= 0:
            raise ValueError("teacher replay batch size must be positive")
        assert self.vector is not None
        assert self.graphic_ids is not None
        assert self.slot_id is not None
        assert self.action_index is not None
        indices = torch.randint(
            self.size, (batch_size,), generator=self.generator, device="cpu"
        )
        vector = self.vector[indices].to(device)
        graphic = F.one_hot(
            self.graphic_ids[indices].to(torch.int64), num_classes=11
        ).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        return (
            vector,
            graphic,
            self.slot_id[indices].to(device),
            self.action_index[indices].to(device),
        )


def select_training_device(requested: str = "auto") -> DeviceSelection:
    """Prefer MPS in auto mode only when this PyTorch process can use it."""

    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    mps = getattr(torch.backends, "mps", None)
    mps_built = bool(mps and mps.is_built())
    mps_available = bool(mps and mps.is_available())
    if normalized == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but is unavailable "
                f"(built={mps_built}, available={mps_available})"
            )
        return DeviceSelection("mps", "mps", "explicitly requested and available")
    if normalized == "cpu":
        return DeviceSelection("cpu", "cpu", "explicitly requested")
    if mps_available:
        return DeviceSelection("auto", "mps", "MPS is available in this PyTorch process")
    return DeviceSelection(
        "auto",
        "cpu",
        f"MPS unavailable (built={mps_built}, available={mps_available})",
    )


def required_wins(episodes: int, threshold: float) -> int:
    """Return the smallest integer win count satisfying a strict win-rate target."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    # Avoid importing math for a single exact ceiling operation.
    raw = episodes * threshold
    rounded = int(raw)
    return rounded if raw == rounded else rounded + 1


def passed_win_rate(*, wins: int, episodes: int, threshold: float) -> bool:
    """Use wins/all episodes; draws do not count as wins."""

    if not 0 <= wins <= episodes:
        raise ValueError("wins must be in [0, episodes]")
    return wins >= required_wins(episodes, threshold)


def online_imitation_update(
    model: IPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    epochs: int = 1,
    minibatch_size: int = 128,
    loss_coef: float = 1.0,
    max_grad_norm: float = 0.5,
) -> OnlineImitationMetrics:
    """Fit the actor to scripted teacher decisions observed on current-policy states."""

    if batch.teacher_action_index is None:
        raise ValueError("rollout batch has no teacher action labels")
    if epochs <= 0 or minibatch_size <= 0 or loss_coef <= 0.0 or max_grad_norm <= 0.0:
        raise ValueError("online imitation hyperparameters must be positive")
    device = next(model.parameters()).device
    batch = batch.to(device)
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    gradient_sum = 0.0
    minibatches = 0
    for _ in range(epochs):
        for start in range(0, len(batch), minibatch_size):
            stop = min(start + minibatch_size, len(batch))
            logits = model(
                batch.vector[start:stop],
                decode_rollout_graphic(batch.graphic[start:stop]),
                batch.slot_id[start:stop],
            ).action_logits
            labels = batch.teacher_action_index[start:stop]
            unscaled_loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            (loss_coef * unscaled_loss).backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite online imitation gradient norm")
            optimizer.step()
            count = int(labels.numel())
            loss_sum += float(unscaled_loss.detach()) * count
            correct += int((torch.argmax(logits.detach(), dim=-1) == labels).sum())
            samples += count
            gradient_sum += float(gradient_norm.detach())
            minibatches += 1
    return OnlineImitationMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
        gradient_norm=gradient_sum / minibatches,
    )


def teacher_replay_update(
    model: IPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    replay: TeacherReplayBuffer,
    *,
    minibatches: int = 8,
    minibatch_size: int = 128,
    loss_coef: float = 1.0,
    max_grad_norm: float = 0.5,
    class_balance_power: float = 0.0,
) -> OnlineImitationMetrics:
    """Train from mixed historical teacher states to avoid phase forgetting."""

    if minibatches <= 0 or minibatch_size <= 0 or loss_coef <= 0.0 or max_grad_norm <= 0.0:
        raise ValueError("teacher replay hyperparameters must be positive")
    if not 0.0 <= class_balance_power <= 1.0:
        raise ValueError("class_balance_power must be in [0,1]")
    device = next(model.parameters()).device
    class_weights = None
    if class_balance_power:
        assert replay.action_index is not None
        counts = torch.bincount(
            replay.action_index[: len(replay)], minlength=model.model_config["n_actions"]
        ).to(torch.float32)
        if bool(torch.any(counts == 0)):
            raise ValueError("cannot class-balance a replay missing an action class")
        class_weights = counts.pow(-class_balance_power)
        class_weights = (class_weights / class_weights.mean()).to(device)
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    gradient_sum = 0.0
    for _ in range(minibatches):
        vector, graphic, slot_id, labels = replay.sample(minibatch_size, device=device)
        logits = model(vector, graphic, slot_id).action_logits
        unscaled_loss = F.cross_entropy(logits, labels, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        (loss_coef * unscaled_loss).backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite teacher replay gradient norm")
        optimizer.step()
        loss_sum += float(unscaled_loss.detach()) * minibatch_size
        correct += int((torch.argmax(logits.detach(), dim=-1) == labels).sum())
        samples += minibatch_size
        gradient_sum += float(gradient_norm.detach())
    return OnlineImitationMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
        gradient_norm=gradient_sum / minibatches,
    )
