"""Versioned JSONL trajectory recording for scripted policies."""

from __future__ import annotations

import base64
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .observation import parse_graphic, parse_vector
from .team_state import TeamSnapshot


TRAJECTORY_SCHEMA_VERSION = "blackout.scripted_trajectory.v1"
GRAPHIC_ENCODING = "zlib+base64+uint8-semantic-id"


def encode_graphic(graphic: np.ndarray) -> dict[str, Any]:
    channels = parse_graphic(graphic).channels
    ids = np.argmax(channels, axis=-1).astype(np.uint8)
    compressed = zlib.compress(ids.tobytes(order="C"), level=9)
    return {
        "encoding": GRAPHIC_ENCODING,
        "shape": list(ids.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def decode_graphic(payload: Mapping[str, Any]) -> np.ndarray:
    if payload.get("encoding") != GRAPHIC_ENCODING:
        raise ValueError(f"unsupported graphic encoding: {payload.get('encoding')}")
    shape = tuple(int(value) for value in payload["shape"])
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError(f"invalid graphic shape: {shape}")
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    ids = np.frombuffer(raw, dtype=np.uint8)
    if ids.size != shape[0] * shape[1]:
        raise ValueError("decoded graphic byte count does not match shape")
    return ids.reshape(shape).copy()


class ScriptedTrajectoryRecorder:
    """Stream team trajectories as compact, independently parseable JSON lines."""

    def __init__(self, path: str | Path, *, metadata: Mapping[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self.records = 0
        self.closed = False
        self._write(
            {
                "record_type": "header",
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": dict(metadata),
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("trajectory recorder is closed")
        self._stream.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    def record_step(
        self,
        *,
        snapshot: TeamSnapshot,
        observations: Mapping[str, Mapping[str, np.ndarray]],
        actions: Mapping[str, np.ndarray],
        rewards: Mapping[str, float],
        score: Mapping[str, float | int],
        events: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        agents = tuple(unit.agent for unit in snapshot.units)
        for label, mapping in (
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
        ):
            missing = [agent for agent in agents if agent not in mapping]
            if missing:
                raise KeyError(f"trajectory {label} missing agents: {missing}")
        first_graphic = observations[agents[0]]["graphic"]
        vectors = {}
        action_payload = {}
        for agent in agents:
            vector = observations[agent]["vector"]
            parse_vector(vector)
            action = np.asarray(actions[agent], dtype=np.float32)
            if action.shape != (2,) or not np.all(np.isfinite(action)):
                raise ValueError(f"invalid action for {agent}: {action}")
            vectors[agent] = vector.tolist()
            action_payload[agent] = action.tolist()
        self._write(
            {
                "record_type": "step",
                "step": snapshot.step,
                "team": snapshot.team,
                "observation": {
                    "vectors": vectors,
                    "team_graphic": encode_graphic(first_graphic),
                },
                "actions": action_payload,
                "unit_state": {
                    unit.agent: {
                        "slot_id": unit.slot_id,
                        "position_normalized": list(unit.position_normalized),
                        "cell": [unit.cell.x, unit.cell.y],
                        "holding_item_id": unit.holding_item_id,
                        "class_id": unit.class_id,
                        "role": unit.role.value,
                        "target": (
                            None if unit.target is None else [unit.target.x, unit.target.y]
                        ),
                    }
                    for unit in snapshot.units
                },
                "rewards": {agent: float(rewards[agent]) for agent in agents},
                "score": dict(score),
                "events": list(events),
            }
        )
        self.records += 1

    def close(self, *, summary: Mapping[str, Any] | None = None) -> None:
        if self.closed:
            return
        self._write(
            {
                "record_type": "footer",
                "records": self.records,
                "closed_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": dict(summary or {}),
            }
        )
        self._stream.close()
        self.closed = True

    def __enter__(self) -> "ScriptedTrajectoryRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(summary={"status": "ok" if exc_type is None else "error"})
