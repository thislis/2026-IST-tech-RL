"""Durable stop provenance and rate-limited resource history across resumes."""
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import uuid


def append_jsonl(path, record):
    """Serialize concurrent CLI/supervisor writers and fsync every full record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if stream.seek(0, os.SEEK_END):
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                line = b"\n" + line
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_events(directory):
    path = Path(directory) / "stop_events.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        rows = []
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A pre-existing interrupted write must not hide later records.
                continue
        return rows


def stop_event(directory, *, event, origin, reason, resource, limits,
               session_id=None, operation=None, **details):
    row = dict(schema="blackout.v11.stop.v1", event=event, event_id=uuid.uuid4().hex,
               time=time.time(), pid=os.getpid(), session_id=session_id,
               operation=operation, origin=origin, reason=reason,
               resource=resource, limits=limits, **details)
    append_jsonl(Path(directory) / "stop_events.jsonl", row)
    return row


def manual_stop(directory, *, resource, limits, session_id=None, operation=None):
    """Persist the CLI request before making it visible to the supervisor."""
    row = stop_event(directory, event="stop_requested", origin="manual_cli",
                     reason="manual CLI stop requested", resource=resource, limits=limits,
                     session_id=session_id, operation=operation, requester_pid=os.getpid())
    (Path(directory) / "stop.request").touch()
    return row


class StopRecorder:
    def __init__(self, directory, limits, session_id, operation, interval=30.):
        self.directory = Path(directory)
        self.stop_path = self.directory / "stop.request"
        self.limits = dict(limits)
        self.session_id = session_id
        self.operation = operation
        self.interval = interval
        self.pending_signals = deque()
        self.current_stop = None
        self.last_stop = next((row for row in reversed(read_events(directory))
                               if row.get("event") in ("stop_requested", "failure")), None)
        self.seen_stop_file = False
        self.guard_reasons = set()
        self.last_resource_time = None

    def signal_handler(self, signum, _frame):
        # Never probe resources, format JSON, take locks or fsync in a handler.
        self.pending_signals.append((signum, time.time()))
        try:
            self.stop_path.touch()
        except OSError:
            pass  # The queued request is persisted at the next safe boundary.

    @property
    def needs_observation(self):
        return bool(self.pending_signals) or (not self.seen_stop_file and self.stop_path.exists())

    def _event(self, event, origin, reason, resource, **details):
        return stop_event(self.directory, event=event, origin=origin, reason=reason,
                          resource=resource, limits=self.limits, session_id=self.session_id,
                          operation=self.operation, **details)

    def _remember(self, row):
        if self.current_stop is None:
            self.current_stop = row
        self.last_stop = row

    def observe(self, resource):
        while self.pending_signals:
            signum, received_at = self.pending_signals[0]
            name = signal.Signals(signum).name
            row = self._event("stop_requested", "signal", "received " + name, resource,
                              signum=signum, signal_name=name, received_at=received_at)
            self.pending_signals.popleft()
            self._remember(row)
            self.seen_stop_file = True
        if self.stop_path.exists() and not self.seen_stop_file:
            request = next((row for row in reversed(read_events(self.directory))
                            if row.get("event") == "stop_requested"
                            and row.get("origin") == "manual_cli"
                            and row.get("session_id") == self.session_id), None)
            if request is None:
                request = self._event("stop_requested", "external_stopfile",
                                      "external stop.request observed", resource)
            else:
                self._event("stop_observed", "manual_cli", request["reason"], resource,
                            request_event_id=request["event_id"])
            self._remember(request)
            self.seen_stop_file = True

    def guard(self, reason, resource):
        if reason not in self.guard_reasons:
            row = self._event("stop_requested", "resource_guard", reason, resource)
            self._remember(row)
            self.guard_reasons.add(reason)
        self.stop_path.touch()
        self.seen_stop_file = True

    def interrupted(self, reason, resource):
        self.observe(resource)
        if self.current_stop is None:
            self._remember(self._event("stop_requested", "interrupted", reason, resource))

    def failure(self, exc, resource, stage=None):
        self.observe(resource)
        row = self._event("failure", "failure", str(exc), resource,
                          exception_type=type(exc).__name__, stage=stage)
        self._remember(row)

    def sample(self, resource, *, now=None, force=False):
        now = time.monotonic() if now is None else now
        if not force and self.last_resource_time is not None and now-self.last_resource_time < self.interval:
            return False
        append_jsonl(self.directory / "resources.jsonl",
                     dict(schema="blackout.v11.resources.v1", time=time.time(), pid=os.getpid(),
                          session_id=self.session_id, operation=self.operation,
                          resource=resource, limits=self.limits))
        self.last_resource_time = now
        return True

    def terminal_fields(self):
        fields = {"last_stop": self.last_stop}
        if self.current_stop is not None:
            fields.update(stop_reason=self.current_stop["reason"],
                          stop_origin=self.current_stop["origin"],
                          stop_resource=self.current_stop["resource"],
                          stop_event_id=self.current_stop["event_id"])
        return fields

    def finish(self, state, resource):
        self.observe(resource)
        self.sample(resource, force=True)
        self._event("stop_completed" if state != "complete" or self.current_stop else "run_completed",
                    "completion", self.current_stop["reason"] if self.current_stop else state,
                    resource, state=state,
                    request_event_id=self.current_stop["event_id"] if self.current_stop else None)
        return self.terminal_fields()
