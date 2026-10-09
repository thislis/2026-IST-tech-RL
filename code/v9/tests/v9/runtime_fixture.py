"""Subprocess lifecycle fixture: no Unity or neural training."""
import argparse
import os
from pathlib import Path
import signal
import time
from blackout_v9.io import atomic_json, digest, read_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--task", required=True)
    path = Path(parser.parse_args().task); task = read_json(path)
    if task.get("hang"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        atomic_json(path.parent / "heartbeat.json", {"time": time.time(), "steps": 2})
        time.sleep(60)
    else:
        atomic_json(path.parent / "heartbeat.json", {"time": time.time(), "steps": 3})
        time.sleep(.1)
        atomic_json(path.parent / "result.json", {"task_sha256": digest(task), "result": {
            "valid": True, "pid": os.getpid(), "session": os.getsid(0), "group": os.getpgrp(),
            "stdin_isatty": os.isatty(0)}})


if __name__ == "__main__": main()
