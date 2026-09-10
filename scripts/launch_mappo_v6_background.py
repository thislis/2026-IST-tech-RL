#!/usr/bin/env python3
"""Detached launcher with the trainer's parser, paths, preflight and process lock."""
from pathlib import Path
import fcntl
import os
import subprocess
import sys
import time

from train_mappo_planner_residual_v6 import ROOT, build_parser, preflight


def main():
    args = build_parser().parse_args()
    preflight(args)
    command = [str(ROOT/"scripts/train_mappo_planner_residual_v6.sh"), *sys.argv[1:]]
    if args.check:
        return subprocess.call(command, cwd=ROOT)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    # Serialize launchers until the child has taken its own lifetime lock.
    with (args.log_dir/"launch.lock").open("a") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.log_dir/"run.lock").open("a") as run_lock:
            try:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("v6 training is already running in this log directory")
        console = args.log_dir/"console.log"
        with console.open("ab", buffering=0) as output:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED":"1"})
        # Wait only for startup acknowledgement, never for training completion.
        for _ in range(100):
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"v6 exited during startup (code={code}); see {console}")
            pid_file = args.log_dir/"training.pid"
            if pid_file.is_file() and pid_file.read_text().strip() == str(process.pid):
                print(f"started v6 training in background\npid={process.pid}\nlog={console.resolve()}")
                return 0
            time.sleep(.1)
        print(f"v6 process spawned; startup not yet acknowledged\npid={process.pid}\nlog={console.resolve()}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
