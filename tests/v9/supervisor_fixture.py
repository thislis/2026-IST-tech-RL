"""Exercise the real supervisor/locks with a waiting synthetic study only."""
import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
from blackout_v9 import runner
from blackout_v9.contracts import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--supervisor-fd", type=int)
    parser.add_argument("--machine-fd", type=int)
    parser.add_argument("--start-token")
    args = parser.parse_args()
    def waiting_study(directory, cfg, registration_sha, stop, progress, memory):
        progress(phase="synthetic_background_fixture")
        while not stop(): time.sleep(.05)
        raise InterruptedError("fixture stopped")
    with patch("blackout_v9.study.run_study", waiting_study):
        runner.supervise(args, config(args.config), Path(os.environ["V9_FIXTURE_DIRECTORY"]))


if __name__ == "__main__": main()
