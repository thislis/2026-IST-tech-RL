#!/usr/bin/env python3
"""Run 23 independently initialized, 2M-step experiments and 1,380 dev games.

v7-1: A0/A1/A2/A3 x five seeds (40M steps).
v7-2: implemented teammate-v2 F1 PPO x three seeds (6M steps).
This does not implement B2/B3 controls or consume the held-out final test.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.frozen_pilot_runtime import sha256, source_files

REGISTRATION = ROOT / 'reports/v7/main_study_registration.json'
DIRECTORY = ROOT / 'logs/v7/main_study'
QUEUES = {suite: ROOT / f'configs/v7/main_study/{suite.replace("-", "_")}_queue.json'
          for suite in ('v7-1', 'v7-2')}
OUTPUTS = {suite: DIRECTORY / f'{suite.replace("-", "_")}_dev_evaluation' for suite in QUEUES}


def validate_registration():
    record = json.loads(REGISTRATION.read_text())
    if record['schema'] != 'blackout.v7.main_study.v1':
        raise ValueError('unsupported main study registration')
    if source_files(ROOT) != record['sources']:
        raise ValueError('registered training sources changed; use the registered snapshot or a new study ID')
    for name, expected in record['files'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'registered main study file changed: {name}')
    if sha256(ROOT / record['source_snapshot']) != record['source_snapshot_sha256']:
        raise ValueError('main study source snapshot checksum mismatch')
    return record


def command(suite):
    # Re-enter here for each suite so pinned sources/configs are rechecked before execution.
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', suite]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage', choices=tuple(QUEUES))
    stage, remainder = parser.parse_known_args()
    if '--status' not in remainder and '--help' not in remainder and '-h' not in remainder:
        validate_registration()
    if stage.stage:
        from scripts import pilot_evaluation
        sys.argv = [sys.argv[0], '--suite', stage.stage, '--pipeline', '--current-runtime',
                    '--queue', str(QUEUES[stage.stage]), '--output-dir', str(OUTPUTS[stage.stage]), *remainder]
        pilot_evaluation.main()
        return
    from scripts import connectome_experiments as runner
    runner.DIRECTORY = DIRECTORY
    runner.SUITE_DIRECTORIES = OUTPUTS
    runner.ENTRYPOINT = Path(__file__).resolve()
    runner.SCOPE = 'v7-1 A0/A1/A2/A3 main comparison + v7-2 F1-only long training; dev target evaluation'
    runner.__doc__ = __doc__
    runner.command = command
    runner.main()


if __name__ == '__main__':
    main()
