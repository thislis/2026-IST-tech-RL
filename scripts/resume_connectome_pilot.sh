#!/bin/bash
# Idempotently prepare the explicit v7-2 correction and start its background queue.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
"$project_root/.venv/bin/python" "$project_root/scripts/recover_connectome_pilot.py" "$@"
exec "$project_root/.venv/bin/python" "$project_root/scripts/launch_v7_queue_background.py" --queue configs/v7/pilot_v7_2_recovery_queue.json "$@"
