#!/bin/bash
# Idempotently prepare the explicit v7-2 correction and start its background queue.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
    [[ "$project_root" != / ]] || { echo "Project root not found" >&2; exit 1; }
    project_root="$(dirname -- "$project_root")"
done
cd "$project_root"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
"$project_root/.venv/bin/python" "$project_root/code/v7/scripts/recover_connectome_pilot.py" "$@"
exec "$project_root/.venv/bin/python" "$project_root/code/v7/scripts/launch_v7_queue_background.py" --queue code/v7/configs/pilot_v7_2_recovery_queue.json "$@"
