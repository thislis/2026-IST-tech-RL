#!/bin/bash
# 40M-step v7-1 main comparison + 6M-step v7-2 PPO extension, then dev evaluations.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
    [[ "$project_root" != / ]] || { echo "Project root not found" >&2; exit 1; }
    project_root="$(dirname -- "$project_root")"
done
exec "$project_root/.venv/bin/python" -u "$project_root/code/v7/scripts/connectome_main.py" "$@"
