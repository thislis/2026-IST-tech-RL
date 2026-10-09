#!/bin/bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
    [[ "$project_root" != / ]] || { echo "Project root not found" >&2; exit 1; }
    project_root="$(dirname -- "$project_root")"
done
cd "$project_root"
exec "$project_root/.venv/bin/python" "$project_root/code/v6/scripts/launch_mappo_v6_background.py" "$@"
