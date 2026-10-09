#!/bin/bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
    [[ "$project_root" != / ]] || { echo "Project root not found" >&2; exit 1; }
    project_root="$(dirname -- "$project_root")"
done
exec "$project_root/code/v7/scripts/start_v7_background.sh" --config code/v7/configs/v7_2_fixed.yaml "$@"
