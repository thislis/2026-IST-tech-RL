#!/bin/bash
# Run the registered v7-1 and corrected v7-2 pilots, then paired dev evaluations.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
    [[ "$project_root" != / ]] || { echo "Project root not found" >&2; exit 1; }
    project_root="$(dirname -- "$project_root")"
done
exec "$project_root/.venv/bin/python" -u "$project_root/code/v7/scripts/connectome_experiments.py" "$@"
