#!/bin/bash
set -euo pipefail
V10_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$V10_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$V10_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    V10_PROJECT_ROOT="$(dirname -- "$V10_PROJECT_ROOT")"
done
exec /bin/bash "$V10_PROJECT_ROOT/code/v10/scripts/run_v10_fast.sh" "$@"
