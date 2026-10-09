#!/bin/bash
set -euo pipefail
V9_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$V9_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$V9_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    V9_PROJECT_ROOT="$(dirname -- "$V9_PROJECT_ROOT")"
done
exec /bin/bash "$V9_PROJECT_ROOT/code/v9/scripts/run_v9_fast.sh" "$@"
