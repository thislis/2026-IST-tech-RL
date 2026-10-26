#!/bin/bash
set -euo pipefail
V11_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$V11_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$V11_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    V11_PROJECT_ROOT="$(dirname -- "$V11_PROJECT_ROOT")"
done
exec /bin/bash "$V11_PROJECT_ROOT/code/v11/scripts/run_v11_fast.sh" "$@"
