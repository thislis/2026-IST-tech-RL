#!/bin/bash
set -euo pipefail
V9_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$V9_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$V9_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    V9_PROJECT_ROOT="$(dirname -- "$V9_PROJECT_ROOT")"
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
exec "$V9_PROJECT_ROOT/.venv/bin/python" "$V9_PROJECT_ROOT/code/v9/scripts/v9_experiments.py" "$@"
