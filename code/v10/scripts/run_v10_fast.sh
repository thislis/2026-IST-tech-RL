#!/bin/bash
set -euo pipefail
V10_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$V10_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$V10_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    V10_PROJECT_ROOT="$(dirname -- "$V10_PROJECT_ROOT")"
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
exec "$V10_PROJECT_ROOT/.venv/bin/python" "$V10_PROJECT_ROOT/code/v10/scripts/v10_experiments.py" "$@"
