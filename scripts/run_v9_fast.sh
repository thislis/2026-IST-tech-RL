#!/bin/bash
set -euo pipefail
V9_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
exec "$V9_PROJECT_ROOT/.venv/bin/python" "$V9_PROJECT_ROOT/scripts/v9_experiments.py" "$@"
