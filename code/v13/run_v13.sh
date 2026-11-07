#!/usr/bin/env bash
set -euo pipefail
V13_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${V13_PYTHON:-}" ]]; then
    V13_EXEC="$V13_PYTHON"
elif [[ -f "$V13_ROOT/.venv/Scripts/python.exe" ]]; then
    V13_EXEC="$V13_ROOT/.venv/Scripts/python.exe"
else
    V13_EXEC="$V13_ROOT/.venv/bin/python"
fi
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$V13_EXEC" "$V13_ROOT/code/v13/scripts/v13_experiments.py" "$@"
