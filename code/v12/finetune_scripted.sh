#!/usr/bin/env bash
set -euo pipefail
V12_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
V12_PYTHON="${V12_PYTHON:-$V12_PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$V12_PYTHON" && -x "$V12_PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
    V12_PYTHON="$V12_PROJECT_ROOT/.venv/Scripts/python.exe"
fi
[[ -x "$V12_PYTHON" ]] || { echo "Python environment missing: $V12_PYTHON" >&2; exit 1; }
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$V12_PYTHON" "$V12_PROJECT_ROOT/code/v12/scripts/finetune_scripted.py" "$@"
