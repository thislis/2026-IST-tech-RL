#!/bin/bash
set -euo pipefail
V8_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$V8_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$V8_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    V8_PROJECT_ROOT="$(dirname -- "$V8_PROJECT_ROOT")"
done
cd -- "$V8_PROJECT_ROOT"
exec "$V8_PROJECT_ROOT/.venv/bin/python" "$V8_PROJECT_ROOT/code/v8/scripts/v8_experiments.py" "$@"
