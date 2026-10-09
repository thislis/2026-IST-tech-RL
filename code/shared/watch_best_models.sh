#!/bin/bash
set -euo pipefail
WATCH_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$WATCH_PROJECT_ROOT/code/shared/project_paths.py" ]]; do
    [[ "$WATCH_PROJECT_ROOT" != / ]] || { echo "Project root not found" >&2; exit 1; }
    WATCH_PROJECT_ROOT="$(dirname -- "$WATCH_PROJECT_ROOT")"
done
cd -- "$WATCH_PROJECT_ROOT"
exec "$WATCH_PROJECT_ROOT/.venv/bin/python" -u "$WATCH_PROJECT_ROOT/code/shared/tools/watch_best_models.py" "$@"
