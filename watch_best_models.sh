#!/bin/bash
set -euo pipefail
WATCH_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$WATCH_PROJECT_ROOT"
exec "$WATCH_PROJECT_ROOT/.venv/bin/python" -u "$WATCH_PROJECT_ROOT/tools/watch_best_models.py" "$@"
