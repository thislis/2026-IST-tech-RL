#!/bin/bash
set -euo pipefail
V8_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$V8_PROJECT_ROOT"
exec "$V8_PROJECT_ROOT/.venv/bin/python" "$V8_PROJECT_ROOT/scripts/v8_experiments.py" "$@"
