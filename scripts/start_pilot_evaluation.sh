#!/bin/bash
# One command starts development evaluation of the completed v7-1 pilots.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec "$project_root/.venv/bin/python" "$project_root/scripts/pilot_evaluation.py" "$@"
