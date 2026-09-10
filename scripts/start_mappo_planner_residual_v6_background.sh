#!/bin/bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec "$project_root/.venv/bin/python" "$project_root/scripts/launch_mappo_v6_background.py" "$@"
