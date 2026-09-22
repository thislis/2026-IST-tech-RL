#!/bin/bash
# 40M-step v7-1 main comparison + 6M-step v7-2 PPO extension, then dev evaluations.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/.venv/bin/python" -u "$project_root/scripts/connectome_main.py" "$@"
