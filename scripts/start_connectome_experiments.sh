#!/bin/bash
# Run the registered v7-1 and corrected v7-2 pilots, then paired dev evaluations.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/.venv/bin/python" -u "$project_root/scripts/connectome_experiments.py" "$@"
