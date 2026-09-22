#!/bin/bash
# Resume the registered main study with measured transport, MPS and process acceleration.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/.venv/bin/python" -u "$project_root/scripts/accelerated_connectome.py" "$@"
