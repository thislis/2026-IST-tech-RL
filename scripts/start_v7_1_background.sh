#!/bin/bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/scripts/start_v7_background.sh" --config configs/v7/v7_1_flywire.yaml "$@"
