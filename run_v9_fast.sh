#!/bin/bash
set -euo pipefail
V9_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /bin/bash "$V9_PROJECT_ROOT/scripts/run_v9_fast.sh" "$@"
