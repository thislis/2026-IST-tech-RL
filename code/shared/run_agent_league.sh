#!/bin/bash
set -euo pipefail
LEAGUE_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
exec "$LEAGUE_ROOT/.venv/bin/python" "$LEAGUE_ROOT/code/run.py" -m agent_league.cli "$@"
