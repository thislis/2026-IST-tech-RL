#!/bin/zsh
set -euo pipefail

project_root="${0:A:h:h}"
exec "$project_root/scripts/train_mappo_planner_residual_v4.sh" "$@"
