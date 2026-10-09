#!/bin/zsh
set -euo pipefail

project_root="${0:A:h}"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
  [[ "$project_root" != / ]] || { print -u2 "Project root not found"; exit 1; }
  project_root="${project_root:h}"
done
exec "$project_root/code/v4/scripts/train_mappo_planner_residual_v4.sh" "$@"
