#!/bin/zsh
set -euo pipefail

project_root="${0:A:h}"
while [[ ! -f "$project_root/code/shared/project_paths.py" ]]; do
  [[ "$project_root" != / ]] || { print -u2 "Project root not found"; exit 1; }
  project_root="${project_root:h}"
done
cd "$project_root"

exec .venv/bin/python code/v2/scripts/train_mappo_vs_win70.py \
  --initial-checkpoint artifacts/checkpoints/pre_v1/win_70_vs_scripted.pt \
  --opponent-checkpoint artifacts/checkpoints/pre_v1/win_70_vs_scripted.pt \
  --target-win-rate 0.85 \
  --max-env-steps 2000000 \
  --rollout-steps 2048 \
  --eval-every 25000 \
  --save-every 5000 \
  --episode-watchdog-steps 8192 \
  --reward-mode combined \
  --score-delta-weight 1.0 \
  --unity-shaping-weight 0.25 \
  --learning-rate 0.00005 \
  --entropy-coef 0.001 \
  --device auto \
  "$@"
