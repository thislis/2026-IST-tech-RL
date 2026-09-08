#!/bin/zsh
set -euo pipefail

project_root="${0:A:h:h}"
cd "$project_root"

exec .venv/bin/python scripts/train_mappo_planner_residual_v5.py \
  --initial-checkpoint checkpoints/win_70_vs_scripted.pt \
  --opponent-checkpoint checkpoints/win_70_vs_scripted.pt \
  --target-checkpoint checkpoints/mappo_win_85_vs_win70_v5.pt \
  --max-env-steps 3000000 \
  --rollout-steps 2048 \
  --eval-every 50000 \
  --save-every 25000 \
  --fallback-logit-bias 6.5 \
  --confirmation-replicas 3 \
  --rollback-drop-tolerance 0.10 \
  --learning-rate 0.00005 \
  --entropy-coef 0.001 \
  --score-delta-weight 1.0 \
  --unity-shaping-weight 0.25 \
  --time-scale 50 \
  --eval-time-scale 100 \
  --device auto \
  "$@"
