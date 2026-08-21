#!/bin/zsh
set -euo pipefail

project_root="${0:A:h:h}"
cd "$project_root"

exec .venv/bin/python scripts/train_mappo_vs_win70.py \
  --initial-checkpoint checkpoints/phase3_selfplay_gen_08.pt \
  --opponent-checkpoint checkpoints/win_70_vs_scripted.pt \
  --target-win-rate 0.85 \
  --max-env-steps 2000000 \
  --rollout-steps 512 \
  --eval-every 25000 \
  --save-every 5000 \
  --reward-mode combined \
  --score-delta-weight 1.0 \
  --unity-shaping-weight 0.05 \
  --device auto \
  "$@"
