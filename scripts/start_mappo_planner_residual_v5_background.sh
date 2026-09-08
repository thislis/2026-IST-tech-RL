#!/bin/zsh
set -euo pipefail

project_root="${0:A:h:h}"
cd "$project_root"

log_dir="logs/mappo_planner_residual_v5"
pid_file="$log_dir/training.pid"
console_log="$log_dir/console.log"
mkdir -p "$log_dir"

if [[ -f "$pid_file" ]]; then
  existing_pid="$(<"$pid_file")"
  if [[ "$existing_pid" == <-> ]] && kill -0 "$existing_pid" 2>/dev/null; then
    print -u2 "v5 training is already running (pid=$existing_pid)"
    exit 1
  fi
fi

nohup ./scripts/train_mappo_planner_residual_v5.sh "$@" \
  >> "$console_log" 2>&1 &
training_pid=$!
print -r -- "$training_pid" > "$pid_file"

print "started v5 training in background"
print "pid=$training_pid"
print "log=$project_root/$console_log"
