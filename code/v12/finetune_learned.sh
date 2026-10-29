#!/usr/bin/env bash
set -euo pipefail
V12_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$V12_SCRIPT_DIR/finetune_scripted.sh" --config "$V12_SCRIPT_DIR/configs/finetune_learned.json" "$@"
