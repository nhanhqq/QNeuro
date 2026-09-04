#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd);cd "$ROOT"; NAME=${CHSZ_SCREEN_NAME:-chsz_quantum_scheduler}; OUT=${CHSZ_OUTPUT:-results/chsz}
if screen -list | rg -q "[.]${NAME}[[:space:]]";then echo "screen already exists: $NAME";exit 1;fi
mkdir -p "$OUT/runtime";screen -dmS "$NAME" bash -lc "cd '$ROOT' && bash scripts/supervise_chsz.sh; rc=\$?; echo CHSZ_SCREEN_EXIT code=\$rc >> '$OUT/runtime/scheduler.log'; exit \$rc"
echo "started screen $NAME; log: $OUT/runtime/scheduler.log"
