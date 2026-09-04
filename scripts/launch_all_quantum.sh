#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd);cd "$ROOT";NAME=${ALLQ_SCREEN_NAME:-all_quantum_scheduler}
screen -list | rg -q "[.]${NAME}[[:space:]]" && { echo "screen exists: $NAME";exit 1; }
screen -dmS "$NAME" bash -lc "cd '$ROOT' && bash scripts/supervise_all_quantum.sh";echo "started $NAME"
