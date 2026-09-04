#!/usr/bin/env bash
# One isolated screen owns exactly one (variant, fold, seed) training process.
set -uo pipefail
OUT=$1; VARIANT=$2; FOLD=$3; SEED=$4; EPOCHS=$5; BATCH_SIZE=$6
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
mkdir -p "$OUT/runtime/job_status"
LOG="$OUT/runtime/${VARIANT}_fold_${FOLD}_seed_${SEED}.log"
python3 scripts/train_chsz.py --fold "$FOLD" --seed "$SEED" --output "$OUT" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --variant "$VARIANT" --skip-audit >"$LOG" 2>&1
RC=$?
printf '%s\n' "$RC" >"$OUT/runtime/job_status/${VARIANT}_${FOLD}_${SEED}.tmp"
mv "$OUT/runtime/job_status/${VARIANT}_${FOLD}_${SEED}.tmp" "$OUT/runtime/job_status/${VARIANT}_${FOLD}_${SEED}.code"
echo "CHSZ_JOB_SCREEN_EXIT variant=$VARIANT fold=$FOLD seed=$SEED code=$RC" >>"$OUT/runtime/scheduler.log"
exit "$RC"
