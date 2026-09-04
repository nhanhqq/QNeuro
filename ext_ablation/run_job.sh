#!/usr/bin/env bash
set -uo pipefail
OUT=$1; DATA=$2; FS=$3; FRAME=$4; HOP=$5; FOLD=$6; VARIANT=$7; SEED=$8; EPOCHS=$9; BATCH=${10}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT/runtime/job_status"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
python3 -m ext_ablation.train --data "$DATA" --output "$OUT" --fs "$FS" --frame-seconds "$FRAME" --hop-seconds "$HOP" --fold "$FOLD" --variant "$VARIANT" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH" >"$OUT/runtime/${VARIANT}_fold_${FOLD}_seed_${SEED}.log" 2>&1
rc=$?
printf '%s\n' "$rc" >"$OUT/runtime/job_status/${VARIANT}_${FOLD}_${SEED}.tmp"
mv "$OUT/runtime/job_status/${VARIANT}_${FOLD}_${SEED}.tmp" "$OUT/runtime/job_status/${VARIANT}_${FOLD}_${SEED}.code"
exit "$rc"
