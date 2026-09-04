#!/usr/bin/env bash
set -uo pipefail
OUT=$1; DATA=$2; FS=$3; FRAME=$4; HOP=$5; FOLD=$6; SEED=$7; EPOCHS=$8; BATCH=$9; RZ_NOISE_STD=${10:-0.10}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p "$OUT/runtime/job_status"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 scripts/train_paperlite.py \
  --data "$DATA" --fs "$FS" --frame-seconds "$FRAME" --hop-seconds "$HOP" \
  --fold "$FOLD" --seed "$SEED" --output "$OUT" --epochs "$EPOCHS" \
  --batch-size "$BATCH" --rz-noise-std "$RZ_NOISE_STD" \
  >"$OUT/runtime/quantum_fold_${FOLD}_seed_${SEED}.log" 2>&1
rc=$?
printf '%s\n' "$rc" >"$OUT/runtime/job_status/${FOLD}_${SEED}.tmp"
mv "$OUT/runtime/job_status/${FOLD}_${SEED}.tmp" "$OUT/runtime/job_status/${FOLD}_${SEED}.code"
exit "$rc"
