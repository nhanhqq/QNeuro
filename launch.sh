#!/usr/bin/env bash
# VRAM-adaptive QNeuro dispatcher. Every dataset/target is an independent
# screen; the dispatcher backfills as soon as capacity becomes available.
set -euo pipefail

root="/home/namphuongtran9196/intel_project/QNeuro"
output="$root/results"
batch_size="${QNEURO_BATCH_SIZE:-32}"
epochs="${QNEURO_EPOCHS:-100}"
min_free_vram_mb="${QNEURO_MIN_FREE_VRAM_MB:-1024}"
job_vram_estimate_mb="${QNEURO_JOB_VRAM_ESTIMATE_MB:-4300}"
launch_settle_seconds="${QNEURO_LAUNCH_SETTLE_SECONDS:-15}"
# Zero means VRAM-only admission, matching the ablation_ext override contract.
max_parallel="${QNEURO_MAX_PARALLEL:-0}"
dry_run=0

while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --max-parallel) max_parallel="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --min-free-vram-mb) min_free_vram_mb="$2"; shift 2 ;;
    --job-vram-estimate-mb) job_vram_estimate_mb="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for key in batch_size epochs min_free_vram_mb job_vram_estimate_mb launch_settle_seconds max_parallel; do
  value="${!key}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$key must be a non-negative integer" >&2; exit 2; }
done
(( batch_size > 0 && epochs > 0 && min_free_vram_mb >= 1024 && job_vram_estimate_mb > 0 )) || exit 2

jobs=()
for dataset_count in seed:15 seediv:15 seedv:16 seedvii:20; do
  dataset="${dataset_count%%:*}"; count="${dataset_count##*:}"
  for (( person=1; person<=count; person++ )); do jobs+=("$dataset:P$person"); done
done

screen_lines() { screen -ls 2>/dev/null || true; }
screen_is_active() { screen_lines | grep -q "[.]$1[[:space:]].*(Detached)"; }
active_count() { screen_lines | awk '/\.qneuro_(seed|seediv|seedv|seedvii)_P[0-9]+[[:space:]].*\(Detached\)/ {n++} END {print n+0}'; }
free_vram_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '; }
complete() {
  local dataset target base
  dataset="${1%%:*}"
  target="${1##*:}"
  base="$output/$dataset/target_$target"
  [[ -s "$base/base/final_epoch.pt" && -s "$base/metrics/62ch_final_epoch.json" ]]
}

echo "QNEURO_PLAN jobs=${#jobs[@]} scheduler=vram_adaptive max_parallel=$max_parallel batch=$batch_size epochs=$epochs workers=0 wandb=disabled reserve_mb=$min_free_vram_mb estimate_mb=$job_vram_estimate_mb"
if (( dry_run )); then
  for item in "${jobs[@]}"; do echo "DRY_RUN job=$item screen=qneuro_${item/:/_}"; done
  exit 0
fi
mkdir -p "$output"

launch_job() {
  local item dataset target name target_dir log
  item="$1"
  dataset="${1%%:*}"
  target="${1##*:}"
  name="qneuro_${1/:/_}"
  target_dir="$output/$dataset/target_$target"
  log="$output/${dataset}_${target}.screen.log"
  rm -rf -- "$target_dir"
  rm -f -- "$log"
  screen -dmS "$name" bash -lc "
    set -o pipefail
    cd '$root'
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    python3 -u run_target.py --dataset '$dataset' --target '$target' \
      --epochs-base '$epochs' --batch-size '$batch_size' --output-dir '$output' \
      2>&1 | tee -a '$log'
    status=\${PIPESTATUS[0]}
    echo QNEURO_SCREEN_EXIT dataset=$dataset target=$target code=\$status timestamp=\$(date --iso-8601=seconds) | tee -a '$log'
    exit \$status
  "
  echo "LAUNCHED job=$item screen=$name active=$(active_count) free_mb=$(free_vram_mb)"
}

declare -A attempts
last_wait=""
while :; do
  remaining=0; launched=0
  for item in "${jobs[@]}"; do
    complete "$item" && continue
    ((remaining += 1))
    name="qneuro_${item/:/_}"
    screen_is_active "$name" && continue
    if (( ${attempts[$item]:-0} >= 3 )); then continue; fi
    current_active="$(active_count)"; current_free="$(free_vram_mb)"
    if (( max_parallel > 0 && current_active >= max_parallel )); then break; fi
    if [[ -z "$current_free" ]] || (( current_free < min_free_vram_mb + job_vram_estimate_mb )); then break; fi
    attempts[$item]=$(( ${attempts[$item]:-0} + 1 ))
    launch_job "$item"
    launched=1
    for (( second=0; second<launch_settle_seconds; second++ )); do
      sleep 1
      screen_is_active "$name" || break
      free_now="$(free_vram_mb)"
      if [[ -n "$free_now" ]] && (( free_now < min_free_vram_mb )); then
        echo "RESERVE_BREACH job=$item free_mb=$free_now reserve_mb=$min_free_vram_mb action=stop_newest"
        screen -S "$name" -X quit || true
        sleep 3
        break
      fi
    done
  done

  failed=0
  for item in "${jobs[@]}"; do
    complete "$item" && continue
    name="qneuro_${item/:/_}"
    if ! screen_is_active "$name" && (( ${attempts[$item]:-0} >= 3 )); then
      echo "FAILED_AFTER_RETRIES job=$item attempts=${attempts[$item]} log=$output/${item/:/_}.screen.log"
      ((failed += 1))
    fi
  done
  if (( remaining == 0 )); then echo "QNEURO_DISPATCH_COMPLETE timestamp=$(date --iso-8601=seconds)"; exit 0; fi
  if (( failed > 0 )); then echo "QNEURO_DISPATCH_FAILED failed=$failed timestamp=$(date --iso-8601=seconds)"; exit 1; fi
  state="WAIT_CAPACITY remaining=$remaining active=$(active_count) free_mb=$(free_vram_mb) reserve_mb=$min_free_vram_mb"
  if [[ "$state" != "$last_wait" ]]; then echo "$state"; last_wait="$state"; fi
  (( launched == 1 )) || sleep 10
done
