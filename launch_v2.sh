#!/usr/bin/env bash
set -euo pipefail
root="/home/namphuongtran9196/intel_project/QNeuro"
out="${QNEURO_V2_OUTPUT_DIR:-$root/results_v2_logonly}"
batch="${QNEURO_V2_BATCH_SIZE:-32}"
epochs="${QNEURO_V2_EPOCHS:-150}"
reserve="${QNEURO_V2_RESERVE_MB:-1024}"
estimate="${QNEURO_V2_JOB_ESTIMATE_MB:-4500}"
max_parallel="${QNEURO_V2_MAX_PARALLEL:-6}"
settle="${QNEURO_V2_SETTLE_SECONDS:-15}"
dry=0
while (($#)); do
  case "$1" in
    --dry-run) dry=1; shift;;
    --max-parallel) max_parallel="$2"; shift 2;;
    --epochs) epochs="$2"; shift 2;;
    --batch-size) batch="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
jobs=()
for spec in seed:15 seediv:15 seedv:16 seedvii:20; do
  d="${spec%%:*}"; n="${spec##*:}"
  for ((i=1;i<=n;i++)); do jobs+=("$d:P$i"); done
done
active() { screen -ls 2>/dev/null | awk '/\.qneuro_v2_(seed|seediv|seedv|seedvii)_P[0-9]+[[:space:]].*Detached/ {n++} END{print n+0}'; }
live() { screen -ls 2>/dev/null | grep -q "\.qneuro_v2_${1/:/_}[[:space:]].*Detached"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
done_job() { [[ -s "$out/${1%%:*}/target_${1##*:}/final_metrics.json" ]]; }
echo "QNEURO_V2_PLAN jobs=${#jobs[@]} epochs=$epochs batch=$batch max_parallel=$max_parallel reserve_mb=$reserve estimate_mb=$estimate"
if ((dry)); then
  for item in "${jobs[@]}"; do echo "DRY_RUN job=$item screen=qneuro_v2_${item/:/_}"; done
  exit 0
fi
mkdir -p "$out"
launch() {
  local item="$1" d t name log
  d="${item%%:*}"; t="${item##*:}"; name="qneuro_v2_${item/:/_}"; log="$out/${d}_${t}.screen.log"
  mkdir -p -- "$out/$d/target_$t"
  rm -f -- "$log"
  screen -dmS "$name" bash -lc "cd '$root'; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; python3 -u run_target.py --dataset '$d' --target '$t' --epochs '$epochs' --batch-size '$batch' --output-dir '$out' 2>&1 | tee -a '$log'; rc=\${PIPESTATUS[0]}; echo QNEURO_V2_SCREEN_EXIT dataset=$d target=$t code=\$rc timestamp=\$(date --iso-8601=seconds) | tee -a '$log'; exit \$rc"
  echo "LAUNCHED job=$item screen=$name active=$(active) free_mb=$(free_mb)"
}
while :; do
  remaining=0; launched=0
  for item in "${jobs[@]}"; do
    done_job "$item" && continue
    remaining=$((remaining+1)); name="qneuro_v2_${item/:/_}"; live "$item" && continue
    cur_active="$(active)"; cur_free="$(free_mb)"
    ((max_parallel > 0 && cur_active >= max_parallel)) && break
    [[ -z "$cur_free" ]] || ((cur_free < reserve + estimate)) && break
    launch "$item"; launched=1
    for ((s=0;s<settle;s++)); do sleep 1; live "$item" || break; now="$(free_mb)"; if [[ -n "$now" ]] && ((now < reserve)); then echo "RESERVE_BREACH job=$item free_mb=$now action=stop_newest"; screen -S "$name" -X quit || true; break; fi; done
  done
  ((remaining == 0)) && { echo "QNEURO_V2_DISPATCH_COMPLETE"; exit 0; }
  ((launched == 1)) || sleep 10
done
