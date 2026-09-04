#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUTROOT=${PL_OUTPUT:-results/paperlite_quantum}
MAX=${PL_MAX_PARALLEL:-20}
BATCH=${PL_BATCH_SIZE:-512}
EPOCHS=${PL_EPOCHS:-100}
SEED=${PL_SEED:-7}
RZ_NOISE_STD=${PL_RZ_NOISE_STD:-0.10}
RAM_LIMIT_MIB=${PL_RAM_LIMIT_MIB:-30000}
RAM_RESERVE_PER_WORKER_MIB=${PL_RAM_RESERVE_PER_WORKER_MIB:-1250}
VRAM_RESERVE_MIB=${PL_VRAM_RESERVE_MIB:-1024}
DATASETS=${PL_DATASETS:-}
mkdir -p "$OUTROOT/runtime"
exec >>"$OUTROOT/runtime/scheduler.log" 2>&1
echo "PAPERLITE_START protocol=train_epoch_then_target_test seed=$SEED epochs=$EPOCHS batch=$BATCH max=$MAX pca=false ram_limit_mib=$RAM_LIMIT_MIB"

# Cache once, sequentially.  Parallel folds only start after deterministic
# preprocessing completes, preventing cache races and peak host-RAM spikes.
if [[ -n "$DATASETS" ]]; then
  read -r -a selected <<<"$DATASETS"
  python3 scripts/build_paperlite_cache.py --datasets "${selected[@]}"
else
  python3 scripts/build_paperlite_cache.py
fi

mapfile -t jobs < <(python3 - "$DATASETS" "$SEED" <<'PY'
import json,sys,pandas as pd
allowed=set(sys.argv[1].split()); seed=sys.argv[2]
for name,cfg in json.load(open('configs/dataset_signal_contracts.json')).items():
    if allowed and name not in allowed: continue
    folds=pd.read_csv(name+'/meta.csv')['subject'].nunique()
    for fold in range(folds):
        print(':'.join(map(str,(name,cfg['sampling_rate'],cfg['frame_seconds'],cfg['hop_seconds'],fold,seed))))
PY
)

declare -A active status_path
running=0
cgroup_used(){
  local bytes
  if [[ -r /sys/fs/cgroup/memory.current ]]; then
    bytes=$(</sys/fs/cgroup/memory.current)
  else
    bytes=$(</sys/fs/cgroup/memory/memory.usage_in_bytes)
  fi
  echo $((bytes/1024/1024))
}
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
screen_active(){
  local name=$1
  screen -ls 2>/dev/null | grep -Eq "[[:space:]][0-9]+\.${name}[[:space:]]"
}
complete_job(){
  local out=$1 fold=$2 seed=$3 code run
  code="$out/runtime/job_status/${fold}_${seed}.code"
  run="$out/runs/quantum_fold${fold}_seed${seed}"
  [[ -f "$code" && $(<"$code") == 0 && -f "$run/best.pt" && -f "$run/last.pt" && -f "$run/target_selected_test.json" ]]
}
launch(){
  local item=$1 data fs frame hop fold seed name out
  IFS=: read -r data fs frame hop fold seed <<<"$item"
  name="pl_${data//-/_}_f${fold}_s${seed}"
  out="$OUTROOT/$data"
  mkdir -p "$out/runtime/job_status"
  screen -dmS "$name" bash -lc "cd '$ROOT' && bash scripts/run_paperlite_job.sh '$out' '$data' '$fs' '$frame' '$hop' '$fold' '$seed' '$EPOCHS' '$BATCH' '$RZ_NOISE_STD'"
  active[$name]=$name
  status_path[$name]="$out/runtime/job_status/${fold}_${seed}.code"
  running=$((running+1))
  echo "PAPERLITE_LAUNCH screen=$name data=$data fold=$fold seed=$seed"
}

# Match the old scheduler's resume semantics: adopt live fold screens, skip
# verified completed folds, and only queue genuinely missing work.
pending=()
for item in "${jobs[@]}"; do
  IFS=: read -r data fs frame hop fold seed <<<"$item"
  name="pl_${data//-/_}_f${fold}_s${seed}"
  out="$OUTROOT/$data"
  if complete_job "$out" "$fold" "$seed"; then
    echo "PAPERLITE_SKIP_COMPLETE data=$data fold=$fold seed=$seed"
  elif screen_active "$name"; then
    active[$name]=$name
    status_path[$name]="$out/runtime/job_status/${fold}_${seed}.code"
    running=$((running+1))
    echo "PAPERLITE_ADOPT screen=$name data=$data fold=$fold seed=$seed"
  else
    pending+=("$item")
  fi
done
jobs=("${pending[@]}")

while ((${#jobs[@]} || running)); do
  for name in "${!active[@]}"; do
    if [[ -f "${status_path[$name]}" ]]; then
      echo "PAPERLITE_EXIT screen=$name code=$(<"${status_path[$name]}")"
      unset 'active[$name]' 'status_path[$name]'
      running=$((running-1))
    fi
  done
  while ((${#jobs[@]} && running < MAX)); do
    ram_used=$(cgroup_used)
    vram_free=$(gpu_free)
    if (( ram_used + RAM_RESERVE_PER_WORKER_MIB > RAM_LIMIT_MIB || vram_free < VRAM_RESERVE_MIB )); then
      echo "PAPERLITE_WAIT cgroup_used_mib=$ram_used vram_free_mib=$vram_free running=$running"
      break
    fi
    item=${jobs[0]}
    jobs=("${jobs[@]:1}")
    launch "$item"
    sleep 2
  done
  sleep 8
done
echo "PAPERLITE_DONE"
