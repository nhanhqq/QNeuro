#!/usr/bin/env bash
# Isolated conservative scheduler. It never inspects or changes prior runs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUTROOT=${AB_OUTPUT:-ablation/results_strict_loso_seed7}; MAX=${AB_MAX_PARALLEL:-8}
BATCH=${AB_BATCH_SIZE:-512}; EPOCHS=${AB_EPOCHS:-50}; SEED=${AB_SEED:-7}
RAM_LIMIT_MIB=${AB_RAM_LIMIT_MIB:-28000}; RAM_PER_WORKER_MIB=${AB_RAM_PER_WORKER_MIB:-1800}; VRAM_RESERVE_MIB=${AB_VRAM_RESERVE_MIB:-2048}
# Scope is deliberately restricted to the two previously observed >90% datasets.
# Supplying AB_DATASETS is allowed only as a further restriction.
DATASETS=${AB_DATASETS:-"CHSZ Sleep-EDF-20"}; VARIANT_FILTER=${AB_VARIANTS:-}
mkdir -p "$OUTROOT/runtime"; exec >>"$OUTROOT/runtime/scheduler.log" 2>&1
echo "AB_STRICT_START seed=$SEED epochs=$EPOCHS batch=$BATCH max=$MAX workers=0 target_selected=false"
mapfile -t jobs < <(python3 - "$DATASETS" "$VARIANT_FILTER" "$SEED" <<'PY'
import json,sys,pandas as pd
from ablation.model import VARIANTS
allow_data=set(sys.argv[1].split()); allow_var=set(sys.argv[2].split())
for name,cfg in json.load(open('configs/dataset_signal_contracts.json')).items():
    if allow_data and name not in allow_data: continue
    for variant in VARIANTS:
        if allow_var and variant not in allow_var: continue
        for fold in range(pd.read_csv(name+'/meta.csv')['subject'].nunique()):
            print(':'.join(map(str,(name,cfg['sampling_rate'],cfg['frame_seconds'],cfg['hop_seconds'],fold,variant,sys.argv[3]))))
PY
)
cgroup_mib(){ local b; b=$(</sys/fs/cgroup/memory.current); echo $((b/1024/1024)); }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
screen_live(){ screen -ls 2>/dev/null | grep -Eq "[[:space:]][0-9]+\.$1[[:space:]]"; }
complete(){ local d=$1 v=$2 f=$3 s=$4; [[ -f "$OUTROOT/$d/runtime/job_status/${v}_${f}_${s}.code" && $(<"$OUTROOT/$d/runtime/job_status/${v}_${f}_${s}.code") == 0 && -f "$OUTROOT/$d/runs/$v/fold_${f}_seed_${s}/strict_test_once.json" && -f "$OUTROOT/$d/runs/$v/fold_${f}_seed_${s}/final_selected_on_source_validation.pt" ]]; }
declare -A active status; running=0; pending=()
for item in "${jobs[@]}"; do
  IFS=: read -r d fs fr hp f v s <<<"$item"; name="ab_${d//-/_}_${v}_f${f}_s${s}"
  if complete "$d" "$v" "$f" "$s"; then echo "AB_SKIP_COMPLETE data=$d variant=$v fold=$f"
  elif screen_live "$name"; then active[$name]=1; status[$name]="$OUTROOT/$d/runtime/job_status/${v}_${f}_${s}.code"; running=$((running+1)); echo "AB_ADOPT $name"
  else pending+=("$item"); fi
done
jobs=("${pending[@]}")
while ((${#jobs[@]} || running)); do
  for name in "${!active[@]}"; do
    if [[ -f "${status[$name]}" ]]; then echo "AB_EXIT screen=$name code=$(<"${status[$name]}")"; unset 'active[$name]' 'status[$name]'; running=$((running-1)); fi
  done
  while ((${#jobs[@]} && running < MAX)); do
    used=$(cgroup_mib); free=$(gpu_free)
    if (( used + RAM_PER_WORKER_MIB > RAM_LIMIT_MIB || free < VRAM_RESERVE_MIB )); then echo "AB_WAIT ram_mib=$used vram_free_mib=$free running=$running"; break; fi
    item=${jobs[0]}; jobs=("${jobs[@]:1}"); IFS=: read -r d fs fr hp f v s <<<"$item"; name="ab_${d//-/_}_${v}_f${f}_s${s}"; out="$OUTROOT/$d"
    screen -dmS "$name" bash -lc "cd '$ROOT' && bash ablation/run_job.sh '$out' '$d' '$fs' '$fr' '$hp' '$f' '$v' '$s' '$EPOCHS' '$BATCH'"
    active[$name]=1; status[$name]="$out/runtime/job_status/${v}_${f}_${s}.code"; running=$((running+1)); echo "AB_LAUNCH screen=$name data=$d variant=$v fold=$f"; sleep 2
  done
  sleep 8
done
echo "AB_STRICT_DONE"
python3 -m ablation.report --output-root "$OUTROOT"
python3 -m ablation.audit --output-root "$OUTROOT"
python3 -m ablation.paper_report --output-root "$OUTROOT"
python3 -m ablation.paired_stats --output-root "$OUTROOT"
