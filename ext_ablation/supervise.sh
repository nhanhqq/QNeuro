#!/usr/bin/env bash
# Isolated strict-LOSO scheduler; only ext_ablation/results_* and extab_* screens.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUTROOT=${EXTAB_OUTPUT:-ext_ablation/results_strict_loso_seed7}
MAX=${EXTAB_MAX_PARALLEL:-12}; BATCH=${EXTAB_BATCH_SIZE:-512}; EPOCHS=${EXTAB_EPOCHS:-50}; SEED=${EXTAB_SEED:-7}
# Admission maintains the user-set ceilings: cgroup RAM <30,000 MiB and
# GPU allocated memory <22,000 MiB on the 24,268-MiB RTX 3090.
RAM_LIMIT_MIB=${EXTAB_RAM_LIMIT_MIB:-30000}; RAM_PER_WORKER_MIB=${EXTAB_RAM_PER_WORKER_MIB:-1800}; VRAM_RESERVE_MIB=${EXTAB_VRAM_RESERVE_MIB:-2268}
DATASETS=${EXTAB_DATASETS:-"CHSZ Sleep-EDF-20"}; VARIANT_FILTER=${EXTAB_VARIANTS:-}; DRY_RUN=0
[[ ${1:-} == --dry-run ]] && DRY_RUN=1
mkdir -p "$OUTROOT/runtime"; exec >>"$OUTROOT/runtime/scheduler.log" 2>&1
echo "EXTAB_START seed=$SEED epochs=$EPOCHS batch=$BATCH max=$MAX workers=0 target_selected=false dry_run=$DRY_RUN"
mapfile -t jobs < <(python3 - "$DATASETS" "$VARIANT_FILTER" "$SEED" <<'PY'
import json,sys,pandas as pd
from ext_ablation.model import VARIANTS
allow_data=set(sys.argv[1].split()); allow_var=set(sys.argv[2].split())
for name,cfg in json.load(open('configs/dataset_signal_contracts.json')).items():
    if name not in allow_data: continue
    for variant in VARIANTS:
        if allow_var and variant not in allow_var: continue
        for fold in range(pd.read_csv(name+'/meta.csv')['subject'].nunique()):
            print(':'.join(map(str,(name,cfg['sampling_rate'],cfg['frame_seconds'],cfg['hop_seconds'],fold,variant,sys.argv[3]))))
PY
)
cgroup_mib(){ local bytes; bytes=$(</sys/fs/cgroup/memory.current); echo $((bytes/1024/1024)); }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
screen_live(){ screen -ls 2>/dev/null | grep -Eq "[[:space:]][0-9]+\.$1[[:space:]]"; }
complete(){ local data=$1 variant=$2 fold=$3 seed=$4; [[ -f "$OUTROOT/$data/runtime/job_status/${variant}_${fold}_${seed}.code" && $(<"$OUTROOT/$data/runtime/job_status/${variant}_${fold}_${seed}.code") == 0 && -f "$OUTROOT/$data/runs/$variant/fold_${fold}_seed_${seed}/strict_test_once.json" && -f "$OUTROOT/$data/runs/$variant/fold_${fold}_seed_${seed}/final_selected_on_source_validation.pt" ]]; }
declare -A active status; running=0; pending=()
for item in "${jobs[@]}"; do
  IFS=: read -r data fs frame hop fold variant seed <<<"$item"; name="extab_${data//-/_}_${variant}_f${fold}_s${seed}"
  if complete "$data" "$variant" "$fold" "$seed"; then echo "EXTAB_SKIP_COMPLETE data=$data variant=$variant fold=$fold"
  elif screen_live "$name"; then active[$name]=1; status[$name]="$OUTROOT/$data/runtime/job_status/${variant}_${fold}_${seed}.code"; running=$((running+1)); echo "EXTAB_ADOPT screen=$name"
  else pending+=("$item"); fi
done
echo "EXTAB_QUEUE pending=${#pending[@]} adopted=$running total=${#jobs[@]}"
if (( DRY_RUN )); then printf '%s\n' "${pending[@]}"; exit 0; fi
while ((${#pending[@]} || running)); do
  for name in "${!active[@]}"; do
    if [[ -f ${status[$name]} ]]; then echo "EXTAB_EXIT screen=$name code=$(<"${status[$name]}")"; unset 'active[$name]' 'status[$name]'; running=$((running-1)); fi
  done
  while ((${#pending[@]} && running < MAX)); do
    used=$(cgroup_mib); free=$(gpu_free)
    if (( used + RAM_PER_WORKER_MIB > RAM_LIMIT_MIB || free < VRAM_RESERVE_MIB )); then echo "EXTAB_WAIT ram_mib=$used vram_free_mib=$free running=$running"; break; fi
    item=${pending[0]}; pending=("${pending[@]:1}"); IFS=: read -r data fs frame hop fold variant seed <<<"$item"; name="extab_${data//-/_}_${variant}_f${fold}_s${seed}"; out="$OUTROOT/$data"
    screen -dmS "$name" bash -lc "cd '$ROOT' && bash ext_ablation/run_job.sh '$out' '$data' '$fs' '$frame' '$hop' '$fold' '$variant' '$seed' '$EPOCHS' '$BATCH'"
    active[$name]=1; status[$name]="$out/runtime/job_status/${variant}_${fold}_${seed}.code"; running=$((running+1)); echo "EXTAB_LAUNCH screen=$name data=$data variant=$variant fold=$fold"; sleep 2
  done
  sleep 8
done
echo "EXTAB_DONE"; python3 -m ext_ablation.report --output-root "$OUTROOT"; python3 -m ext_ablation.audit --output-root "$OUTROOT"; python3 -m ext_ablation.paper_report --output-root "$OUTROOT"; python3 -m ext_ablation.paired_stats --output-root "$OUTROOT"
