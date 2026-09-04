#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd);cd "$ROOT"
OUTROOT=${ALLQ_OUTPUT:-results/all_quantum_v2}; MAX=${ALLQ_MAX_PARALLEL:-8}; BATCH=${ALLQ_BATCH_SIZE:-128}; EPOCHS=${ALLQ_EPOCHS:-300}; SEEDS=${ALLQ_SEEDS:-"7"}; RESERVE=${ALLQ_VRAM_RESERVE_MIB:-1024}; HOST_RESERVE=${ALLQ_HOST_RESERVE_MIB:-12288}; DATASETS=${ALLQ_DATASETS:-}; RZ_NOISE_STD=${ALLQ_RZ_NOISE_STD:-0.10}; PCA_COMPONENTS=${ALLQ_PCA_COMPONENTS:-8}
mkdir -p "$OUTROOT/runtime";exec >>"$OUTROOT/runtime/scheduler.log" 2>&1
echo "ALLQ_START protocol=LOSO_target_selected seeds=$SEEDS epochs=$EPOCHS batch=$BATCH max=$MAX datasets=${DATASETS:-all} rz_noise_std=$RZ_NOISE_STD pca_components=$PCA_COMPONENTS encoding_scale=per_layer_qubit"
jobs=()
is_complete(){
 local out=$1 fold=$2 seed=$3 code run
 code="$out/runtime/job_status/${fold}_${seed}.code"
 run="$out/runs/quantum_fold${fold}_seed${seed}"
 [[ -f "$code" && $(<"$code") == 0 && -f "$run/best.pt" && -f "$run/last.pt" && -f "$run/target_selected_test.json" ]]
}
while read -r data fs frame hop folds;do
 python3 scripts/audit_chsz.py --data "$data" --results "$OUTROOT/$data/audit" >/dev/null
 for seed in $SEEDS;do for ((fold=0;fold<folds;fold++));do
  if is_complete "$OUTROOT/$data" "$fold" "$seed";then
   echo "ALLQ_SKIP_COMPLETE data=$data fold=$fold seed=$seed"
  else
   jobs+=("$data:$fs:$frame:$hop:$fold:$seed")
  fi
 done;done
done < <(python3 - "$DATASETS" <<'PY'
import json
import sys
import pandas as pd

allowed = set(sys.argv[1].split())
contracts = json.load(open('configs/dataset_signal_contracts.json'))
for name, cfg in contracts.items():
    if allowed and name not in allowed:
        continue
    print(name, cfg['sampling_rate'], cfg['frame_seconds'], cfg['hop_seconds'], pd.read_csv(name + '/meta.csv')['subject'].nunique())
PY
)
declare -A active code_path; running=0
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -dc '0-9'; }
host_avail(){ awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
screen_is_active(){
 local name=$1
 screen -ls 2>/dev/null | grep -Eq "[[:space:]][0-9]+\\.${name}[[:space:]]"
}
launch(){
 local item data fs frame hop fold seed name out
 item=$1;IFS=: read -r data fs frame hop fold seed <<<"$item";name="allq_${data//-/_}_f${fold}_s${seed}";out="$OUTROOT/$data"
 mkdir -p "$out/runtime/job_status"
 screen -dmS "$name" bash -lc "cd '$ROOT' && bash scripts/run_quantum_job.sh '$out' '$data' '$fs' '$frame' '$hop' '$fold' '$seed' '$EPOCHS' '$BATCH' '$RZ_NOISE_STD' '$PCA_COMPONENTS'"
 active[$name]=$name;code_path[$name]="$out/runtime/job_status/${fold}_${seed}.code";running=$((running+1));echo "ALLQ_LAUNCH screen=$name data=$data fold=$fold seed=$seed"
}
# A scheduler may be restarted while independent worker screens continue.  Add
# those workers to the admission accounting before launching anything so no
# duplicate fold/seed is ever started and MAX remains a real concurrency cap.
pending=()
for item in "${jobs[@]}"; do
 IFS=: read -r data fs frame hop fold seed <<<"$item"; name="allq_${data//-/_}_f${fold}_s${seed}"; out="$OUTROOT/$data"
 if screen_is_active "$name"; then
  active[$name]=$name;code_path[$name]="$out/runtime/job_status/${fold}_${seed}.code";running=$((running+1));echo "ALLQ_ADOPT_ACTIVE screen=$name data=$data fold=$fold seed=$seed"
 else
  pending+=("$item")
 fi
done
jobs=("${pending[@]}")
while ((${#jobs[@]}||running));do
 for name in "${!active[@]}";do if test -f "${code_path[$name]}";then echo "ALLQ_EXIT screen=$name code=$(<"${code_path[$name]}")";unset 'active[$name]' 'code_path[$name]';running=$((running-1));fi;done
 while ((${#jobs[@]}&&running<MAX));do
  if (( $(gpu_free)<RESERVE || $(host_avail)<HOST_RESERVE ));then echo "ALLQ_WAIT free=$(gpu_free) host=$(host_avail)";break;fi
  item=${jobs[0]};jobs=("${jobs[@]:1}");launch "$item";sleep 3
 done
 sleep 8
done
echo "ALLQ_EXIT code=0"
