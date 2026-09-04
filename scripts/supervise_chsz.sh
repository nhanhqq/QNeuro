#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=${CHSZ_OUTPUT:-results/chsz}; RESERVE_MIB=${CHSZ_VRAM_RESERVE_MIB:-1024}; TARGET_USED_MIB=${CHSZ_TARGET_USED_MIB:-22000}; HOST_RESERVE_MIB=${CHSZ_HOST_RESERVE_MIB:-6144}; MAX_PARALLEL=${CHSZ_MAX_PARALLEL:-16}; LAUNCH_SETTLE_SECONDS=${CHSZ_LAUNCH_SETTLE_SECONDS:-3}; EPOCHS=${CHSZ_EPOCHS:-100}; BATCH_SIZE=${CHSZ_BATCH_SIZE:-128}; SEEDS=${CHSZ_SEEDS:-"7 17 27"}
mkdir -p "$OUT/runtime"; exec >>"$OUT/runtime/scheduler.log" 2>&1
echo "CHSZ_SCHEDULER_START $(date -Is) max_parallel=$MAX_PARALLEL batch_size=$BATCH_SIZE target_used_mib=$TARGET_USED_MIB reserve_mib=$RESERVE_MIB host_reserve_mib=$HOST_RESERVE_MIB epochs=$EPOCHS seeds=$SEEDS"
python3 scripts/audit_chsz.py --data CHSZ --results "$OUT/audit" >/dev/null
# Keep proposed quantum jobs first; then use independent mandated controls as safe backfill.
jobs=()
for variant in quantum classical identity no_bilstm; do
 for seed in $SEEDS; do
  for f in 0 1 2; do
   test -f "$OUT/runs/${variant}_fold${f}_seed${seed}/run.json" || jobs+=("$variant:$f:$seed")
  done
 done
done
mkdir -p "$OUT/runtime/job_status"; declare -A screen_name job_key; running=0
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
used_mib(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }
host_available_mib(){ awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
launch(){
 local item variant f seed name key
 item=$1; variant=${item%%:*}; item=${item#*:}; f=${item%%:*}; seed=${item##*:}
 name="chsz_${variant}_f${f}_s${seed}"; key="${variant}_${f}_${seed}"
 if screen -list | rg -q "[.]${name}[[:space:]]"; then
  echo "SKIP_DUPLICATE_SCREEN name=$name"; return
 fi
 screen -dmS "$name" bash -lc "cd '$ROOT' && bash scripts/run_job_chsz.sh '$OUT' '$variant' '$f' '$seed' '$EPOCHS' '$BATCH_SIZE'"
 screen_name[$name]=$name; job_key[$name]=$key; running=$((running+1)); echo "CHSZ_LAUNCH_SCREEN name=$name variant=$variant fold=$f seed=$seed"
}
while ((${#jobs[@]} || running)); do
 for name in "${!screen_name[@]}";do key=${job_key[$name]}; code="$OUT/runtime/job_status/${key}.code"; if test -f "$code";then rc=$(<"$code");echo "CHSZ_JOB_EXIT screen=$name key=$key code=$rc";unset 'screen_name[$name]' 'job_key[$name]';running=$((running-1));fi;done
 while ((${#jobs[@]} && running<MAX_PARALLEL));do
  free=$(free_mib); used=$(used_mib); host_avail=$(host_available_mib)
  if ((free<RESERVE_MIB || used>=TARGET_USED_MIB || host_avail<HOST_RESERVE_MIB));then echo "WAIT_CAPACITY used_mib=$used target_mib=$TARGET_USED_MIB free_mib=$free host_available_mib=$host_avail";break;fi
  f=${jobs[0]};jobs=("${jobs[@]:1}");launch "$f"; sleep "$LAUNCH_SETTLE_SECONDS"
 done
 sleep 10
done
echo "CHSZ_SCHEDULER_EXIT $(date -Is) code=0"
