#!/usr/bin/env bash
set -euo pipefail
root="/home/namphuongtran9196/intel_project/QNeuro"
while true; do
  if "$root/launch.sh"; then exit 0; fi
  echo "QNEURO_SUPERVISOR_RESTART timestamp=$(date --iso-8601=seconds)"
  sleep 15
done
