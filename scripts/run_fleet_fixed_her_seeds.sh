#!/usr/bin/env bash
# Extend the fixed-code HER run (k=4, future) from seed 42 to the other four
# seeds at 25x25, into runs/fleet_fixed/, so the HER-versus-PDR comparison
# rests on five seeds like every other multi-query arm.
# Usage: bash scripts/run_fleet_fixed_her_seeds.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=runs/fleet_fixed
STATUS="$OUT/STATUS.txt"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "CUDA not available - refusing to run on CPU" | tee -a "$STATUS"; exit 1; }

for s in 1337 2024 7 314159; do
  name=her25_s$s
  log="$OUT/logs/$name.log"
  if [ -f "$OUT/reeval_seed${s}_25x25_her.json" ]; then
    echo "$(date '+%F %T') SKIP $name (already complete)" >> "$STATUS"; continue
  fi
  echo "$(date '+%F %T') START $name" >> "$STATUS"
  if "$PY" scripts/run_fleet_mode.py --seed "$s" --scale 25x25 --tag-suffix _her \
        --her-k 4 --her-strategy future --out-dir "$OUT" --device cuda > "$log" 2>&1 \
     && "$PY" scripts/reeval_fleet.py --seed "$s" --scale 25x25 --tag-suffix _her \
        --fleet-dir "$OUT" --n-train-queries 20 --n-eval-queries 150 --device cuda \
        --out "$OUT/reeval_seed${s}_25x25_her.json" >> "$log" 2>&1; then
    echo "$(date '+%F %T') DONE $name" >> "$STATUS"
  else
    echo "$(date '+%F %T') FAIL $name (see $log)" >> "$STATUS"
  fi
done
echo "$(date '+%F %T') HER-SEEDS END" >> "$STATUS"
