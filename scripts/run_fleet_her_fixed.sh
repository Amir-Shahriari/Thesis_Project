#!/usr/bin/env bash
# Re-run the 25x25 HER arm (k=4, future) on all five seeds after the
# relabelling fix (skip relabels to the episode's own goal), into
# runs/fleet_her_fixed/. Same commands as run_fleet_fixed_her_seeds.sh; two
# seeds run at a time on the GPU.
# Usage: bash scripts/run_fleet_her_fixed.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=runs/fleet_her_fixed
STATUS="$OUT/STATUS.txt"
mkdir -p "$OUT/logs"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "CUDA not available - refusing to run on CPU" | tee -a "$STATUS"; exit 1; }
"$PY" - "$OUT" <<'EOF'
import json, subprocess, sys, torch
json.dump({"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0)},
          open(f"{sys.argv[1]}/provenance.json", "w"), indent=2)
EOF

run_seed() {
  s=$1; name=her25_s$s; log="$OUT/logs/$name.log"
  if [ -f "$OUT/reeval_seed${s}_25x25_her.json" ]; then
    echo "$(date '+%F %T') SKIP $name (already complete)" >> "$STATUS"; return
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
}

(for s in 42 2024 314159; do run_seed $s; done) &
(for s in 1337 7; do run_seed $s; done) &
wait
echo "$(date '+%F %T') HER-FIXED END" >> "$STATUS"
