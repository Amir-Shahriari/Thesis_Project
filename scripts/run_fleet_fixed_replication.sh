#!/usr/bin/env bash
# Re-run every multi-query (fleet) experiment of the thesis after the
# batch-goal fix (commit 4a231ba), into runs/fleet_fixed/ so the original
# runs/fleet/ artefacts stay untouched. Two lanes on one GPU, most important
# experiments first; RUN_LOG records that more than two concurrent jobs slow
# throughput badly.
#
# Usage: bash scripts/run_fleet_fixed_replication.sh
# Progress: runs/fleet_fixed/STATUS.txt (one START/DONE/FAIL line per job).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=runs/fleet_fixed
mkdir -p "$OUT/logs"
STATUS="$OUT/STATUS.txt"

"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "CUDA not available - refusing to run on CPU" | tee -a "$STATUS"; exit 1; }
"$PY" scripts/analysis/record_provenance.py "$OUT" "fleet_fixed_replication"

# job <name> <seed> <scale> <tag> <extra train args...>
job() {
  local name=$1 seed=$2 scale=$3 tag=$4; shift 4
  local log="$OUT/logs/$name.log"
  if [ -f "$OUT/reeval_seed${seed}_${scale}${tag}.json" ]; then
    echo "$(date '+%F %T') SKIP $name (already complete)" >> "$STATUS"; return 0
  fi
  echo "$(date '+%F %T') START $name" >> "$STATUS"
  if "$PY" scripts/run_fleet_mode.py --seed "$seed" --scale "$scale" \
        --tag-suffix "$tag" --out-dir "$OUT" --device cuda "$@" > "$log" 2>&1 \
     && "$PY" scripts/reeval_fleet.py --seed "$seed" --scale "$scale" \
        --tag-suffix "$tag" --fleet-dir "$OUT" --n-train-queries 20 \
        --n-eval-queries 150 --device cuda \
        --out "$OUT/reeval_seed${seed}_${scale}${tag}.json" >> "$log" 2>&1; then
    echo "$(date '+%F %T') DONE $name" >> "$STATUS"
  else
    echo "$(date '+%F %T') FAIL $name (see $log)" >> "$STATUS"
  fi
}

MQ50="--n-train-queries 20 --n-eval-queries 20 --iterations 10 --episodes 80"

lane_small() {   # 25x25: reactive, masked, then the two interventions
  for s in 1337 2024 7 314159; do job r25_s$s $s 25x25 ""; done
  for s in 42 1337 2024 7 314159; do
    job m25_s$s $s 25x25 _masked --n-train-queries 20 --n-eval-queries 20 --mask-visited
  done
  for s in 42 1337 2024 7 314159; do job gr25_s$s $s 25x25 _gr --goal-relative; done
  job her25_s42 42 25x25 _her --her-k 4 --her-strategy future
}

lane_large() {   # 50x50: masked (specified action space) first, then reactive
  for s in 42 1337 2024; do job m50_s$s $s 50x50 _mq_masked $MQ50 --mask-visited; done
  for s in 42 1337 2024; do job r50_s$s $s 50x50 _mq $MQ50; done
}

echo "$(date '+%F %T') QUEUE START (commit $(git rev-parse --short HEAD))" >> "$STATUS"
lane_small & P1=$!
lane_large & P2=$!
wait $P1 $P2
echo "$(date '+%F %T') QUEUE END" >> "$STATUS"
