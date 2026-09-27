#!/usr/bin/env bash
# =====================================================================
#  50x50 REACTIVE arm -- the missing half of the action-space A/B.
#
#  WHY THIS EXISTS
#  The 25x25 correction rests on a clean A/B: two runs, one environment,
#  one flag. The 50x50 result does not -- its masked arm is ours, but its
#  reactive arm is the published June sweep, so codebase, torch version and
#  device all differ alongside the action space. That is the exact confound
#  that made the first 25x25 claim unsound. This run supplies a reactive arm
#  in the current environment so the 50x50 contrast becomes a like-for-like
#  A/B against runs/sweep_50x50_masked/.
#
#  PARALLELISM
#  The 25 cells are independent. One process already saturates 16 threads
#  (torch's default = physical cores), leaving ~16 logical cores idle on a
#  16C/32T part. Five per-seed shards at 6 threads each use ~30 of 32 and
#  cut wall-clock from ~8 h to roughly 2 h.
#
#  Usage:  bash scripts/run_50x50_reactive_parallel.sh
#          bash scripts/run_50x50_reactive_parallel.sh --merge-only
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
SEEDS="42 1337 2024 7 314159"
BASE=runs/sweep_50x50_reactive_ctrl
THREADS=6

mkdir -p "$BASE" runs

if [ "${1:-}" != "--merge-only" ]; then

  # Refuse to start while another GPU job is live -- five more processes on
  # top of a running sweep would contend for GPU memory and slow everything.
  if tasklist 2>/dev/null | grep -qi "python.exe"; then
      running=$(tasklist 2>/dev/null | grep -ci "python.exe")
      if [ "$running" -gt 2 ]; then
          echo "REFUSING TO START: $running python processes already running."
          echo "Wait for stage 3 (100x100) to finish, then re-run this script."
          echo "Override with:  FORCE=1 bash $0"
          [ "${FORCE:-}" != "1" ] && exit 1
      fi
  fi

  echo "======================================================"
  echo "  50x50 REACTIVE arm, 5 shards x ${THREADS} threads"
  echo "  mask_visited = FALSE (this is the reactive half)"
  echo "  started $(date +'%Y-%m-%d %H:%M:%S')"
  echo "======================================================"

  pids=""
  for S in $SEEDS; do
      out="$BASE/seed$S"
      if [ -f "$out/sweep_v1_50x50_1x.json" ]; then
          echo "[skip] seed $S already complete"
          continue
      fi
      echo "[launch] seed $S -> $out"
      # MASK_VISITED deliberately unset: this is the reactive arm.
      OMP_NUM_THREADS=$THREADS \
      MKL_NUM_THREADS=$THREADS \
      QWARM_SEEDS="$S" \
      QWARM_OUT_DIR="$out" \
      QWARM_SKIP_4X=1 \
      "$PY" scripts/run_sweep_50x50.py > "runs/reactive_50x50_seed$S.log" 2>&1 &
      pids="$pids $!"
      sleep 20   # stagger CUDA context creation so five don't init at once
  done

  echo "waiting on:$pids"
  fail=0
  for p in $pids; do wait "$p" || fail=$((fail+1)); done
  echo "all shards finished $(date +'%Y-%m-%d %H:%M:%S'), $fail non-zero exits"
fi

echo
echo "--- merging shards ---"
"$PY" scripts/analysis/merge_sweep_shards.py "$BASE" \
      "$BASE/sweep_v1_50x50_1x.json"

echo
echo "--- A/B analysis ---"
"$PY" scripts/analysis/ab_50x50_action_space.py
