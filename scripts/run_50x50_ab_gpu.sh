#!/usr/bin/env bash
# =====================================================================
#  50x50 ACTION-SPACE A/B -- BOTH ARMS, ONE STACK.
#
#  WHY THIS REPLACES run_50x50_reactive_parallel.sh
#  That script re-ran only the reactive arm and compared it against the
#  masked arm of 2026-08-11 07:39. Those cannot be compared: `.venv` keeps
#  sys.prefix pointed at itself, so the `uv sync` at 2026-08-11 20:21
#  replaced torch 2.11.0+cu128 with the 2.13.0+cpu wheel, and no artefact
#  recorded a device. The masked arm therefore ran on a stack that can no
#  longer be identified, and no attribution to the action space would be
#  sound -- the same confound that made the first 25x25 claim unsound.
#
#  So BOTH arms are re-run here, exactly as 25x25 did: one codebase, one
#  torch, one device, the action space the only variable. Each arm drops a
#  provenance.json recording torch version and resolved device.
#
#  The two arms run SEQUENTIALLY. Never two chains at once.
#
#  Usage:  bash scripts/run_50x50_ab_gpu.sh
#          bash scripts/run_50x50_ab_gpu.sh --merge-only
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
SEEDS="42 1337 2024 7 314159"
THREADS=6
REACT_DIR=runs/sweep_50x50_reactive_ctrl
MASK_DIR=runs/sweep_50x50_masked_gpu

# ---------------------------------------------------------------------
# run_arm <out_base> <mask_visited 0|1> <label>
# ---------------------------------------------------------------------
run_arm() {
  local base="$1" mask="$2" label="$3"
  mkdir -p "$base"

  echo
  echo "======================================================"
  echo "  ARM: $label   (mask_visited=$mask)"
  echo "  out: $base"
  echo "  started $(date +'%Y-%m-%d %H:%M:%S')"
  echo "======================================================"

  QWARM_MASK_VISITED="$mask" "$PY" scripts/analysis/record_provenance.py \
      "$base" "50x50 $label"

  local pids=""
  for S in $SEEDS; do
      out="$base/seed$S"
      if [ -f "$out/sweep_v1_50x50_1x.json" ]; then
          echo "[skip] seed $S already complete"
          continue
      fi
      echo "[launch] seed $S -> $out"
      OMP_NUM_THREADS=$THREADS \
      MKL_NUM_THREADS=$THREADS \
      QWARM_MASK_VISITED="$mask" \
      QWARM_SEEDS="$S" \
      QWARM_OUT_DIR="$out" \
      QWARM_SKIP_4X=1 \
      "$PY" scripts/run_sweep_50x50.py \
          > "runs/ab_gpu_${label}_seed$S.log" 2>&1 &
      pids="$pids $!"
      sleep 20   # stagger CUDA context creation
  done

  local fail=0
  for p in $pids; do wait "$p" || fail=$((fail+1)); done
  echo "[$label] shards finished $(date +'%Y-%m-%d %H:%M:%S'), $fail non-zero exits"

  "$PY" scripts/analysis/merge_sweep_shards.py "$base" \
        "$base/sweep_v1_50x50_1x.json"
}

if [ "${1:-}" != "--merge-only" ]; then
  if tasklist 2>/dev/null | grep -qi "python.exe"; then
      running=$(tasklist 2>/dev/null | grep -ci "python.exe")
      if [ "$running" -gt 2 ]; then
          echo "REFUSING TO START: $running python processes already running."
          echo "Override with:  FORCE=1 bash $0"
          [ "${FORCE:-}" != "1" ] && exit 1
      fi
  fi

  run_arm "$REACT_DIR" 0 reactive
  run_arm "$MASK_DIR"  1 masked
fi

echo
echo "--- A/B analysis (same-stack) ---"
QWARM_AB_MASK_DIR="$MASK_DIR" "$PY" scripts/analysis/ab_50x50_action_space.py
