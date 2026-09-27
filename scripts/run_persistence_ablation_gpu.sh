#!/usr/bin/env bash
# =====================================================================
#  50x50 PERSISTENCE ABLATION -- ALL FOUR ARMS, ONE STACK.
#
#  Pre-registered in the thesis repo: thesis-sections/PREREG-persistence-ablation.md
#  Every arm (persistent, dqfd_pretrain, bc_pretrain, cold) runs inside every
#  cell, so no comparison crosses an execution stack. Five seed shards run in
#  parallel on the one GPU, as in run_50x50_ab_gpu.sh.
#
#  BEFORE RUNNING: the .venv must hold the CUDA build. `uv sync` silently
#  reinstalls the +cpu wheel (see RUN_LOG.md, 2026-08-12). This script checks
#  and refuses to start otherwise.
#
#  Usage:  bash scripts/run_persistence_ablation_gpu.sh
#          bash scripts/run_persistence_ablation_gpu.sh --aggregate-only
#  Resumable: completed cells are skipped on relaunch.
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
SEEDS="42 1337 2024 7 314159"
THREADS=6
OUT=runs/persistence_ablation_50x50

if [ "${1:-}" = "--aggregate-only" ]; then
  QWARM_OUT_DIR="$OUT" "$PY" scripts/run_persistence_ablation.py --aggregate
  exit $?
fi

# --- pre-flight: CUDA build with sm_120 kernels (RTX 5080) ------------------
"$PY" - <<'PY' || { echo "PRE-FLIGHT FAILED: install the CUDA torch build first."; exit 1; }
import sys, torch
ok = torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"arch={torch.cuda.get_arch_list() if torch.cuda.is_available() else []}")
sys.exit(0 if ok else 1)
PY

if tasklist 2>/dev/null | grep -qi "python.exe"; then
  running=$(tasklist 2>/dev/null | grep -ci "python.exe")
  if [ "$running" -gt 2 ] && [ "${FORCE:-}" != "1" ]; then
    echo "REFUSING TO START: $running python processes already running. Override: FORCE=1"
    exit 1
  fi
fi

mkdir -p "$OUT"
echo "started $(date +'%Y-%m-%d %H:%M:%S')"
pids=""
for S in $SEEDS; do
  echo "[launch] seed $S"
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  QWARM_SEEDS="$S" QWARM_OUT_DIR="$OUT" \
  "$PY" scripts/run_persistence_ablation.py > "$OUT/seed$S.log" 2>&1 &
  pids="$pids $!"
  sleep 20   # stagger CUDA context creation
done

fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "shards finished $(date +'%Y-%m-%d %H:%M:%S'), $fail non-zero exits"

QWARM_OUT_DIR="$OUT" "$PY" scripts/run_persistence_ablation.py --aggregate
