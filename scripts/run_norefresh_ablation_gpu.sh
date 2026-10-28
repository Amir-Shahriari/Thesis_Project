#!/usr/bin/env bash
# No-refresh arm of the pre-registered persistence ablation: five seed shards on one GPU,
# as in scripts/run_goaladj_ablation_gpu.sh. Resumable: completed cells are skipped.
set -u
cd /c/Users/amirh/Desktop/Thesis_Project
PY=./.venv/Scripts/python.exe
SEEDS="42 1337 2024 7 314159"
THREADS=6
OUT=runs/persistence_norefresh_50x50
ARMS=persistent,keep_no_refresh

"$PY" - <<'PY' || { echo "PRE-FLIGHT FAILED: CUDA torch build required."; exit 1; }
import sys, torch
ok = torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
sys.exit(0 if ok else 1)
PY

mkdir -p "$OUT"
echo "started $(date +'%Y-%m-%d %H:%M:%S')"
pids=""
for S in $SEEDS; do
  echo "[launch] seed $S"
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
  QWARM_SEEDS="$S" QWARM_ARMS="$ARMS" QWARM_OUT_DIR="$OUT" \
  "$PY" scripts/run_persistence_ablation.py > "$OUT/seed$S.log" 2>&1 &
  pids="$pids $!"
  sleep 20   # stagger CUDA context creation
done

fail=0
for p in $pids; do wait "$p" || fail=$((fail+1)); done
echo "shards finished $(date +'%Y-%m-%d %H:%M:%S'), $fail non-zero exits"

QWARM_ARMS="$ARMS" QWARM_OUT_DIR="$OUT" "$PY" scripts/run_persistence_ablation.py --aggregate
