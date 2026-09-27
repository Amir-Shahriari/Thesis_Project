#!/usr/bin/env bash
# =====================================================================
#  Masked action-space controls -- all outstanding GPU runs, SEQUENTIAL.
#
#  One GPU: these must not run concurrently. Each stage logs to runs/
#  and is skipped if its output already exists, so the script is safe to
#  re-run after an interruption.
#
#  Total: roughly 26-36 h.
#
#  Usage:  bash scripts/run_all_masked.sh
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
mkdir -p runs

stamp () { date +"%Y-%m-%d %H:%M:%S"; }
banner () { echo; echo "======================================================"; \
            echo "  $1"; echo "  started $(stamp)"; \
            echo "======================================================"; }

# ---------------------------------------------------------------------
# STAGE 1 -- 50x50 single-query masked replication, 1x tier only (~7 h)
#   Tests the 88% vs 12% reach claim, the last MRes claim resting on the
#   uncorrected revisit rule. All three env vars are required:
#   QWARM_OUT_DIR keeps it clear of the published artefacts, and
#   QWARM_SKIP_4X prevents the 4x tier auto-running (+28 h).
# ---------------------------------------------------------------------
if [ -f runs/sweep_50x50_masked/sweep_v1_50x50_1x.json ]; then
    echo "[skip] stage 1 already complete"
else
    banner "STAGE 1/3  50x50 single-query, masked, 1x tier"
    QWARM_MASK_VISITED=1 \
    QWARM_OUT_DIR=runs/sweep_50x50_masked \
    QWARM_SKIP_4X=1 \
    "$PY" scripts/run_sweep_50x50.py > runs/masked_50x50.log 2>&1
    echo "[stage 1] exit=$? finished $(stamp)"
fi

# ---------------------------------------------------------------------
# STAGE 2 -- 50x50 multi-query masked (~9 h, 3 seeds)
#   --episodes 80 reproduces the thesis protocol: the 50x50 preset's 20
#   episodes would give 10% of the single-query per-goal budget, where
#   the published study held it at 40% (10 iters x 80 eps).
# ---------------------------------------------------------------------
banner "STAGE 2/3  50x50 multi-query, masked, 3 seeds"
for S in 42 1337 2024; do
    if [ -f "runs/fleet/reeval_seed${S}_50x50_mq_masked.json" ]; then
        echo "[skip] stage 2 seed $S already complete"
        continue
    fi
    echo "--- seed $S  $(stamp) ---"
    "$PY" scripts/run_fleet_mode.py --seed "$S" --scale 50x50 \
        --n-train-queries 20 --n-eval-queries 20 \
        --iterations 10 --episodes 80 \
        --mask-visited --tag-suffix _mq_masked >> runs/fleet_50x50_masked.log 2>&1
    "$PY" scripts/reeval_fleet.py --seed "$S" --scale 50x50 \
        --tag-suffix _mq_masked --n-train-queries 20 \
        --n-eval-queries 150 >> runs/fleet_50x50_masked.log 2>&1
done
echo "[stage 2] finished $(stamp)"

# ---------------------------------------------------------------------
# STAGE 3 -- 100x100 single-query masked (~10-20 h)
#   configs/default.yaml is the headline config, restored from
#   Desktop/Demo/configs/. Full oracle pool matches the published run;
#   FaithfulQAOA short-circuits above 1000 nodes so it contributes
#   nothing at this scale either way.
# ---------------------------------------------------------------------
if [ -f runs/sweep_100x100_masked.json ]; then
    echo "[skip] stage 3 already complete"
else
    banner "STAGE 3/3  100x100 single-query, masked"
    "$PY" scripts/run_multi_seed_warm_vs_cold.py \
        --config configs/default.yaml \
        --seeds 42 1337 2024 7 314159 --n-scenarios 5 \
        --mask-visited \
        --trace-root runs/traces_100x100_masked \
        --out runs/sweep_100x100_masked.json \
        --cell-timeout 7200 --resume > runs/masked_100x100.log 2>&1
    echo "[stage 3] exit=$? finished $(stamp)"
fi

echo
echo "ALL STAGES DONE $(stamp)"
echo "Analyse with:"
echo "  $PY scripts/analysis/analyse_masked.py      # edit NEW= per stage"
echo "  $PY scripts/analysis/analyse_mq_masked.py   # edit glob to _50x50_mq_masked"
