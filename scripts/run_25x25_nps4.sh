#!/usr/bin/env bash
# 25x25 single-query A/B re-run with pre_seed_n_states=4 (configs/nps4_25x25.yaml),
# masked then reactive, sequential on one GPU. Same seeds, cells, classical-only
# pool and evaluation state (5) as runs/sweep_phase3_{masked,unmasked_ctrl}.json.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=runs/sweep_25x25_nps4
mkdir -p "$OUT"
"$PY" scripts/analysis/record_provenance.py "$OUT" "25x25 A/B, pre_seed_n_states=4"

for ARM in masked reactive; do
    FLAG=""; [ "$ARM" = masked ] && FLAG="--mask-visited"
    if [ -f "$OUT/sweep_$ARM.json.done" ]; then echo "[skip] $ARM"; continue; fi
    echo "=== $ARM started $(date +'%F %T')"
    "$PY" scripts/run_multi_seed_warm_vs_cold.py \
        --seeds 42 1337 2024 7 314159 --n-scenarios 5 $FLAG \
        --oracle-pool classical_only --config configs/nps4_25x25.yaml \
        --out "$OUT/sweep_$ARM.json" --cell-timeout 3600 \
        > "$OUT/$ARM.log" 2>&1 && touch "$OUT/sweep_$ARM.json.done"
    echo "=== $ARM exit=$? finished $(date +'%F %T')"
done
