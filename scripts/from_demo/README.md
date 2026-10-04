# Scripts carried over from the Demo working tree

These evaluation and analysis scripts were developed in the separate `Demo` working tree (git commit a899a14) and produced artefacts that the thesis reports. Among them are the within-trajectory study (`run_v3_realtime_eval.py`), the unified benchmark (`run_unified_benchmark_100x100.py`), the V2/V3 evaluation (`run_v3_evaluation.py`) and the optimal-return analysis (`analyze_invalid_action_returns.py`). They are copied here unchanged, on 2026-10-04, so that every script behind a reported result lives in this repository.

- They import the `qwarm` package. Run them from the repository root, with `src/` on `PYTHONPATH`.
- Six scripts exist in both trees with different content (`audit_eval_reachability.py`, `cell_watchdog.py`, `run_demo_source_ablation.py`, `run_multi_seed_warm_vs_cold.py`, `run_shaping_control.py`, `run_sweep_50x50.py`). The versions in `scripts/` were kept, and the Demo copies were not brought over.
- Paths inside these scripts may still point at the Demo tree's `runs/` layout.
