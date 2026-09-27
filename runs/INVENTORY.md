# runs/ — Inventory and Provenance

**This directory is the single artefact tree behind the thesis.** Every
`runs/...` path cited in `Desktop\thesis\thesis-sections\*.tex` resolves here
(63/63 verified, 2026-08-16). Nothing in the thesis now depends on a second
tree.

Which artefact backs which claim is given by `tab:results_artefacts` at the end
of Chapter 4. This file records only where the files came from and what to
distrust.

---

## 1. History: why there were two trees

Artefacts were produced under successive revisions of the codebase in two
locations:

| Tree | Period | Contents |
|---|---|---|
| `Desktop\Demo\runs` | May–Jul 2026 | Headline-scale work: V1/V2/V3 sweeps, long-train subsets, unified benchmark, reward and perturbation ablations |
| `qwarm-gnn-rl\runs` | Jul–Aug 2026 | Action-space controls, multi-query/fleet studies, 50×50 A/B, geometry study, demonstration-source ablation |

The Chapter 4 artefact caption promised a consolidated tree; **this is it,
consolidated 2026-08-16.** The June-era inventory describing the old
`Demo - Copy` layout is preserved verbatim as `INVENTORY_demo_june2026.md` — it
describes a directory that is no longer the canonical one, and is kept only for
the clone history it records.

## 2. Imported from `Desktop\Demo\runs` on 2026-08-16

Copied with original modification times preserved. **No existing file was
overwritten.** Files present in both trees were left as they were here.

Cited directly in the thesis (19):

```
INVENTORY.md -> INVENTORY_demo_june2026.md   sweep_v2_headline.json
perturbation_phase4.json                     sweep_v3_adaptive.json
reward_ablation/reward_ablation_results.json unified_benchmark_100x100.json
sweep_50x50/aggregate_50x50.json             unified_benchmark_100x100_aggregate.json
v1_long_train_results.json                   unified_benchmark_sanity.json
v1_long_train_subset_results.json            unified_pareto_100x100.csv
v1_long_train_aggregate_15cell.json          v3_lambda_ablation.json
v3_long_train_results.json                   v3_realtime_25x25_aggregate.json
v3_long_train_subset_results.json            v3_transfer_benchmark.json
v3_long_train_subset_aggregate.json
```

Not cited by name, but the source of figures that are (3):

| File | Why it is here |
|---|---|
| `stats_reanalysis.json` | The paired-test p-values behind `tab:gate_scorecard`. Records the test method per comparison (`paired_seed_test_inf_to_10x_max` at 25×25 and 100×100, `drop_pairs_unless_both_finite` at 50×50) — needed to interpret any M3 row. |
| `sweep_50x50/REPORT.md` | The 50×50 run configuration (`extra_edges=3`, `deactivate_prob=0.22`, `batch_size=64`) and the record that the target-encoder dropout fix was applied at 50×50 but **not** at 100×100. |
| `sweep_50x50/GPU_DIAGNOSTIC.md` | Companion to the above. |

## 3. Duplicated across both trees

These existed in both and were **not** re-copied; the copies here are the ones
the analysis scripts read:

`cold_4x_control_{results,aggregate}.json`, `demo_source_ablation.log`,
`demo_source_ablation_partial.json`, `eval_reachability_audit.json`,
`fleet_1779277545_seed42/`, `learning_curves_25x25.json`,
`sweep_phase3_final.json`, `sweep_phase3_traced.json`,
`sweep_v1_on_100x100.json`, `sweep_50x50/sweep_v1_50x50_{1x,4x}.json`,
`traces_25x25/`.

All are byte-identical across the two trees except `sweep_phase3_traced.json`,
which differs only in timing fields — reach, cost and Dijkstra reference agree
on all 25 cells (warm 24/24, cold 12/24 over solvable), which is what
`tab:demo_ablation` cites.

## 4. Read these caveats before citing anything here

- **Execution stack.** Only `sweep_50x50_reactive_ctrl/` and
  `sweep_50x50_masked_gpu/` carry a `provenance.json`. Everything produced
  before 2026-08-12 03:20 has no recorded device and its stack cannot be
  established retrospectively. Do not compare an unrecorded artefact against a
  recorded one and attribute the difference to the variable you changed.
- **Superseded, do not cite.** `sweep_50x50_masked/` — an earlier masked 50×50
  run with an unrecorded stack. It agrees with the A/B masked arm on every
  outcome field including warm cost on all 25 cells; it is superseded on
  provenance grounds only.
- **Incomplete.** `demo_source_ablation_partial.json` holds 9 of 50 cells; the
  aggregation aborted on a reproducibility guard. The figures in
  §`results_demo_ablation` are recovered from `demo_source_ablation.log` by
  `scripts/analysis/ablation_from_log.py`.
- **Not thesis evidence.** `shaping_control/` (reward-shaping control, reactive
  action space + CPU stack), `ood_warm_vs_cold.json` and `ood_robust_pilot*`
  (demo-track checkpoints, outside the evaluation grid),
  `_aborted_cpu_reactive_ctrl_0315/` (9 partial cells, quarantined). Appendix A
  §`app-supplementary` records why.
- **`traces_25x25/*/warm/seeding_diversity.json`** carries
  `recomputed_offline: true` — recomputed after the fact, not written by the
  original run.

## 5. Re-deriving the thesis figures from this tree alone

```bash
./.venv/Scripts/python.exe verify_demo_claims.py                 # repo-side gate
cd ..\thesis && python tools\verify_claims.py                    # thesis-side gate
```

Both pass as of 2026-08-16. `tools/verify_claims.py` still declares two artefact
roots at the top of the file; with this consolidation both can now point at
`qwarm-gnn-rl\runs` and all 29 checks still pass.
