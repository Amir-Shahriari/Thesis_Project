# Experiment Run Log

Chronological record of runs, with commands, timings and outcomes.
Companion to `..\thesis\PROGRESS.md`, which holds the thesis-side status.

Interpreter for everything below: `./.venv/Scripts/python.exe`
(system python lacks numpy/torch).

> **Device warning (added 2026-08-12).** `./.venv/Scripts/python.exe` is **not**
> a trampoline that inherits the conda env's CUDA torch — see the correction
> under 2026-08-10/11. Runs dated before 2026-08-12 03:20 have **no recorded
> device** and cannot be assumed to be GPU. Every run from that point drops a
> `provenance.json` (`scripts/analysis/record_provenance.py`); check it before
> comparing any two artefacts.

---

## 2026-08-09 — desk calibration (no GPU)

All four settled before committing compute. Scripts now in `scripts/analysis/`.

| # | script | question | outcome |
|---|---|---|---|
| 1 | `step1_returns.py` | Does the Dijkstra-optimal path beat the -5 escape hatch? | **No, at low density.** +50.3 as-used; **-96.4** pure grid, clearing -5 on only 2/20 queries. `lambda_shape >= 1.0` fixes it but leaks the solution. |
| 2 | `step1b_nonleak.py` + `step23_calibrate.py` | Non-leaking fix? | `cost_scale = mean step cost` **and** `gamma = 0.99`. 20/20 at every density. Normalisation alone fails at 50x50 (12/20) — gamma=0.95's horizon is shorter than the path. |
| 3 | `step23_calibrate.py` | Density axis | `n_chords` 0 / 60 / 1250 -> diameter **47 / 15 / 5**. Legacy `extra_edges=2` == 1,250 chords. |
| 4 | `step23_calibrate.py` | Solvability under perturbation | `p_deact = 0.15` holds solvability at 98-99 % across all densities. p=0.30 leaves the sparse config at 91 % vs 100 %. Milder than feared. |
| 5 | `step23_calibrate.py` | `T_max` sizing | p95 optimal = 37 hops -> ~148 needed. Current 400/300 **already adequate**. No change. |

## 2026-08-09 — environment diagnostics (no GPU)

| script | finding |
|---|---|
| `hopdist.py` | `extra_edges` is **per node**. 25x25 = 1,236 chords vs 1,200 grid edges; 100x100 = 39,963 vs 19,800. Diameter **5 at every scale**; hop IQR 1 / 1 / **0**. Pure grid: 47 / 96 / 194. |
| `locality_control.py` | **Positive control fails.** Ground-truth coordinates score rho = 0.091 / 0.034 / **-0.012** on the as-used graphs vs **0.978 / 0.980 / 0.978** on pure grids. The locality probe cannot detect metric structure here even when handed a perfect representation -> the geometry "metric structure" claim was **withdrawn** from the thesis. |
| `blockers.py` | Reactive revisit rule kills **98.2 %** of uniform rollouts at the as-used density (median 5 steps). Not a low-degree effect — already dominant in every published experiment. |
| `revisit_rules.py` | Rule x epsilon. reactive 0.6 / 12.5 / 67.8 %; mask 10.0 / 82.8 / 99.4 %; fallback 49.8 / 100 / 100 % (eps = 1.0 / 0.4 / 0.05, n_chords=0). eps<1 uses the Dijkstra next hop = **oracle upper bound**. |
| `v2_mcnemar.py` | V2 "isolated effect" 6/25 -> 10/25 is **not significant**: gains 6, loses 2, exact McNemar **p = 0.29**. V2->V3 p = 0.63; V1->V3 p = 0.070. Transfer benchmark V1 vs V3 p = 0.375. All were previously asserted without a test. |
| `sample_eff_test.py` | Sample efficiency (reactive) 17/24 vs 1/24 -> 16 vs 0 discordant, **p = 3.1e-5**. Result existed but had no test and appeared in no summary. |
| `mq_cost.py` | Multi-query paired **cost**: 25x25 training goals, 34 both-reached, warm cheaper 25/34, **Wilcoxon p = 0.018**. Closed a gap §5.3 had listed as needing new runs. |
| `geom.py` | Extracted the unreported geometry study: IsoScore 0.044/0.034, effective rank 20/64 and 26/128, whitening drives saturation 78.5 % -> 0.1 % (p = 0.002) with retrieval quality unmoved (p = 0.28). |

## 2026-08-09 21:04 — masked replication, ATTEMPT 1 — ABORTED

```bash
./.venv/Scripts/python.exe scripts/run_multi_seed_warm_vs_cold.py \
  --seeds 42 1337 2024 7 314159 --n-scenarios 5 --mask-visited \
  --trace-root runs/traces_25x25_masked --out runs/sweep_phase3_masked.json \
  --cell-timeout 5400
```
**Failed:** `qiskit-aer` missing -> full oracle pool raised ImportError -> cells
**SKIPPED**, producing nothing. Caught at cell 12/25. Stopped.

## 2026-08-09 21:15 — ATTEMPT 2 — ABORTED

Installed `qiskit 2.5.1` — **wrong**, pyproject pins `qiskit>=1.0,<2`
(lock: 1.4.5 / aer 0.13.3). Would have fed the warm arm demonstrations from a
different solver than the published run. Reverted:
```bash
uv sync --extra qiskit_backend
```
Even on 1.4.5 a single cell exceeded 550 s, so QAOA slowness is real, not a
version artefact. Switched to `--oracle-pool classical_only`, licensed by §4.4's
source-invariance result (24/24 under all three pools, McNemar p = 1.0) and
already precedented by §4.6.

## 2026-08-09 21:30 — masked replication, 25x25 single-query — COMPLETE

```bash
./.venv/Scripts/python.exe scripts/run_multi_seed_warm_vs_cold.py \
  --seeds 42 1337 2024 7 314159 --n-scenarios 5 --mask-visited \
  --oracle-pool classical_only \
  --trace-root runs/traces_25x25_masked \
  --out runs/sweep_phase3_masked.json --cell-timeout 3600
```
**~60 min, 25/25 cells, 0 skipped.** ~2.3 min/cell.
Artefacts: `runs/sweep_phase3_masked.json`, `runs/traces_25x25_masked/`,
log `runs/masked_replication.log`.

Analyse: `scripts/analysis/analyse_masked.py`, `scripts/analysis/masked_cost.py`

**Result — the reach and sample-efficiency advantages are ARTEFACTS:**
- cold reach 13/25 -> **24/25**; discordant 11 vs 0 -> **0 vs 0**, p = 1.0
- cold threshold-crossing 1/24 -> **25/25**; both arms median 25 episodes
- **route quality SURVIVES:** warm cheaper 15/21, **Wilcoxon p = 0.0199**,
  median 157 vs 391, **2.76x vs 8.22x** Dijkstra
- Gates re-scored: M1 60 % FAIL, M2 7.05 FAIL (median 2.76 passes), M3 p = 0.0199

## 2026-08-09 23:10 — masked multi-query, 25x25, 5 seeds — COMPLETE

Required wiring `--mask-visited` through `scripts/run_fleet_mode.py`
(signature -> `kwargs` -> call site).

```bash
for S in 42 1337 2024 7 314159; do
  ./.venv/Scripts/python.exe scripts/run_fleet_mode.py --seed $S --scale 25x25 \
    --n-train-queries 20 --n-eval-queries 20 --mask-visited --tag-suffix _masked
  ./.venv/Scripts/python.exe scripts/reeval_fleet.py --seed $S --scale 25x25 \
    --tag-suffix _masked --n-train-queries 20 --n-eval-queries 150
done
```
**~90 min total,** ~18 min/seed (9-10 warm + 8-9 cold).
Artefacts: `runs/fleet/reeval_seed*_25x25_masked.json`, log `runs/fleet_masked.log`.

Analyse: `scripts/analysis/analyse_mq_masked.py`

**Result — the goal-coverage advantage SURVIVES and sharpens:**
- demonstrated goals 73/95 vs 30/95; **53 vs 10 discordant; p = 3.4e-8; 5/5 seeds**
  (reactive was 44 vs 8, p = 4.0e-7)
- held-out remains null: 153 vs 132, p = 0.236 (reactive 156 vs 178, p = 0.25)
- Caveat: masking lowered absolute reach for **both** arms (warm 78->73,
  cold 42->30). No mechanism claimed.

**Combined reading:** one goal at 25x25 is solvable by any agent that can finish
an episode; twenty goals sharing one network at 200 episodes each are not, and
there demonstrations are decisive.

---

## 2026-08-10 — demonstration-source ablation, RE-ANALYSED (no GPU)

`scripts/analysis/ablation_from_log.py`. The JSON holds only **9 of 50 cells**:
the aggregation aborted on a reproducibility guard ("14/25 warm cells differ
>1 % from published, baseline 5") before joining. Recovered all 48 logged cells
from `runs/demo_source_ablation.log`.

| | classical-only | quantum-only |
|---|---|---|
| cells | 24 | 24 |
| strict reach | 24/24 | 24/24 -> 0 vs 0 discordant, **p = 1.0 = NO POWER** |
| median cost | **115** | 162 |
| paired cost | cheaper on 10 | on 6, 8 ties -> **sign test p = 0.45** |

Thesis restated in six places: the claim is a **bound** (the quantum component
buys nothing measurable) and **not an invariance** (reach saturates, so the
test could not have detected a difference).

## 2026-08-10 17:36 — ACCIDENTAL SWEEP START (no damage)

Ran `scripts/run_sweep_50x50.py --help` to inspect its CLI. **That script has no
argparse**: the flag was ignored and the real 25-cell sweep began, writing to
`runs/sweep_50x50/` — the directory holding the published artefacts. Killed
after ~2 min.

Damage check: `sweep_v1_50x50_1x.json` and `sweep_v1_50x50_4x.json` **untouched**
(mtime still 2026-07-24). `cells.json` was rewritten but its content is
byte-identical to `Desktop/Demo/runs/sweep_50x50/cells.json` (the sampler is
deterministic given seeds). No results lost.

**Never probe `run_sweep_50x50.py` with flags.** Read the source instead.

Consequence: three env switches added to that script the same day
(`QWARM_MASK_VISITED`, `QWARM_OUT_DIR`, `QWARM_SKIP_4X`), all defaulting to
existing behaviour and verified by importing the module.

## 2026-08-10/11 — the action-space controls (device unrecorded)

Switched to `torch 2.11.0+cu128` after finding `.venv` reporting CPU-only.
`.venv\Scripts\python.exe` turned out to be a trampoline into
`miniconda3\envs\qwarm_env`, which is what actually executes.

> **CORRECTION, 2026-08-12.** The sentence above is wrong, and the section
> title's "(GPU)" is unverified. `.venv\Scripts\python.exe` is a 274 KB uv
> launcher, not a 5 MB trampoline. It does exec the conda interpreter as a
> child process --- which is what the original note half-observed --- but
> `pyvenv.cfg` sets `include-system-site-packages = false` and `sys.prefix`
> stays `.venv`, so **imports resolve from `.venv\Lib\site-packages`, not from
> the conda env**. Measured in script mode:
>
> ```
> sys.prefix        : ...\qwarm-gnn-rl\.venv
> sys.base_prefix   : ...\miniconda3\envs\qwarm_env
> torch.__version__ : 2.13.0+cpu        <- from .venv, not conda
> cuda.is_available : False
> resolve_device()  : cpu
> ```
>
> Checking the conda interpreter directly reports `2.11.0+cu128 True`, which is
> real but **unreachable by the runs** (`import qwarm` fails there). That check
> is what made this look healthy for two days.
>
> Compounding it: `uv.lock` pins torch from PyPI with no CUDA index, so **any
> `uv sync` installs the +cpu wheel** and silently reverts a manual CUDA
> install. `.venv`'s torch was replaced at **2026-08-11 20:21**, mid-study.
> Artefacts before that date and after it may differ in stack, and none of them
> recorded one. The runs in the table below therefore have an **unknown
> device**; treat cross-run comparisons among them as confounded.
>
> Fixed 2026-08-12: torch reinstalled as `2.11.0+cu128` from the cu128 index
> (verified `sm_120` in `torch.cuda.get_arch_list()` --- required for the
> RTX 5080; a cu121/cu124 wheel installs cleanly and then fails at kernel
> launch), and `record_provenance.py` now stamps every run.

| run | result | artefact |
|---|---|---|
| 25x25 masked | warm 24/25, cold **24/25**, p = 1.0 | `runs/sweep_phase3_masked.json` |
| 25x25 reactive A/B ctrl | warm 24/25, cold **12/25**, p = 0.00049 (reproduces published 13/25) | `runs/sweep_phase3_unmasked_ctrl.json` |
| termination probe | cold: **93 %** episodes end on revisit, **7 %** reach goal (reactive) vs **0 %** / **73 %** (masked) | `runs/traces_reactive_probe/` |
| 25x25 multi-query masked | train goals 73/95 vs 30/95, **p = 3.4e-8**; held-out null | `runs/fleet/reeval_seed*_25x25_masked.json` |
| 50x50 masked | warm 22/25, cold 9/25, **p = 0.00024** | `runs/sweep_50x50_masked/` |
| 50x50 multi-query masked | train goals 53/58 vs 11/58, **p = 4.6e-13**, 42 vs 0 discordant | `runs/fleet/reeval_seed*_50x50_mq_masked.json` |
| 100x100 masked | warm 8/25, cold **0/25**, 8 vs 0 discordant, **p = 0.0078** (chain finished 2026-08-12 01:21) | `runs/sweep_100x100_masked.json` |

**Verdict:** the warm advantage survives the corrected action space at every
configuration except 25x25 single-query — the easiest cell in the study.
Full table: `RESULTS_SUMMARY.md` (regenerate with
`scripts/analysis/dump_all_results.py`).

### Incidents
- An orphaned chain (PID 30784) survived a TaskStop and ran concurrently with
  its replacement; three GPU jobs dropped throughput ~10.6 -> ~36 min/cell and
  risked a file collision on the `_mq_masked` outputs. Killed by PID tree.
  **Never stop a chain without killing the bash parent.**
- `taskkill /F /IM python.exe` killed unrelated processes. Use `/T /PID`.
- `run_sweep_50x50.py --help` started the full sweep (no argparse). No results
  lost; `cells.json` was rewritten with byte-identical content.

## OUTSTANDING

| priority | run | cost | why |
|---|---|---|---|
| 1 | 50x50 masked replication (1x tier) | ~7 h | 88 % vs 12 % reach claim still rests on the uncorrected rule. Command in `..\thesis\PROGRESS.md` §4 STEP 1 — **all three env vars are required** |
| 2 | 100x100 masked replication | ~10-20 h | optional — already fails M1-M3 |
| 3 | chord-density sweep (PhD) | ~1 week overnight | calibration done; needs `goal_bonus`/`cost_scale` plumbing only |

Exact commands in `..\thesis\PROGRESS.md` §4.

**Do not:** install qiskit 2.x; use `lambda_shape` for the density sweep;
compare masked runs against reactive artefacts without stating the difference;
edit `thesis_full.tex` directly (it is generated).
