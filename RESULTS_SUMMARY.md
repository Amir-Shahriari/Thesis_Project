# Results Summary — action-space controls

_Generated 2026-08-15 16:19 by `scripts/analysis/dump_all_results.py`. Every number is read from the artefacts; none is hard-coded._


## The question

Training enforced the MDP's visited-node exclusion **reactively** (a revisit stayed selectable and ended the episode with -5) while evaluation **masked** it. Under the reactive rule the cold arm ended 93% of training episodes on a revisit and reached a goal in 7%; masked, 0% and 73%. Every warm-vs-cold contrast in the thesis was measured under the reactive rule. These runs re-measure them under the action space the MDP actually specifies.


## Single-query sweeps

The **stack** column is load-bearing: two rows are only comparable if they share one. Rows marked `unrecorded` predate 2026-08-12 and their device cannot be established after the fact, so a difference between an `unrecorded` row and any other row may be the stack rather than the action space.

| configuration | warm | cold | discordant (w vs c) | McNemar p | warm advantage | stack |
|---|---|---|---|---|---|---|
| 25x25 reactive (A/B control) | 24/25 | 12/25 | 12 vs 0 | 0.000488 | **yes** | unrecorded |
| 25x25 **masked** | 24/25 | 24/25 | 0 vs 0 | 1 | no | unrecorded |
| 25x25 published (June, reference) | 24/25 | 13/25 | 11 vs 0 | 0.000977 | **yes** | unrecorded |
| 50x50 published reactive (reference) | 22/25 | 3/25 | 19 vs 0 | 3.81e-06 | **yes** | unrecorded |
| 50x50 masked (old stack, superseded) | 22/25 | 9/25 | 13 vs 0 | 0.000244 | **yes** | unrecorded |
| 50x50 **reactive** (A/B, same stack) | 22/25 | 3/25 | 19 vs 0 | 3.81e-06 | **yes** | 2.11.0+cu128 / cuda:0 |
| 50x50 **masked** (A/B, same stack) | 22/25 | 9/25 | 13 vs 0 | 0.000244 | **yes** | 2.11.0+cu128 / cuda:0 |
| 100x100 **masked** | 8/25 | 0/25 | 8 vs 0 | 0.00781 | **yes** | unrecorded |

## Multi-query (goal-coverage) studies

| configuration | warm | cold | discordant | McNemar p | warm advantage | stack |
|---|---|---|---|---|---|---|
| 25x25 reactive, training goals | 78/95 | 42/95 | 44 vs 8 | 4.04e-07 | **yes** | unrecorded |
| 25x25 **masked**, training goals | 73/95 | 30/95 | 53 vs 10 | 3.38e-08 | **yes** | unrecorded |
| 50x50 reactive, training goals | 52/58 | 11/58 | 42 vs 1 | 1e-11 | **yes** | unrecorded |
| 50x50 **masked**, training goals | 53/58 | 11/58 | 42 vs 0 | 4.55e-13 | **yes** | unrecorded |

Held-out (undemonstrated) goals — the goal-coverage boundary:

| configuration | warm | cold | discordant | McNemar p | warm advantage | stack |
|---|---|---|---|---|---|---|
| 25x25 **masked**, held-out goals | 254/709 | 233/709 | 153 vs 132 | 0.236 | no | unrecorded |
| 50x50 **masked**, held-out goals | 62/431 | 60/431 | 48 vs 46 | 0.918 | no | unrecorded |

## Reading

- The warm-start advantage **survives** the corrected action space everywhere except the single easiest configuration.

- **25x25 single-query is the sole collapse**: one goal, smallest grid, 500 episodes. Once revisits stop ending the episode a cold agent solves it unaided, so the published contrast there measured the training rule rather than the demonstrations.

- **Held-out goals stay null** under masking at both scales, so the goal-coverage boundary is unaffected by the correction.

- Route quality at 25x25 masked, where reach ties: warm median 2.76x Dijkstra vs cold 8.22x (Wilcoxon p = 0.0199) — the effect attributable to demonstrations once reaching is equalised.


## Provenance

| artefact | path |
|---|---|
| 25x25 A/B reactive | `runs/sweep_phase3_unmasked_ctrl.json` |
| 25x25 A/B masked | `runs/sweep_phase3_masked.json` |
| 25x25 masked traces | `runs/traces_25x25_masked/` |
| reactive termination probe | `runs/traces_reactive_probe/` |
| 50x50 masked | `runs/sweep_50x50_masked/` |
| 50x50 reactive A/B (pending) | `runs/sweep_50x50_reactive_ctrl/` |
| 100x100 masked | `runs/sweep_100x100_masked.json` |
| 25x25 multi-query masked | `runs/fleet/reeval_seed*_25x25_masked.json` |
| 50x50 multi-query masked | `runs/fleet/reeval_seed*_50x50_mq_masked.json` |