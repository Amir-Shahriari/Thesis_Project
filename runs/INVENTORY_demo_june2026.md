# runs/ Inventory and Provenance

Written 2026-06-12 during cross-clone consolidation. This folder
(`Desktop\Demo - Copy\runs`) is the canonical results location; training
currently happens in this clone.

## Clone landscape at consolidation time

Seven copies of the project existed on this machine:

| Clone | Path | Role |
|---|---|---|
| Demo - Copy | `Desktop\Demo - Copy` | **Canonical (this one)** — active training |
| Demo | `Desktop\Demo` | Sibling clone; held the only copy of `cold_4x_control` |
| HPF | `Desktop\Hybrid_Path_finding_Framework` | Byte-identical to Demo/Demo - Copy for shared artifacts |
| HPF - Copy | `Desktop\Hybrid_Path_Finding_Framework - Copy` | Old-generation sweeps; incomplete `sweep_50x50/sweep_v1_50x50_4x.json` |
| backups\V1 | `Desktop\backups\V1` | Only copy of v1-realtime artifacts (May 24-25) |
| backups\myproject2 | `Desktop\backups\myproject2` | Old-generation sweeps incl. `sweep_v2_*` |
| backups\v2 | `Desktop\backups\v2` | Old-generation sweeps, subset |

Full per-file inventory with SHA256 of all 982 artifacts across all clones
was captured to `%TEMP%\qwarm_hashes.csv` on 2026-06-12.

## Two generations of sweep JSONs

The repeated sweep result files exist in two generations:

- **New generation** (file mtime 2026-06-05, ~0.5 KB larger per file):
  carried identically by Demo, Demo - Copy, and HPF. **The files at the top
  level of this runs/ folder are this generation. Thesis tables must cite
  these.**
- **Old generation** (mtimes 2026-05-11 to 2026-05-23): carried by
  HPF - Copy and the three `backups\*` copies. Preserved under
  `legacy_pre_jun5/` (see below).

Affected files: `perturbation_phase4.json`, `sweep_phase3_dqfd.json`,
`sweep_phase3_final.json`, `sweep_phase3_fixed.json`,
`sweep_phase3_postfix.json`, `sweep_phase3_ratio055.json`,
`sweep_phase3_separategraph.json`, `sweep_v1_on_100x100.json`,
`sweep_v2_headline.json`, `sweep_v2_smoke.json`, `sweep_1778482362.json`,
`sweep_1778484964.json`.

### Generation verification (2026-06-12)

Spot-checked thesis headline values against the NEW-generation files in this
folder — all match:

| Value | File | Expected | Found |
|---|---|---|---|
| 25x25 warm reach | `sweep_phase3_final.json` | 24/25 | 24/25 ✓ |
| 25x25 warm win | `sweep_phase3_final.json` | 21/25 | 21/25 ✓ |
| 100x100 warm reach | `sweep_v1_on_100x100.json` | 6/25 | 6/25 ✓ |

(Reach = scenarios with non-null `warm_cost`; win = warm reached and either
cold failed or `warm_cost < cold_cost`.) The old generation yields the same
counts for these three metrics; the Jun 5 regeneration added fields rather
than changing these results.

## Files copied into this folder on 2026-06-12

All copies verified by SHA256 against source after copying. Nothing was
moved or deleted from any source clone.

### From `Desktop\Demo\runs` (was the ONLY copy on this machine)

| Artifact | Source mtime | SHA256 |
|---|---|---|
| `cold_4x_control/` (15 files) | 2026-06-11 21:21 | per-file hashes in `%TEMP%\qwarm_hashes.csv` |
| `cold_4x_control_results.json` | 2026-06-11 21:21:31 | `F64CB364BD634E5A007AAA66B25E59CDCBBC10AD2CA1BB7F8E3799A57D30F983` |
| `cold_4x_control_aggregate.json` | 2026-06-11 21:21:33 | `2477202DB7C4A61B9C200617C56BD525DEA20EFA183A1ECA7A04E67D341289AF` |

### From `Desktop\backups\V1\runs` (was the ONLY copy on this machine)

| Artifact | Source mtime | SHA256 |
|---|---|---|
| `agents_v1_realtime/` (18 files) | 2026-05-25 | per-file hashes in `%TEMP%\qwarm_hashes.csv` |
| `v1_realtime_25x25.json` | 2026-05-25 03:48:26 | `A08C97C4858C5858AB4065EB09D6AF32C9DF2E76AA8998CF59E3552F6C0260DF` |
| `v1_realtime_25x25_aggregate.json` | 2026-05-25 03:48:26 | `301A84EC5FEA097BEEA91A751C30461079874D8061C09B88952D09D5A6CA22B4` |
| `v1_realtime_v2_25x25.json` | 2026-05-25 21:43:39 | `5EF1637BA410B2572EC3897AEDC3CBC04C1FD2A792965C6D3A189A61E091AA90` |
| `v1_realtime_v2_25x25_aggregate.json` | 2026-05-25 21:43:39 | `43CC37E7E4C9033D0010301ECA2EE66DCB168A8CD75374D96E322BEFAD14A633` |
| `v1_realtime_test_grid.json` | 2026-05-24 22:34:23 | `DB73D25DE2F30080CFD567A404884E94B68CCA5FFEA1966BB3969DA7AF7175F1` |

### `legacy_pre_jun5/` — old-generation sweep JSONs (12 files)

Provenance copies of the pre-Jun-5 generation, kept for auditability. Do
NOT cite these in the thesis; the canonical versions are the top-level
files. Sources: `backups\V1\runs` for all except `sweep_v2_headline.json`
and `sweep_v2_smoke.json`, which came from `backups\myproject2\runs`
(bk-V1 lacks them; the myproject2 copies are hash-identical to the bk-v2
and HPF - Copy copies).

## Artifacts native to this clone (not copied)

`traces_25x25/`, `sweep_phase3_traced.json`, `_div_smoke/`, `_dryrun/`,
`demo_source_ablation_sanity.json` (all Jun 11-12, demo-source-ablation
work), plus the shared canonical set: `sweep_phase3_*`, `sweep_v1_on_100x100`,
`sweep_v3_*`, `sweep_50x50/`, `fleet_*`, `libraries*`, `agents_v1*`,
`unified_benchmark*`, `v1_long_train*`, `v3_*`, `reward_ablation/`,
`eval_reachability_audit.json`, `perturbation_phase4.json`.

Related but outside runs/: `demo_agents/` (20 .pt checkpoints incl.
multi-seed s0-s4 variants, Jun 11), `runs_rerun/` (Jun 10 reproduction of
canonical runs + REPRODUCTION_REPORT.md).

## Known issues in other clones

- `HPF - Copy\runs\sweep_50x50\sweep_v1_50x50_4x.json` is an older,
  incomplete run (7 KB vs 24 KB) and that clone lacks
  `sweep_50x50\aggregate_50x50.json`. Do not source sweep_50x50 from there.
