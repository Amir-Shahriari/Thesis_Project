# Provenance note (written after the run, 2026-09-27)

All five shards recorded `torch 2.11.0+cu128`, device `cuda:0` (RTX 5080), git
revision `1e9719c`, and the same seven uncommitted (dirty) tracked files, listed in
each `provenance_seeds_*.json`.

**Recording defect:** the field `git_diff_sha256` in every provenance file is
`e3b0c442...`, the SHA-256 of an empty string. The script's `git diff HEAD` call
failed to decode the diff under the default Windows text encoding, and the fallback
silently recorded an empty diff. The uncommitted code was not empty.

**Correct value, captured after the run:** `git diff HEAD`, hashed as raw bytes, is
`ddbf943b766cd5c869c02c89f1197bdd02d5186c1a648202bd078294b752ca41`.
No tracked file was modified between launch (2026-09-26 15:24) and this capture;
the only files added since are untracked (`scripts/run_persistence_ablation*.`,
`scripts/ablation_shards.ps1`, and the notes in this directory).

**Independent reproducibility check:** the `persistent` and `cold` arms of this
run reproduce the published masked 50x50 A/B (`runs/sweep_50x50_masked_gpu/`) on
all 25 cells, identical in goal-reach and in route cost to 1e-6. So the code
path under test behaves identically to the stack that produced the published result.

**Budget detail:** the pretraining arms record 9,000 total gradient steps
(1,000 pretraining + 8,000 RL) in most cells and 8,996 in a few, because a
pretraining batch with no usable transition performs no optimiser step.
