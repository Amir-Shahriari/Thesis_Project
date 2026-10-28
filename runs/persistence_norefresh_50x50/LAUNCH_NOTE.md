# No-refresh arm of the persistence ablation

This run executes the optional secondary arm `keep_no_refresh`, pre-registered in
`runs/persistence_ablation_50x50/PREREGISTRATION.md` (§2 arms table, §4 secondary
tests: "`keep_no_refresh` vs `persistent`, if run"). Design, cells, budget and test
are unchanged from that plan; the result was not known when this run was launched.

- Arms: `persistent` (re-run on this stack as the replication check against the
  published 22/25) and `keep_no_refresh` (the same pre-seeded pool kept at
  rho = 0.30 for the whole run, never refreshed).
- Cells: the same 25 masked 50x50 cells (5 seeds x 5 scenarios).
- Test: exact two-sided binomial McNemar, `persistent` vs `keep_no_refresh`, on
  strict goal-reach at 300 steps; route cost by Wilcoxon on both-reached cells.
- Stack: torch 2.11.0+cu128 on an RTX 5080, from the clean commit b8f2ede of this
  repository (provenance files are written per shard).
- Code note: since the goal-adjacent ablation (7ba1e7b), `train_gnn_dqn` scores each
  transition against its own `goal_node` instead of the first transition's goal.
  In single-query training every transition carries `goal_node = dst`, so the
  change is behaviour-preserving here; the `persistent` re-run checks it empirically.
- Smoke test (1 cell, 1 iteration, 10 episodes) passed before launch: equal gradient
  steps in both arms, pre-seeded pool of 61 transitions, CUDA device.
