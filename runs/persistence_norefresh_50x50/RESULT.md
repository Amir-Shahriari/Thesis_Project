# Result: no-refresh arm (finished 2026-10-29 01:57, 0 failed shards)

Replication: `persistent` reproduces the published run (`runs/persistence_ablation_50x50`)
exactly on all 25 cells (same reach flag and bit-identical route cost).

Pre-registered secondary test, `persistent` vs `keep_no_refresh` (strict reach, 300 steps):
- reach 22/25 vs 21/25; discordant b = 1, c = 0; exact McNemar p = 1.0
- at 1000 steps: 22/25 vs 22/25
- route cost on 21 both-reached cells: Wilcoxon p = 0.29; persistent cheaper on 12/21;
  median cost 670 vs 1055 (median paired ratio no-refresh/persistent 1.15)

Exploratory (not pre-registered; same 25 cells, compared against the published run's arms):
`keep_no_refresh` vs `dqfd_pretrain` 11/0 (p = 0.001), vs `bc_pretrain` 13/0 (p = 0.0002),
vs `cold` 12/0 (p = 0.0005).

Reading: keeping a fixed, never-refreshed demonstration pool in every batch recovers
essentially the whole reach advantage; refreshing adds no detectable reach benefit at n = 25.
A route-cost effect of refreshing is not ruled out (p = 0.29). Non-significance is not equivalence.
