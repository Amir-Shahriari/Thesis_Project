# Pre-registration: does persistence matter? (50×50 persistence ablation)

**Written:** 2026-09-26, before any run of this experiment.
**Author decision:** the author approved the experiment on 2026-09-26. Nothing below
may be changed after results exist. Any deviation is reported as a deviation.
**Script:** `qwarm-gnn-rl/scripts/run_persistence_ablation.py`

## 1. Question

The warm agent keeps expert demonstrations in its replay stream throughout
training, refreshed every iteration. The ICDM paper names this "persistent
expert replay". The literature's standard alternatives supply demonstrations
**only at the start**:

- **DQfD's own pretraining phase**, and Fabrizio et al.'s DQfD pre-training stage
  (survey `tab:warmstart`, "demonstration-augmented value learning").
- **Imitation pretraining (behaviour cloning, then RL)**, the lineage of LOGGIA
  and DAgger. It is the only mechanism in the survey's table with a graph-RL
  pathfinding instance.

Is the warm-start advantage attributable to the demonstrations being
persistent, or would supplying the same demonstrations once, at the start, do
as well?

## 2. Arms (all in every cell, on one recorded execution stack)

| Arm | Demonstrations | Loss while they are present |
|---|---|---|
| `persistent` | pre-seeded, then kept at ρ = 0.30 and refreshed every iteration (= the thesis warm agent) | TD + DQfD margin |
| `dqfd_pretrain` | same pre-seeded pool; `N_pre` expert-only gradient steps, then **removed**; RL with ρ = 0 | TD + DQfD margin (pretraining only) |
| `bc_pretrain` | same pre-seeded pool; `N_pre` supervised steps, then **removed**; RL with ρ = 0 | cross-entropy on the expert action over active neighbours |
| `cold` | none (= the thesis cold agent) | TD |
| `keep_no_refresh` *(optional, secondary)* | pre-seeded pool kept at ρ = 0.30, **not** refreshed | TD + DQfD margin |

The pre-seeded pool is built identically in every demonstration arm: the same
`discover_all_paths` call, library seeding, and goal-adjacent terminal
transitions as `train_gnn_dqn`.

## 3. Fixed settings

- 50×50, **masked action space** (`QWARM_MASK_VISITED=1`), the configuration in
  which the warm-versus-cold reach contrast survives every control (22/25
  against 9/25).
- The same 25 cells (5 seeds × 5 scenarios, sampled exactly as in
  `run_sweep_50x50.py`), 1× tier (10 iterations × 200 episodes × 4 gradient
  steps), batch 64, d = 128, γ = 0.95, oracles as in the 50×50 sweep.
- `N_pre` = **1,000** gradient steps, **in addition to** the RL budget, which is
  identical for every arm. This favours the pretraining arms (about 12% more
  gradient steps), so a win for `persistent` is conservative.
- Evaluation: greedy rollout with visited nodes masked, `max_steps` = 300
  (primary) and 1000 (secondary), as in the 50×50 sweep.

## 4. Tests

- **Primary:** strict goal-reach at 300 steps, `persistent` vs `dqfd_pretrain`
  and `persistent` vs `bc_pretrain`. Exact two-sided binomial McNemar test,
  Holm-corrected over these two comparisons, α = 0.05.
- **Secondary:** each pretraining arm vs `cold` (exact McNemar); route cost on
  both-reached cells (Wilcoxon signed-rank; reported as untestable when fewer
  than 6 pairs); `keep_no_refresh` vs `persistent`, if run.
- **Replication check:** `persistent` vs `cold` on the new stack is compared
  with the published 22/25 against 9/25. A material departure (for example, the
  contrast losing significance) is reported as a stack or rerun effect before
  any ablation result is interpreted.
- **Power:** with 25 cells, p < 0.05 needs at least 6 discordant cells, all in
  one direction. A pretraining arm must therefore fail at least 6 cells that
  `persistent` reaches, with no reverse cases, for the primary test to detect
  it.

## 5. How each outcome will be reported (fixed now)

| Outcome | What the thesis says |
|---|---|
| **A.** `persistent` beats both pretraining arms (Holm p < 0.05) | Persistence is supported as the operative mechanism: demonstrations supplied once do not reproduce the advantage. This strengthens RQ1 and the ICDM framing. |
| **B.** no detectable difference from one or both pretraining arms | Reported as a **bound, not an equivalence**: the advantage is attributable to the demonstrations, but persistence is not shown to be necessary at this power. The thesis must then say that the ICDM paper names the mechanism used, not one shown to be necessary. |
| **C.** a pretraining arm beats `persistent` | Reported as a reversal, in the same register as the action-space finding: the advantage does not require persistence, and persistence may cost reach. |
| **D.** the replication check fails | Reported first. The ablation is interpreted only against the new-stack `persistent` and `cold` arms, never against the published ones. |

Route-cost results qualify the reach result and are reported beside it whatever
their direction. Nothing is dropped for being null.
