# Pre-registration: goal-adjacent seeding ablation (50×50, masked)

**Written:** 2026-10-17, before any run of this experiment. This file and the script are
committed to git before the first full-run launch, so the commit dates the plan.
**Author decision:** the author asked for this experiment on 2026-10-17. Nothing below
may be changed after results exist. Any deviation is reported as a deviation.
**Script:** `scripts/run_goaladj_ablation.py` (launcher `scripts/run_goaladj_ablation_gpu.sh`).

## 1. Question

Besides the oracle demonstrations, the warm agent's expert pool receives a block of
**goal-adjacent terminal transitions**: for every active neighbour of the destination, a
one-step transition into the goal with reward `-step_cost + 100`. They are read off graph
adjacency, not from any oracle, and the cold agent never receives them
(`src/qwarm/training/train_gnn_dqn.py`, pre-population block). Two mock examiners named
this the most decisive missing control: is the warm-versus-cold reach advantage carried by
the oracle demonstrations, or by these goal-adjacent transitions?

## 2. Arms (all trained and evaluated inside every cell)

| Arm | Oracle demonstrations | Goal-adjacent transitions |
|---|---|---|
| `warm_no_goaladj` | yes: pre-seeded library, refreshed every iteration, ρ = 0.30 | **no** |
| `cold_goaladj` | **no** | yes, kept in the expert pool at ρ = 0.30 for the whole run |
| `persistent` | yes (= the thesis warm agent) | yes |
| `cold` | no (= the thesis cold agent) | no |

`warm_no_goaladj` is the thesis warm agent with only the goal-adjacent block removed
(`goal_adjacent_seeding=False`; that block draws no random numbers). `cold_goaladj` is the
thesis cold agent whose expert pool holds only the goal-adjacent transitions, built by the
same code on the same initial graph state, and trained under the same TD + DQfD-margin loss.

**Dose (stated in advance):** the goal-adjacent pool is small (one transition per active
neighbour of the destination, about 10–15), so with ρ = 0.30 every `cold_goaladj` batch
contains all of them, whereas in the warm agent they are diluted among hundreds of oracle
transitions. `cold_goaladj` therefore receives a **larger** dose than the warm agent ever
does. A null for `cold_goaladj` is accordingly conservative; a positive result may reflect
the dose and is reported with that caveat.

## 3. Fixed settings

- 50×50, **masked action space** (`QWARM_MASK_VISITED=1`), the configuration of the
  published 22/25 against 9/25 contrast and of the persistence ablation.
- The same 25 cells (5 seeds × 5 scenarios, sampled as in `run_sweep_50x50.py`), 1× tier
  (10 iterations × 200 episodes × 4 gradient steps), batch 64, d = 128, γ = 0.95, oracles
  as in the 50×50 sweep. Identical RL budget (8,000 gradient steps) in every arm.
- Evaluation: greedy rollout with visited nodes masked, `max_steps` = 300 (primary) and
  1000 (secondary).
- Stack: RTX 5080, `torch 2.11.0+cu128`, Python 3.11.15. The original environment
  (`qwarm-gnn-rl/.venv`) has since reverted to a CPU torch build, so a fresh
  `Thesis_Project/.venv` was built with the same versions of every shared package
  (checked with `uv pip freeze`; only dev tools differ). The replication check below tests
  whether this stack reproduces the published arms.

## 4. Tests

- **Primary:** strict goal-reach at 300 steps, (i) `warm_no_goaladj` vs `cold` (does the
  warm advantage survive without goal-adjacent transitions?) and (ii) `cold_goaladj` vs
  `cold` (do goal-adjacent transitions alone produce it?). Exact two-sided binomial McNemar
  test, Holm-corrected over these two comparisons, α = 0.05.
- **Secondary:** `persistent` vs `warm_no_goaladj` and `persistent` vs `cold_goaladj`
  (exact McNemar, uncorrected, reported as secondary); the same tests at 1000 steps; route
  cost on both-reached cells (Wilcoxon signed-rank, untestable when fewer than 6 pairs).
- **Replication check:** `persistent` and `cold` are compared cell by cell with the
  published masked 50×50 arms (`runs/sweep_50x50_masked_gpu/`, 22/25 against 9/25).
  A departure is reported first, before any ablation result is interpreted.
- **Power:** with 25 cells, p < 0.05 needs at least 6 discordant cells, all in one
  direction.

## 5. How each outcome will be reported (fixed now)

| Outcome | What the thesis says |
|---|---|
| **A.** (i) significant for `warm_no_goaladj`, (ii) not significant | The advantage is carried by the oracle demonstrations; the goal-adjacent transitions do not explain it at this power (a bound, not an equivalence). |
| **B.** (i) not significant, (ii) significant for `cold_goaladj` | The goal-adjacent transitions carry the advantage. Reported as a reversal of the attribution to demonstrations, in the same register as the action-space finding, with the dose caveat of §2. |
| **C.** both significant | Each component alone suffices to beat `cold`; the attribution is shared, and `persistent` vs each arm says which matters more. |
| **D.** neither significant | Neither component alone is shown to suffice; the advantage needs both, or the test is underpowered. Reported as such. |
| **E.** replication check fails | Reported first; the ablation is interpreted only against the new-stack `persistent` and `cold` arms. |

Route-cost results qualify the reach result and are reported beside it whatever their
direction. Nothing is dropped for being null.
