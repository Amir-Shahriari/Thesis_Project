# 50x50 Sweep -- HEADLINE Protocol -- REPORT

Generated: 2026-08-10 21:39:48 UTC

## Headline Config Comparison (STEP 0 Verification)

Source of truth: run_multi_seed_warm_vs_cold.py + configs/default.yaml

| Parameter              | 100x100 Headline | 50x50 This Run | Match? |
|------------------------|-----------------|----------------|--------|
| batch_size             | 64              | 64             | YES    |
| n_iterations           | 10              | 10             | YES    |
| episodes_per_iteration | 200             | 200            | YES    |
| grad_steps_per_episode | 4               | 4              | YES    |
| hidden_dim             | 128             | 128            | YES    |
| expert_ratio           | 0.30            | 0.30           | YES    |
| pre_seed_n_states      | 3               | 3              | YES    |
| pre_seed_k_paths       | 3               | 3              | YES    |
| gamma                  | 0.95 (default)  | 0.95           | YES    |
| epsilon_start          | 0.40 (default)  | 0.40 (default) | YES    |
| epsilon_end            | 0.05 (default)  | 0.05 (default) | YES    |
| lr                     | 1e-4 (default)  | 1e-4 (default) | YES    |
| target_update_interval | 100 (default)   | 100 (default)  | YES    |
| grid_width/height      | 100x100         | 50x50          | NO (intentional) |
| extra_edges            | 4               | 3              | NO (declared 50x50 env) |
| deactivate_prob        | 0.30            | 0.22           | NO (declared 50x50 env) |
| node_deactivate_prob   | N/A             | 0.05           | NO (declared 50x50 env) |
| S4 fix                 | NOT applied     | Applied        | Disclosed below |

NOTE: The previous 50x50 run used batch_size=256 (a ~10x factor vs 64). This was
a protocol mismatch that broke comparability and caused ~4x slowdown. This run
corrects it to batch_size=64, matching the headline exactly.

Oracle disclosure: FaithfulSimulatedQAOA returns immediately for graphs with
>1000 nodes (faithful_qaoa.py:104). Both 50x50 (2500 nodes) and 100x100
(10000 nodes) exceed this threshold, so QAOA contributed ZERO paths at either
scale. The effective oracle pool is ClassicalAStar + QuantumInspiredStochasticOracle
at both scales. This is expected and matches the headline behavior.

## Configuration

Grid: 50x50, extra_edges=3, deactivate_prob=0.22, node_deactivate_prob=0.05
Training: n_iterations=10 (1x) / 40 (4x), episodes_per_iteration=200,
          grad_steps_per_episode=4, batch_size=64, hidden_dim=128,
          expert_ratio=0.30, pre_seed_n_states=3, pre_seed_k_paths=3
gamma: 0.95  |  dropout: 0.1 (default GraphSAGEEncoder)
Seeds: [42, 1337, 2024, 7, 314159]  |  Scenarios/seed: 5  |  Total cells: 25
Oracles: ClassicalAStar, QuantumInspiredStochasticOracle, FaithfulSimulatedQAOA
(Identical oracle pool to 100x100 HEADLINE -- no substitutions.)

## S4 Fix (Disclosed)

Applied: `self.target_encoder.eval()` added before computing target embeddings
in `learn_from_batch` (src/qwarm/agents/gnn_dqn.py, inside `with torch.no_grad()` block).

Root cause: target_encoder is a deepcopy of _encoder_raw, which starts in PyTorch
default train mode.  learn_from_batch calls `self._encoder_raw.train()` at the top,
but never explicitly sets target_encoder to eval mode.  With dropout=0.1, 10% of
target encoder activations were randomly zeroed during TD target computation,
injecting stochastic noise into target values and destabilizing learning.

Fix: `self.target_encoder.eval()` called before `t_emb = self.target_encoder(x, ei)`.
No other changes were made.

## Iteration 1 Verification

Buffer size after pre-seeding (first cell, seed=42):
  - From path library (discover_all_paths): 18
  - From goal-adjacent seeding:             7
  - Total pre-seed:                         25 transitions
Batch size threshold:                        64

Note: Pre-seeding (25) is below batch_size (64).
      This is expected on a 50x50 grid with pre_seed_n_states=3, pre_seed_k_paths=3.
      The buffer fills from online data within the first ~1 episodes of iter 1.
Estimated gradient steps in iteration 1: ~796
  (= max(0, 200 - 1) episodes
   x 4 grad steps/episode)

Training IS occurring: gradient steps begin in iteration 1 after ~1 episodes.
This is NOT a failure -- protocol fidelity is maintained.

## 1x Tier (10 iterations)

Cells completed:            25 / 25
Wall-clock total:           10.43 h
Wall-clock per cell:        1501 s
Avg warm grad steps/cell:   8000
Avg cold grad steps/cell:   8000

### Primary budget (max_steps=300)
  Win rate (warm < cold):          80.00%
  Warm strict reach rate:          88.00%  (95% CI: [70.04%, 95.83%])
  Cold strict reach rate:          36.00%
  Mean warm cost ratio:            14.0949
  Paired t-test:                   t=-3.122, p=0.0142
  McNemar p (reach):               0.0009

  Gates (evaluate only -- not engineered toward):
    M1_win_ge_80pct: PASS  value=0.8000  threshold=0.8
    M2_mean_cost_ratio_le_5: FAIL  value=14.0949  threshold=5.0
    M3_paired_p_lt_001: FAIL  value=0.0142  threshold=0.01

### Extended budget (max_steps=1000)
  Win rate (warm < cold):          80.00%
  Warm strict reach rate:          88.00%
  Cold strict reach rate:          44.00%
  Mean warm cost ratio:            14.0949
  Paired t-test:                   t=-3.263, p=0.0085

## 4x Tier

  Skipped: QWARM_SKIP_4X=1 set by the operator.

## Output Files

  runs\sweep_50x50_masked/sweep_v1_50x50_1x.json
  runs\sweep_50x50_masked/aggregate_50x50.json
  runs\sweep_50x50_masked/cells.json
  runs\sweep_50x50_masked/REPORT.md
