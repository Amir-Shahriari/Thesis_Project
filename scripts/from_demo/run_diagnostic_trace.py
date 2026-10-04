"""Phase 1 diagnostic — capture per-step training trace for one warm cell.

Runs the warm agent on seed=42, scenario_idx=4 (the cell where cold won
decisively: warm=870, cold=28 in the prior sweep). Writes a full JSONL
trace to runs/diag_seed42_sc4/training_trace.jsonl and prints:
  - first 50 gradient steps (all 15 fields)
  - expert_fraction_in_batch time series (all steps)
  - q_value_max_abs time series (all steps)
  - per-iteration goal_reach_rate from the training logs
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.utils.seeding import set_global_seed

SEED = 42
SCENARIO_IDX = 4  # 0-based; matches "Cell 4" in the prior sweep output
TRACE_DIR = pathlib.Path("runs/diag_seed42_sc4")

GRID_TEMPLATE = {
    "grid_width": 25,
    "grid_height": 25,
    "extra_edges": 2,
    "deactivate_prob": 0.15,
}
TRAIN_CFG = {
    "n_iterations": 5,
    "episodes_per_iteration": 100,
    "grad_steps_per_episode": 2,
    "batch_size": 64,
    "hidden_dim": 64,
    "expert_ratio": 0.40,
}


def main() -> None:
    set_global_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Reproduce the same scenario list as the sweep
    scenarios = sample_scenarios(
        GRID_TEMPLATE, n_scenarios=max(SCENARIO_IDX + 1, 5), rng=rng,
        min_euclidean_fraction=0.6,
    )
    scenario = scenarios[SCENARIO_IDX]
    src = scenario.source_node
    dst = scenario.destination_node

    print(f"[diag] seed={SEED}  scenario_idx={SCENARIO_IDX}")
    print(f"[diag] {src} -> {dst}  (dist={scenario.euclidean_distance:.1f})")
    print(f"[diag] trace -> {TRACE_DIR}/training_trace.jsonl\n")

    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    queries = [(src, dst)]
    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]

    agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=SEED)
    buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(SEED),
    )

    logs = train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=SEED,
        trace_dir=TRACE_DIR,
    )

    # ── Per-iteration summary ─────────────────────────────────────────────────
    print("\n=== Per-iteration training log ===")
    print(f"{'iter':>4}  {'goal_reach':>10}  {'mean_return':>12}  {'mean_loss':>10}  "
          f"{'exp_pool':>8}  {'onl_pool':>8}")
    for i in range(len(logs["iteration"])):
        print(f"{logs['iteration'][i]:>4}  "
              f"{logs['goal_reach_rate'][i]:>10.3f}  "
              f"{logs['mean_return'][i]:>12.1f}  "
              f"{logs['mean_loss'][i]:>10.4f}  "
              f"{logs['expert_pool_size'][i]:>8}  "
              f"{logs['online_pool_size'][i]:>8}")

    # ── Read trace file ───────────────────────────────────────────────────────
    trace_path = TRACE_DIR / "training_trace.jsonl"
    if not trace_path.exists():
        print("\n[ERROR] Trace file not written — check open_trace path.")
        sys.exit(1)

    steps = []
    with open(trace_path) as f:
        for line in f:
            steps.append(json.loads(line.strip()))

    print(f"\nTotal gradient steps logged: {len(steps)}")

    # ── First 50 steps — all 15 fields ───────────────────────────────────────
    print("\n=== First 50 gradient steps ===")
    header = (
        f"{'step':>4}  {'iter':>4}  {'exp_frac':>8}  {'loss':>10}  "
        f"{'loss_exp':>10}  {'loss_onl':>10}  "
        f"{'td_mean':>8}  {'td_max':>8}  "
        f"{'q_mean':>8}  {'q_std':>7}  {'q_maxabs':>8}  "
        f"{'gn_raw':>8}  {'gn_clip':>8}  "
        f"{'exp_buf':>7}  {'onl_buf':>7}"
    )
    print(header)
    print("-" * len(header))

    def _f(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "   None"

    for s in steps[:50]:
        print(
            f"{s['step']:>4}  {s['iteration']:>4}  "
            f"{_f(s['expert_fraction_in_batch'], '8.3f')}  "
            f"{_f(s['loss'], '10.2f')}  "
            f"{_f(s['loss_expert'], '10.2f')}  "
            f"{_f(s['loss_online'], '10.2f')}  "
            f"{_f(s['td_error_mean_abs'], '8.2f')}  "
            f"{_f(s['td_error_max_abs'], '8.2f')}  "
            f"{_f(s['q_value_mean'], '8.2f')}  "
            f"{_f(s['q_value_std'], '7.2f')}  "
            f"{_f(s['q_value_max_abs'], '8.2f')}  "
            f"{_f(s['grad_norm'], '8.4f')}  "
            f"{_f(s['grad_norm_clipped'], '8.4f')}  "
            f"{s['expert_buffer_size']:>7}  "
            f"{s['online_buffer_size']:>7}"
        )

    # ── Key time series: expert_fraction and q_value_max_abs ─────────────────
    print("\n=== expert_fraction_in_batch — full trace ===")
    fracs = [s["expert_fraction_in_batch"] for s in steps]
    iters = [s["iteration"] for s in steps]
    for iter_id in sorted(set(iters)):
        chunk = [f for s, f in zip(steps, fracs) if s["iteration"] == iter_id]
        arr = np.array(chunk)
        print(f"  iter={iter_id}: min={arr.min():.3f}  mean={arr.mean():.3f}  "
              f"max={arr.max():.3f}  n={len(arr)}")

    print("\n=== q_value_max_abs — full trace ===")
    qmaxs = [s["q_value_max_abs"] for s in steps if s["q_value_max_abs"] is not None]
    for iter_id in sorted(set(iters)):
        chunk = [s["q_value_max_abs"] for s in steps
                 if s["iteration"] == iter_id and s["q_value_max_abs"] is not None]
        if chunk:
            arr = np.array(chunk)
            print(f"  iter={iter_id}: min={arr.min():.2f}  mean={arr.mean():.2f}  "
                  f"max={arr.max():.2f}")

    print("\n=== grad_norm (pre-clip) — full trace ===")
    for iter_id in sorted(set(iters)):
        chunk = [s["grad_norm"] for s in steps if s["iteration"] == iter_id]
        if chunk:
            arr = np.array(chunk)
            print(f"  iter={iter_id}: min={arr.min():.4f}  mean={arr.mean():.4f}  "
                  f"max={arr.max():.4f}")

    print(f"\n[diag] Complete. Trace saved to {trace_path}")


if __name__ == "__main__":
    main()
