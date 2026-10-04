"""Phase 4 — Multi-perturbation-seed robustness for Aim 3a (Dynamic Obstacle Test).

Trains warm and cold agents once on the median-seed scenario (seed=1337,
scenario 0), then evaluates both on 20 independently perturbed copies of the
trained graph. Each perturbed copy applies PERTURBATION_STEPS rounds of
update_graph with a fresh RNG so perturbations are structurally independent.

Gate P1 (aggregate, not per-seed): warm reasonable-reach count across the 20
perturbation seeds >= cold reasonable-reach count, AND warm reaches on >= 16
of the 20 seeds.

Usage:
    uv run python scripts/run_perturbation_robustness.py
    uv run python scripts/run_perturbation_robustness.py --out runs/perturb.json
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time

import numpy as np
import pandas as pd

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.utils.seeding import set_global_seed

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
MEDIAN_SEED = 1337           # median of [7, 42, 314159, 1337, 2024] sorted
PERTURBATION_SEEDS = list(range(20))
PERTURBATION_STEPS = 5


def _clone_and_perturb(base_graph: DynamicGraph, n_steps: int, seed: int) -> DynamicGraph:
    """Deep-copy base_graph and apply n_steps of update_graph with a fresh RNG."""
    g = copy.deepcopy(base_graph)
    g._rng = np.random.default_rng(seed)
    for i in range(n_steps):
        g.update_graph(iteration=i + 1)
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--median-seed", type=int, default=MEDIAN_SEED)
    parser.add_argument("--perturbation-steps", type=int, default=PERTURBATION_STEPS)
    parser.add_argument("--k-threshold", type=float, default=3.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print("\nPhase 4 — Perturbation Robustness")
    print(f"Training seed: {args.median_seed}  Perturbation steps: {args.perturbation_steps}")
    print(f"Perturbation seeds: 0..{len(PERTURBATION_SEEDS)-1}")

    # ── Train warm + cold on median-seed scenario ──────────────────────────────
    set_global_seed(args.median_seed)
    rng = np.random.default_rng(args.median_seed)
    scenarios = sample_scenarios(GRID_TEMPLATE, 1, rng, min_euclidean_fraction=0.6)
    scenario = scenarios[0]

    g_base = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    src = scenario.source_node
    dst = scenario.destination_node
    queries = [(src, dst)]
    oracles = [
        ClassicalAStar(g_base.nodes, g_base.graph),
        QuantumInspiredStochasticOracle(g_base.nodes, g_base.graph),
    ]

    print(f"\nScenario: {src} -> {dst}  (Euclidean dist={scenario.euclidean_distance:.1f})")
    print("Training warm agent...", flush=True)
    warm_agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=args.median_seed)
    warm_buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(args.median_seed),
    )
    t0 = time.perf_counter()
    train_gnn_dqn(
        g_base, PathfindingEnv, warm_agent, warm_buf, oracles, queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=args.median_seed,
    )
    print(f"  done in {time.perf_counter()-t0:.0f}s", flush=True)

    print("Training cold agent...", flush=True)
    cold_agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=args.median_seed)
    cold_buf = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(args.median_seed))
    t0 = time.perf_counter()
    train_gnn_dqn(
        g_base, PathfindingEnv, cold_agent, cold_buf, [], queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=False,
        seed=args.median_seed,
    )
    print(f"  done in {time.perf_counter()-t0:.0f}s\n", flush=True)

    # ── Evaluate on 20 perturbed copies ───────────────────────────────────────
    rows = []
    for pseed in PERTURBATION_SEEDS:
        g_perturbed = _clone_and_perturb(g_base, args.perturbation_steps, pseed)
        data = dynamic_graph_to_pyg(g_perturbed, device=warm_agent.device)

        warm_v = evaluate_with_reasonableness(
            g_perturbed, warm_agent, src, dst,
            k_threshold=args.k_threshold, data=data,
        )
        cold_v = evaluate_with_reasonableness(
            g_perturbed, cold_agent, src, dst,
            k_threshold=args.k_threshold, data=data,
        )

        warm_beats = warm_v.cost < cold_v.cost
        ratio = warm_v.cost / max(cold_v.cost, 1e-9)
        print(
            f"  pseed={pseed:>2}: warm={warm_v.cost:>8.1f} "
            f"cold={cold_v.cost:>8.1f}  "
            f"{'warm<cold' if warm_beats else 'cold<=warm'}  "
            f"ratio={ratio:.3f}  "
            f"w_reach={warm_v.reached_goal_reasonable}  "
            f"c_reach={cold_v.reached_goal_reasonable}",
            flush=True,
        )
        rows.append({
            "perturbation_seed": pseed,
            "warm_cost": warm_v.cost,
            "cold_cost": cold_v.cost,
            "warm_strict": warm_v.reached_goal_strict,
            "cold_strict": cold_v.reached_goal_strict,
            "warm_reasonable": warm_v.reached_goal_reasonable,
            "cold_reasonable": cold_v.reached_goal_reasonable,
            "cost_ratio": ratio,
            "dijkstra_cost": warm_v.dijkstra_reference_cost,
        })

    df = pd.DataFrame(rows)

    warm_reach = df["warm_reasonable"].sum()
    cold_reach = df["cold_reasonable"].sum()
    warm_beats_cold = (df["warm_cost"] < df["cold_cost"]).sum()
    mean_ratio = df[df["cost_ratio"] < float("inf")]["cost_ratio"].mean()

    p1_pass = warm_reach >= cold_reach and warm_reach >= 16

    print("\n" + "=" * 65)
    print("  PERTURBATION ROBUSTNESS — RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<48} {'Value':>12}")
    print(f"  {'-'*48} {'-'*12}")
    print(f"  {'Warm reasonable-reach rate (post-pert.)':<48} {warm_reach:>10}/20")
    print(f"  {'Cold reasonable-reach rate (post-pert.)':<48} {cold_reach:>10}/20")
    print(f"  {'% seeds where warm_cost < cold_cost':<48} {warm_beats_cold:>10}/20")
    print(f"  {'Mean cost ratio (warm/cold, pert.)':<48} {mean_ratio:>12.3f}")
    print("-" * 65)
    print(f"  Gate P1 (warm_reach >= cold_reach AND >= 16/20): "
          f"{'PASS' if p1_pass else 'FAIL'}  "
          f"(warm={warm_reach}/20, cold={cold_reach}/20)")
    print("=" * 65)

    out_path = pathlib.Path(args.out) if args.out else (
        pathlib.Path("runs") / f"perturbation_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
