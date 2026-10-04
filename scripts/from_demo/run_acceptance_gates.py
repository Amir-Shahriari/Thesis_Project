"""Run all 6 acceptance gates on a seeded configuration and exit 1 if any fail.

Usage:
    uv run python scripts/run_acceptance_gates.py --config configs/default.yaml --seed 42
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import yaml

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.fleet_benchmark import run_fleet_benchmark
from qwarm.eval.path_evaluator import evaluate_agent
from qwarm.eval.transfer_benchmark import run_transfer_benchmark
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.classical_dijkstra import ClassicalDijkstra
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.training.train_tabular import train_qlearning_agent
from qwarm.utils.seeding import set_global_seed


def _iters_to_90(logs: dict) -> int:
    for i, rate in enumerate(logs["goal_reach_rate"]):
        if rate >= 0.90:
            return i + 1
    return len(logs["goal_reach_rate"]) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_global_seed(args.seed)
    n_iters = cfg["n_iterations"]
    eps_per_iter = cfg["episodes_per_iteration"]
    hidden_dim = cfg.get("hidden_dim", 128)
    expert_ratio = cfg.get("expert_ratio", 0.30)

    print(f"\nRunning acceptance gates  config={args.config}  seed={args.seed}")
    print("=" * 60)

    g = DynamicGraph(
        grid_width=cfg["grid_width"],
        grid_height=cfg["grid_height"],
        extra_edges=cfg["extra_edges"],
        deactivate_prob=cfg["deactivate_prob"],
        seed=args.seed,
    )
    src = "Node_1"
    dst = f"Node_{g.num_nodes}"
    queries = [(src, dst)]

    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        ClassicalDijkstra(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]

    # G1 — oracles can route
    oracle_hits = [
        float((oracle.find_optimized_route(src, dst)[0]) < float("inf"))
        for oracle in oracles
    ]
    g1_val = float(np.mean(oracle_hits))
    g1_pass = g1_val >= 0.90

    # Train warm and cold GNN
    gnn_warm = GNNDQN(node_in_dim=4, hidden_dim=hidden_dim, seed=args.seed)
    warm_buf = ExpertReplayBuffer(expert_ratio=expert_ratio, rng=np.random.default_rng(args.seed))
    warm_logs = train_gnn_dqn(
        g, PathfindingEnv, gnn_warm, warm_buf, oracles, queries,
        n_iterations=n_iters, episodes_per_iteration=eps_per_iter,
        re_seed_experts_each_iteration=True, seed=args.seed,
    )

    gnn_cold = GNNDQN(node_in_dim=4, hidden_dim=hidden_dim, seed=args.seed)
    cold_buf = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(args.seed))
    cold_logs = train_gnn_dqn(
        g, PathfindingEnv, gnn_cold, cold_buf, [], queries,
        n_iterations=n_iters, episodes_per_iteration=eps_per_iter,
        re_seed_experts_each_iteration=False, seed=args.seed,
    )

    data = dynamic_graph_to_pyg(g)
    warm_eval = evaluate_agent(gnn_warm, g, src, dst, data)
    cold_eval = evaluate_agent(gnn_cold, g, src, dst, data)

    # G2 — warm cost strictly < cold
    if cold_eval["cost"] < float("inf") and warm_eval["cost"] < float("inf"):
        g2_val = warm_eval["cost"] / cold_eval["cost"]
    elif not cold_eval["reached_goal"] and warm_eval["reached_goal"]:
        g2_val = 0.0  # warm reached, cold didn't — warm wins
    else:
        g2_val = 1.0
    g2_pass = g2_val <= 0.95

    # G3 — warm goal-reach rate
    g3_val = warm_logs["goal_reach_rate"][-1]
    g3_pass = g3_val >= 0.90

    # G6 — iterations to 90 %
    warm_i90 = _iters_to_90(warm_logs)
    cold_i90 = _iters_to_90(cold_logs)
    g6_val = warm_i90 / max(cold_i90, 1)
    g6_pass = g6_val <= 0.50

    # G4 — transfer
    test_cfgs = [
        {"grid_width": 50, "grid_height": 50, "extra_edges": 2, "seed": 99},
        {"grid_width": 75, "grid_height": 75, "extra_edges": 3, "seed": 99},
        {"grid_width": 60, "grid_height": 40, "extra_edges": 2, "seed": 99},
    ]
    transfer = run_transfer_benchmark(
        {"grid_width": 50, "grid_height": 50, "extra_edges": 2, "seed": args.seed},
        test_cfgs,
        n_train_iterations=n_iters,
        episodes_per_iteration=eps_per_iter,
        seed=args.seed,
    )
    g4_ratios = []
    for gk in transfer["gnn_warm"]:
        gnn_rate = transfer["gnn_warm"][gk]["goal_reach_rate"]
        tab_rate = transfer["tabular_warm"][gk]["goal_reach_rate"]
        g4_ratios.append(gnn_rate / max(tab_rate, 0.01))
    g4_val = float(min(g4_ratios)) if g4_ratios else 0.0
    g4_pass = g4_val >= 2.0

    # G5 — fleet throughput
    fleet = run_fleet_benchmark(g, gnn_warm, n_queries=1000, seed=args.seed)
    g5_val = fleet["gnn_per_query_ms"] / max(fleet["astar_per_query_ms"], 1e-9)
    g5_pass = g5_val <= 0.02

    gate_rows = [
        ("G1", g1_val, "≥ 0.90", g1_pass),
        ("G2", g2_val, "≤ 0.95 (warm/cold cost ratio)", g2_pass),
        ("G3", g3_val, "≥ 0.90", g3_pass),
        ("G4", g4_val, "≥ 2.0 (GNN/tabular goal-reach ratio)", g4_pass),
        ("G5", g5_val, "≤ 0.02 (GNN/A* per-query time ratio)", g5_pass),
        ("G6", g6_val, "≤ 0.50 (warm/cold iters-to-90% ratio)", g6_pass),
    ]

    all_pass = True
    for gid, val, threshold_desc, passed in gate_rows:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {gid}: {status}  measured={val:.4f}  threshold={threshold_desc}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL 6 GATES PASSED")
    else:
        print("ONE OR MORE GATES FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
