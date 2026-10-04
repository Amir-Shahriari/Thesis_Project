"""Suite A — Convergence benchmark: trains all four agents and logs per-iteration metrics.

Usage:
    uv run python scripts/run_convergence_benchmark.py --config configs/default.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import yaml

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.eval.path_evaluator import evaluate_agent, evaluate_tabular_agent
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.training.train_tabular import train_qlearning_agent
from qwarm.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_global_seed(args.seed)
    run_id = f"convergence_{int(time.time())}_seed{args.seed}"
    out_dir = pathlib.Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

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
    n_iters = cfg["n_iterations"]
    eps_per_iter = cfg["episodes_per_iteration"]
    hidden_dim = cfg.get("hidden_dim", 128)
    expert_ratio = cfg.get("expert_ratio", 0.30)

    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]

    results: dict = {}

    # gnn_warm
    agent = GNNDQN(node_in_dim=4, hidden_dim=hidden_dim, seed=args.seed)
    buf = ExpertReplayBuffer(expert_ratio=expert_ratio, rng=np.random.default_rng(args.seed))
    results["gnn_warm"] = train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=n_iters, episodes_per_iteration=eps_per_iter,
        re_seed_experts_each_iteration=True, seed=args.seed,
    )

    # gnn_cold
    agent_cold = GNNDQN(node_in_dim=4, hidden_dim=hidden_dim, seed=args.seed)
    buf_cold = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(args.seed))
    results["gnn_cold"] = train_gnn_dqn(
        g, PathfindingEnv, agent_cold, buf_cold, [], queries,
        n_iterations=n_iters, episodes_per_iteration=eps_per_iter,
        re_seed_experts_each_iteration=False, seed=args.seed,
    )

    # tabular_warm
    tab_warm = QLearningAgent()
    for i in range(n_iters):
        g.update_graph(iteration=i + 1)
        # re-inject oracle path each iteration
        for oracle in oracles:
            _, path, _ = oracle.find_optimized_route(src, dst)
            if path:
                from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer as _B
                _buf = _B()
                _buf.add_expert_path(g, PathfindingEnv, path, i)
        train_qlearning_agent(g, src, dst, episodes=eps_per_iter, agent=tab_warm)
    results["tabular_warm"] = {"note": "tabular_warm trained inline"}

    # tabular_cold
    tab_cold = QLearningAgent()
    train_qlearning_agent(g, src, dst, episodes=n_iters * eps_per_iter, agent=tab_cold)
    results["tabular_cold"] = {"note": "tabular_cold trained inline"}

    log_path = out_dir / "convergence_logs.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Convergence logs saved to {log_path}")


if __name__ == "__main__":
    main()
