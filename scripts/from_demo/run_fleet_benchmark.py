"""Suite C -- Fleet benchmark script.

Usage:
    uv run python scripts/run_fleet_benchmark.py --seed 42 --queries 1000
    uv run python scripts/run_fleet_benchmark.py --seed 42 --grid-size 100 --repeats 5
"""
import argparse
import json
import pathlib
import time

import numpy as np

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.eval.fleet_benchmark import run_fleet_benchmark
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.utils.device import resolve_device
from qwarm.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Times to repeat GNN timing for mean+/-std")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / cpu / auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_global_seed(args.seed)

    out_dir = pathlib.Path("runs") / f"fleet_{int(time.time())}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        import torch
        print(f"GPU:    {torch.cuda.get_device_name(device)}")

    g = DynamicGraph(
        args.grid_size, args.grid_size,
        extra_edges=4, deactivate_prob=0.30, seed=args.seed,
    )
    src, dst = "Node_1", f"Node_{g.num_nodes}"
    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]

    agent = GNNDQN(node_in_dim=4, hidden_dim=128, seed=args.seed, device=str(device))
    buf = ExpertReplayBuffer(expert_ratio=0.30, rng=np.random.default_rng(args.seed))
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, [(src, dst)],
        n_iterations=5, episodes_per_iteration=100,
        re_seed_experts_each_iteration=True, seed=args.seed,
    )

    result = run_fleet_benchmark(
        g, agent, n_queries=args.queries, seed=args.seed, n_repeats=args.repeats
    )

    out_path = out_dir / "fleet_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    ratio = result["throughput_ratio"]
    gnn_ms = result["gnn_per_query_ms"]
    gnn_std = result["gnn_per_query_std_ms"]
    astar_ms = result["astar_per_query_ms"]
    g5 = gnn_ms / max(astar_ms, 1e-9)
    gpu2_ms = result["gnn_total_s"] * 1000

    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  Grid:        {args.grid_size}x{args.grid_size}  ({g.num_nodes} nodes)")
    print(f"  Queries:     {result['n_queries']}  (repeats={result['n_repeats']})")
    print(f"  A* total:    {result['astar_total_s']:.3f} s   per-query: {astar_ms:.3f} ms")
    print(f"  GNN total:   {result['gnn_total_s']:.3f} s   per-query: {gnn_ms:.4f} +/- {gnn_std:.4f} ms")
    print(f"  Throughput:  {ratio:.1f}x  (A*/GNN)")
    print(f"  G5 (GNN/A*): {g5:.5f}  {'PASS' if g5 <= 0.02 else 'FAIL'} (<=0.02)")
    # GPU2 fleet-mode: trained agent on perturbed graph — longer paths than gate.
    # Authoritative gate (untrained agent, fresh graph, max_steps=150) lives in
    # tests/test_batch_infer.py::test_batch_infer_gpu2_1000q_under_100ms.
    print(f"  GPU2 (fleet-mode):     {gpu2_ms:.1f} ms  (gate: test_batch_infer, max_steps=150)")
    print(f"  GPU3 (>=50x speedup):  {ratio:.1f}x  {'PASS' if ratio >= 50 else 'FAIL'}")
    print(sep)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
