"""Build an ExpertLibrary for a given seed.

Trains a brief warm agent on the graph for that seed, collects expert paths from
classical oracles, computes MC Q-targets, embeds the paths using the trained
encoder, and saves the populated ExpertLibrary to disk.

Usage:
    uv run python scripts/build_expert_library.py \\
        --seed 42 --num-pairs 10 --out runs/libraries/seed_42.pt

    uv run python scripts/build_expert_library.py \\
        --seed 42 --num-pairs 20 --config configs/small.yaml \\
        --out runs/libraries/seed_42.pt

The library is then loaded by:
    run_multi_seed_warm_vs_cold.py --use-library --lambda-retr 0.5

Note: embeddings are frozen to this training run's encoder. For highest
retrieval quality, rebuild the library after running the headline sweep
with a fully-trained checkpoint.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.replay.expert_library import ExpertLibrary
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.utils.seeding import set_global_seed

# Defaults match configs/small.yaml for fast smoke builds
_GRID = {
    "grid_width": 25,
    "grid_height": 25,
    "extra_edges": 2,
    "deactivate_prob": 0.10,
}
_TRAIN = {
    "n_iterations": 2,
    "episodes_per_iteration": 20,
    "grad_steps_per_episode": 4,
    "batch_size": 64,
    "hidden_dim": 64,
    "expert_ratio": 0.40,
    "pre_seed_n_states": 3,
    "pre_seed_k_paths": 5,
}


def _apply_config(path: str) -> None:
    global _GRID, _TRAIN
    if not _HAS_YAML:
        raise RuntimeError("PyYAML not installed; run: uv add pyyaml")
    with open(path) as fh:
        cfg = _yaml.safe_load(fh)
    grid_keys = {"grid_width", "grid_height", "extra_edges", "deactivate_prob"}
    train_keys = {"n_iterations", "episodes_per_iteration", "hidden_dim", "expert_ratio",
                  "grad_steps_per_episode", "batch_size", "pre_seed_n_states", "pre_seed_k_paths"}
    for k, v in cfg.items():
        if k in grid_keys:
            _GRID[k] = v
        elif k in train_keys:
            _TRAIN[k] = v


def _mc_returns(path: list[str], dyn_graph, gamma: float = 0.95) -> list[float]:
    """Compute MC Q-targets for each hop in the path using PathfindingEnv."""
    if len(path) < 2:
        return []
    env = PathfindingEnv(
        dyn_graph.graph, dyn_graph.nodes, path[0], path[-1], max_steps=9999, lambda_shape=0.0
    )
    env.reset()
    steps: list[tuple[str, str, float]] = []
    for s, a in zip(path[:-1], path[1:]):
        env.current_node = s
        env.visited_nodes = {s}
        _, reward, done = env.step(a)
        if reward == -5.0:
            break
        steps.append((s, a, reward))
        if done:
            break
    if not steps:
        return []
    T = len(steps)
    mc: list[float] = [0.0] * T
    mc[T - 1] = steps[T - 1][2]
    for t in range(T - 2, -1, -1):
        mc[t] = steps[t][2] + gamma * mc[t + 1]
    return mc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ExpertLibrary for one seed.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-pairs", type=int, default=10,
                        help="Number of (source, dest) pairs to collect expert paths for.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output path for the library (e.g. runs/libraries/seed_42.pt).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config to override grid/train defaults.")
    parser.add_argument("--lambda-retr", type=float, default=0.5,
                        help="Retrieval weight (stored as metadata only; used at inference).")
    args = parser.parse_args()

    if args.config:
        _apply_config(args.config)

    set_global_seed(args.seed)
    print(f"\nbuild_expert_library: seed={args.seed}  num-pairs={args.num_pairs}")
    print(f"  Grid: {_GRID}")
    print(f"  Train: {_TRAIN}")

    # ── 1. Build graph ────────────────────────────────────────────────────────
    g = DynamicGraph(
        grid_width=_GRID["grid_width"],
        grid_height=_GRID["grid_height"],
        extra_edges=_GRID["extra_edges"],
        deactivate_prob=_GRID["deactivate_prob"],
        seed=args.seed,
    )
    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]

    # ── 2. Sample (source, dest) pairs ───────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    scenarios = sample_scenarios(
        _GRID,
        n_scenarios=args.num_pairs,
        rng=rng,
        min_euclidean_fraction=0.4,
    )
    queries = [(s.source_node, s.destination_node) for s in scenarios]
    print(f"  Sampled {len(queries)} (source, dest) pairs.")

    # ── 3. Train a brief agent to get a meaningful encoder ───────────────────
    print("  [training brief agent for encoder...]", flush=True)
    t0 = time.perf_counter()
    agent = GNNDQN(
        node_in_dim=4,
        hidden_dim=_TRAIN["hidden_dim"],
        seed=args.seed,
    )
    buf = ExpertReplayBuffer(
        expert_ratio=_TRAIN["expert_ratio"],
        rng=np.random.default_rng(args.seed),
    )
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=_TRAIN["n_iterations"],
        episodes_per_iteration=_TRAIN["episodes_per_iteration"],
        grad_steps_per_episode=_TRAIN["grad_steps_per_episode"],
        batch_size=_TRAIN["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=args.seed,
        pre_seed_n_states=_TRAIN["pre_seed_n_states"],
        pre_seed_k_paths=_TRAIN["pre_seed_k_paths"],
    )
    print(f"  [training done in {time.perf_counter() - t0:.0f}s]", flush=True)

    data = dynamic_graph_to_pyg(g, device=agent.device)
    agent._encoder_raw.eval()

    # ── 4. Collect expert paths and build library ─────────────────────────────
    library = ExpertLibrary(
        embed_dim=_TRAIN["hidden_dim"],
        max_size=5000,
        similarity="cosine",
    )
    n_added = 0
    for src, dst in queries:
        for oracle in oracles:
            cost, path, _ = oracle.find_optimized_route(src, dst)
            if cost >= float("inf") or len(path) < 2:
                continue
            q_targets = _mc_returns(path, g)
            if not q_targets:
                continue
            library.add_path(agent._encoder_raw, data, path, q_targets)
            n_added += len(q_targets)

    print(f"  Library populated: {len(library)} entries from {n_added} path hops.")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    library.save(str(out_path))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
