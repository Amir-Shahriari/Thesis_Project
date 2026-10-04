"""Rebuild ExpertLibrary against a pre-trained agent's encoder.

Loads a checkpoint produced by the long-training run, builds the same
DynamicGraph used during training (seed=191664964, 100x100), runs three
oracles over 20 sampled (src, dst) pairs, and saves the populated library.

Unlike build_expert_library.py, no training is performed — the encoder is
already fully trained.

CLI:
    uv run python scripts/rebuild_library_from_agent.py \\
        --agent runs/agents_v1_long/warm_seed42_4x_compute.pt \\
        --out runs/libraries_long/seed_42_long.pt
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_library import ExpertLibrary

GRID_SEED = 191664964
GRID_TEMPLATE = {
    "grid_width": 100,
    "grid_height": 100,
    "extra_edges": 4,
    "deactivate_prob": 0.30,
}
NUM_PAIRS = 20
SAMPLE_SEED = 42
MIN_EUCLIDEAN_FRACTION = 0.4
EMBED_DIM = 128
HIDDEN_DIM = 128


def _load_checkpoint(agent: GNNDQN, path: pathlib.Path) -> None:
    ck = torch.load(str(path), map_location=agent.device, weights_only=False)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None


def _mc_returns(path: list[str], dyn_graph: DynamicGraph, gamma: float = 0.95) -> list[float]:
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
    ap = argparse.ArgumentParser(description="Rebuild ExpertLibrary from a pre-trained agent.")
    ap.add_argument("--agent", required=True, help="Path to the long-trained agent checkpoint.")
    ap.add_argument("--out", required=True, help="Output path for the library .pt file.")
    args = ap.parse_args()

    agent_path = pathlib.Path(args.agent)
    out_path = pathlib.Path(args.out)

    if not agent_path.exists():
        raise FileNotFoundError(f"Agent checkpoint not found: {agent_path}")

    print(f"\nrebuild_library_from_agent")
    print(f"  Agent      : {agent_path}")
    print(f"  Output     : {out_path}")
    print(f"  Grid seed  : {GRID_SEED}  ({GRID_TEMPLATE['grid_width']}x{GRID_TEMPLATE['grid_height']})")
    print(f"  Num pairs  : {NUM_PAIRS}")
    print()

    # ── 1. Build graph ─────────────────────────────────────────────────────────
    print("Building DynamicGraph...", flush=True)
    t0 = time.perf_counter()
    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=GRID_SEED,
    )
    print(f"  Graph built in {time.perf_counter() - t0:.1f}s  ({g.num_nodes} nodes)", flush=True)

    # ── 2. Build oracles ───────────────────────────────────────────────────────
    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
        FaithfulSimulatedQAOA(g.nodes, g.graph),
    ]
    print(f"  Oracles    : {[o.__class__.__name__ for o in oracles]}", flush=True)

    # ── 3. Sample (src, dst) pairs ─────────────────────────────────────────────
    rng = np.random.default_rng(SAMPLE_SEED)
    print(f"Sampling {NUM_PAIRS} scenario pairs (seed={SAMPLE_SEED}, min_euclidean={MIN_EUCLIDEAN_FRACTION})...", flush=True)
    scenarios = sample_scenarios(
        GRID_TEMPLATE,
        n_scenarios=NUM_PAIRS,
        rng=rng,
        min_euclidean_fraction=MIN_EUCLIDEAN_FRACTION,
    )
    queries = [(s.source_node, s.destination_node) for s in scenarios]
    print(f"  Sampled {len(queries)} pairs.", flush=True)

    # ── 4. Load long-trained agent ─────────────────────────────────────────────
    print("Loading long-trained agent...", flush=True)
    agent = GNNDQN(node_in_dim=4, hidden_dim=HIDDEN_DIM, seed=SAMPLE_SEED)
    _load_checkpoint(agent, agent_path)
    agent._encoder_raw.eval()
    print(f"  Loaded from {agent_path}", flush=True)

    # ── 5. Build PyG data object ───────────────────────────────────────────────
    data = dynamic_graph_to_pyg(g, device=agent.device)

    # ── 6. Collect expert paths and populate library ───────────────────────────
    library = ExpertLibrary(embed_dim=EMBED_DIM, max_size=5000, similarity="cosine")
    n_paths = 0
    n_hops = 0

    total_pairs = len(queries)
    for pair_idx, (src, dst) in enumerate(queries, 1):
        print(f"  [{pair_idx:>2}/{total_pairs}] {src} -> {dst}", end="", flush=True)
        pair_paths: set[tuple[str, ...]] = set()

        for oracle in oracles:
            t_oracle = time.perf_counter()
            try:
                cost, path, _ = oracle.find_optimized_route(src, dst)
            except Exception as exc:
                print(f"\n    {oracle.__class__.__name__} ERROR: {exc}", end="", flush=True)
                continue
            elapsed_o = time.perf_counter() - t_oracle
            if cost >= float("inf") or len(path) < 2:
                print(f"\n    {oracle.__class__.__name__}: no route ({elapsed_o:.1f}s)", end="", flush=True)
                continue
            path_key = tuple(path)
            if path_key in pair_paths:
                continue
            pair_paths.add(path_key)

            q_targets = _mc_returns(path, g)
            if not q_targets:
                continue
            library.add_path(agent._encoder_raw, data, path, q_targets)
            n_paths += 1
            n_hops += len(q_targets)
            print(f"\n    {oracle.__class__.__name__}: cost={cost:.1f} len={len(path)} ({elapsed_o:.1f}s)", end="", flush=True)

        print(flush=True)

    print(f"\nLibrary populated: {len(library)} entries from {n_paths} paths ({n_hops} hops total).")

    # ── 7. Save ────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    library.save(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
