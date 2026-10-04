"""Build ExpertLibrary for each of the 6 V1-realtime cells.

Loads the static-trained V1 agent for each cell, builds the matching
DynamicGraph (25x25, grid_seed from the test grid JSON), runs three
oracles on the cell's canonical (src, dst) plus 5 extra sampled pairs,
and saves the populated library to runs/libraries_realtime/.

Usage:
    uv run python scripts/build_realtime_libraries.py
    # or
    python scripts/build_realtime_libraries.py
"""
from __future__ import annotations

import json
import pathlib
import random
import time
from collections import deque

import numpy as np
import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_library import ExpertLibrary

_QAOA_AVAILABLE = False  # disabled: too slow (~22 min/cell) vs ClassicalAStar+QIS (~2 min/cell)

AGENTS_DIR = pathlib.Path("runs/agents_v1_realtime_v2")
TEST_GRID_JSON = pathlib.Path("runs/v1_realtime_test_grid_v2.json")
OUT_DIR = pathlib.Path("runs/libraries_realtime")
GAMMA = 0.95
N_EXTRA_PAIRS = 5
SAMPLE_SEED_BASE = 99991  # reproducible per cell


def _load_static_agent(cell_idx: int, outer_seed: int) -> tuple[GNNDQN, int]:
    path = AGENTS_DIR / f"static_seed{outer_seed}_scen{cell_idx}.pt"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    hidden_dim: int = ck.get("hidden_dim", 64)
    node_in_dim: int = ck.get("node_in_dim", 4)
    agent = GNNDQN(node_in_dim=node_in_dim, hidden_dim=hidden_dim, seed=0)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    agent._cached_embeddings = None
    agent._cached_data_id = None
    return agent, hidden_dim


def _is_reachable_on(g: DynamicGraph, src: str, dst: str) -> bool:
    if src == dst:
        return False
    visited: set[str] = {src}
    q: deque[str] = deque([src])
    while q:
        node = q.popleft()
        for nb, edata in g.graph[node].items():
            if nb not in visited and edata["active"] and g.nodes[nb]["active"]:
                if nb == dst:
                    return True
                visited.add(nb)
                q.append(nb)
    return False


def _sample_extra_pairs(
    g: DynamicGraph,
    n: int,
    seed: int,
    exclude: tuple[str, str],
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    node_ids = [nid for nid, nd in g.nodes.items() if nd["active"]]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = {exclude}
    attempts = 0
    while len(pairs) < n and attempts < 2000:
        attempts += 1
        src = rng.choice(node_ids)
        dst = rng.choice(node_ids)
        if (src, dst) in seen or src == dst:
            continue
        if _is_reachable_on(g, src, dst):
            pairs.append((src, dst))
            seen.add((src, dst))
    return pairs


def _mc_returns(path: list[str], g: DynamicGraph) -> list[float]:
    if len(path) < 2:
        return []
    env = PathfindingEnv(
        g.graph, g.nodes, path[0], path[-1], max_steps=9999, lambda_shape=0.0
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
    mc = [0.0] * T
    mc[T - 1] = steps[T - 1][2]
    for t in range(T - 2, -1, -1):
        mc[t] = steps[t][2] + GAMMA * mc[t + 1]
    return mc


def build_cell_library(cell: dict, wall_t0: float, skip_existing: bool = True) -> pathlib.Path | None:
    cell_idx = cell["cell_idx"]
    outer_seed = cell["outer_seed"]
    grid_seed = cell["grid_seed"]
    scenario_id = cell["scenario_id"]
    src_node = cell["source"]
    dst_node = cell["destination"]
    grid_cfg = cell["grid"]

    out_path = OUT_DIR / f"cell{cell_idx}_seed{outer_seed}_scen{scenario_id}.pt"
    if skip_existing and out_path.exists():
        print(f"\nSkipping cell {cell_idx} — library already exists: {out_path}")
        return out_path
    print(f"\n{'='*60}")
    print(f"Cell {cell_idx} | outer_seed={outer_seed} | grid_seed={grid_seed}")
    print(f"  src={src_node}  dst={dst_node}")
    print(f"  out={out_path}")
    t_cell = time.perf_counter()

    # 1. Load static agent
    print("  [1] Loading static agent...", flush=True)
    agent, hidden_dim = _load_static_agent(cell_idx, outer_seed)
    print(f"      hidden_dim={hidden_dim}", flush=True)

    # 2. Build DynamicGraph (no perturbation steps — static trained)
    print("  [2] Building DynamicGraph...", flush=True)
    g = DynamicGraph(
        grid_width=grid_cfg["grid_width"],
        grid_height=grid_cfg["grid_height"],
        extra_edges=grid_cfg.get("extra_edges", 2),
        deactivate_prob=grid_cfg.get("deactivate_prob", 0.10),
        seed=grid_seed,
    )
    print(f"      {g.num_nodes} nodes", flush=True)

    # 3. Build oracles
    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]
    if _QAOA_AVAILABLE:
        oracles.append(FaithfulSimulatedQAOA(g.nodes, g.graph))
    print(f"  [3] Oracles: {[o.__class__.__name__ for o in oracles]}", flush=True)

    # 4. Build PyG data
    data = dynamic_graph_to_pyg(g, device=agent.device)

    # 5. Collect query pairs: canonical + 5 extra
    extra_pairs = _sample_extra_pairs(
        g, N_EXTRA_PAIRS, seed=SAMPLE_SEED_BASE + cell_idx, exclude=(src_node, dst_node)
    )
    all_pairs = [(src_node, dst_node)] + extra_pairs
    print(f"  [4] Query pairs: 1 canonical + {len(extra_pairs)} extra = {len(all_pairs)} total", flush=True)

    # 6. Populate library
    library = ExpertLibrary(embed_dim=hidden_dim, max_size=5000, similarity="cosine")
    n_paths = 0
    n_hops = 0

    for pidx, (s, d) in enumerate(all_pairs):
        label = "canonical" if pidx == 0 else f"extra-{pidx}"
        print(f"  [{label}] {s} -> {d}", flush=True)
        pair_paths: set[tuple[str, ...]] = set()

        for oracle in oracles:
            try:
                cost, path, _ = oracle.find_optimized_route(s, d)
            except Exception as exc:
                print(f"    {oracle.__class__.__name__} ERROR: {exc}", flush=True)
                continue
            if cost >= float("inf") or len(path) < 2:
                continue
            key = tuple(path)
            if key in pair_paths:
                continue
            pair_paths.add(key)
            mc = _mc_returns(path, g)
            if not mc:
                continue
            library.add_path(agent._encoder_raw, data, path, mc)
            n_paths += 1
            n_hops += len(mc)
            print(f"    {oracle.__class__.__name__}: cost={cost:.1f} len={len(path)} hops={len(mc)}", flush=True)

    print(f"  Library: {len(library)} entries, {n_paths} paths, {n_hops} hops", flush=True)

    # 7. Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    library.save(str(out_path))
    elapsed = time.perf_counter() - t_cell
    wall_elapsed = time.perf_counter() - wall_t0
    print(f"  Saved: {out_path}  (cell time={elapsed:.1f}s  wall={wall_elapsed:.1f}s)", flush=True)
    return out_path


def main() -> None:
    cells = json.loads(TEST_GRID_JSON.read_text())
    print(f"Building libraries for {len(cells)} cells")
    print(f"QAOA available: {_QAOA_AVAILABLE}")
    wall_t0 = time.perf_counter()

    saved: list[pathlib.Path] = []
    for cell in cells:
        p = build_cell_library(cell, wall_t0, skip_existing=True)
        if p is not None:
            saved.append(p)

    total = time.perf_counter() - wall_t0
    print(f"\n{'='*60}")
    print(f"Done. Total wall-clock: {total:.1f}s")
    print(f"Saved {len(saved)} libraries:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
