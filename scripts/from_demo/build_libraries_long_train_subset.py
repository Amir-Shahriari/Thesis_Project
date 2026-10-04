"""Build ExpertLibrary per cell against long-trained V1 encoders (n=15 experiment).

For each of the 15 cells in runs/v1_long_train_subset_results.json:
  - Skip cells where warm_strict=False (4x training failed to reach goal)
  - Load the long-trained V1 agent from runs/agents_v1_long/warm_seed{seed}_scen{scenario_id}_4x.pt
  - Build DynamicGraph (100x100, extra_edges=4, deactivate_prob=0.30, grid_seed from scenario_id)
  - Apply 5 perturbation steps to approximate the trained-final graph state
  - Run ClassicalAStar + QuantumInspiredStochastic on:
      * canonical (source, destination) from the cell record
      * 5 extra sampled (src, dst) pairs (euclidean >= 0.4 * diagonal, cell seed for determinism)
  - Embed path hops with the loaded agent's _encoder_raw and populate ExpertLibrary
  - Save to runs/libraries_long/cell{idx:02d}_seed{seed}_scen{scenario_id}.pt

Skip cells whose library file already exists (resume-friendly).

CLI:
    uv run python scripts/build_libraries_long_train_subset.py
"""
from __future__ import annotations

import json
import math
import pathlib
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

SUBSET_JSON   = pathlib.Path("runs/v1_long_train_subset_results.json")
AGENTS_DIR    = pathlib.Path("runs/agents_v1_long")
OUT_DIR       = pathlib.Path("runs/libraries_long")
PERTURBATION_STEPS = 5
GAMMA          = 0.95
N_EXTRA_PAIRS  = 5
MIN_EUCLIDEAN_FRACTION = 0.4
HIDDEN_DIM     = 128


def _parse_grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _load_agent(seed: int, scenario_id: str) -> GNNDQN:
    path = AGENTS_DIR / f"warm_seed{seed}_scen{scenario_id}_4x.pt"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    hidden_dim = ck.get("hidden_dim", HIDDEN_DIM)
    node_in_dim = ck.get("node_in_dim", 4)
    agent = GNNDQN(node_in_dim=node_in_dim, hidden_dim=hidden_dim, seed=seed)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    agent._cached_embeddings = None
    agent._cached_data_id = None
    return agent


def _build_graph(grid_seed: int) -> DynamicGraph:
    g = DynamicGraph(
        grid_width=100,
        grid_height=100,
        extra_edges=4,
        deactivate_prob=0.30,
        seed=grid_seed,
    )
    for _ in range(PERTURBATION_STEPS):
        g.update_graph()
    return g


def _is_reachable(g: DynamicGraph, src: str, dst: str) -> bool:
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
    cell_seed: int,
    exclude: tuple[str, str],
) -> list[tuple[str, str]]:
    """Sample n (src, dst) pairs with euclidean >= 0.4 * diagonal and BFS reachability."""
    diagonal = math.sqrt((g.grid_width - 1) ** 2 + (g.grid_height - 1) ** 2)
    min_dist = MIN_EUCLIDEAN_FRACTION * diagonal
    rng = np.random.default_rng(cell_seed)
    node_ids = [nid for nid, nd in g.nodes.items() if nd["active"]]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = {exclude}
    attempts = 0
    while len(pairs) < n and attempts < 3000:
        attempts += 1
        src = str(rng.choice(node_ids))
        dst = str(rng.choice(node_ids))
        if (src, dst) in seen or src == dst:
            continue
        cx, cy = g.nodes[src]["coords"]
        dx, dy = g.nodes[dst]["coords"]
        dist = math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
        if dist < min_dist:
            continue
        if not _is_reachable(g, src, dst):
            continue
        pairs.append((src, dst))
        seen.add((src, dst))
    if len(pairs) < n:
        print(f"    [warn] only found {len(pairs)}/{n} extra pairs after {attempts} attempts")
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


def build_cell_library(cell_idx: int, cell: dict, wall_t0: float) -> pathlib.Path | None:
    seed        = cell["seed"]
    scenario_id = cell["scenario_id"]
    src_node    = cell["source"]
    dst_node    = cell["destination"]

    out_path = OUT_DIR / f"cell{cell_idx:02d}_seed{seed}_scen{scenario_id}.pt"
    if out_path.exists():
        print(f"\nSkipping cell {cell_idx:02d} — library already exists: {out_path}")
        return out_path

    print(f"\n{'='*60}")
    print(f"Cell {cell_idx:02d} | seed={seed} | scenario={scenario_id}")
    print(f"  src={src_node}  dst={dst_node}")
    print(f"  out={out_path}")
    t_cell = time.perf_counter()

    print("  [1] Loading long-trained agent...", flush=True)
    agent = _load_agent(seed, scenario_id)
    print(f"      hidden_dim={HIDDEN_DIM}", flush=True)

    grid_seed = _parse_grid_seed(scenario_id)
    print(f"  [2] Building DynamicGraph (grid_seed={grid_seed}, {PERTURBATION_STEPS} perturbation steps)...", flush=True)
    g = _build_graph(grid_seed)
    print(f"      {g.num_nodes} nodes", flush=True)

    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]
    print(f"  [3] Oracles: {[o.__class__.__name__ for o in oracles]}", flush=True)

    data = dynamic_graph_to_pyg(g, device=agent.device)

    extra_pairs = _sample_extra_pairs(g, N_EXTRA_PAIRS, seed, (src_node, dst_node))
    all_pairs = [(src_node, dst_node)] + extra_pairs
    print(f"  [4] Query pairs: 1 canonical + {len(extra_pairs)} extra = {len(all_pairs)} total", flush=True)

    library = ExpertLibrary(embed_dim=HIDDEN_DIM, max_size=5000, similarity="cosine")
    n_paths = 0
    n_hops  = 0

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
            n_hops  += len(mc)
            print(f"    {oracle.__class__.__name__}: cost={cost:.1f} len={len(path)} hops={len(mc)}", flush=True)

    print(f"  Library: {len(library)} entries, {n_paths} paths, {n_hops} hops", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    library.save(str(out_path))
    elapsed      = time.perf_counter() - t_cell
    wall_elapsed = time.perf_counter() - wall_t0
    print(f"  Saved: {out_path}  (cell={elapsed:.1f}s  wall={wall_elapsed:.1f}s)", flush=True)
    return out_path


def main() -> None:
    cells = json.loads(SUBSET_JSON.read_text())
    print(f"V3 Long-Train Library Builder — {len(cells)} cells in subset JSON")
    print(f"Agents dir : {AGENTS_DIR}")
    print(f"Output dir : {OUT_DIR}")
    wall_t0 = time.perf_counter()

    saved: list[pathlib.Path] = []
    n_skipped_fail = 0

    for idx, cell in enumerate(cells):
        seed        = cell["seed"]
        scenario_id = cell["scenario_id"]
        reached_4x  = cell.get("warm_strict", False)

        if not reached_4x:
            print(f"\nCell {idx:02d} | seed={seed} scen={scenario_id} — [skipped: 4x failed]")
            n_skipped_fail += 1
            continue

        p = build_cell_library(idx, cell, wall_t0)
        if p is not None:
            saved.append(p)

    total = time.perf_counter() - wall_t0
    print(f"\n{'='*60}")
    print(f"Done.  Wall-clock: {total:.1f}s  ({total/60:.1f} min)")
    print(f"Skipped (4x failed): {n_skipped_fail}")
    print(f"Libraries built/found: {len(saved)}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
