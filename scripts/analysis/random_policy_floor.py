"""Random-policy reach floor on the masked single-query cells of all three scales.

For every cell of the masked sweeps, rebuild the evaluation graph state (a fresh DynamicGraph
from the cell's grid seed, perturbed once per training iteration exactly as train_gnn_dqn does)
and check it against the stored Dijkstra reference. Then run R uniform random rollouts under the
evaluation rules: at each step choose uniformly among active, unvisited neighbours (the masked
action space), stop on reaching the goal, on a dead end or at the 300-step budget.

Reported per scale: the mean per-cell probability that one random rollout reaches the goal, and
the expected number of the 25 cells a single random rollout would reach (the sum of per-cell
probabilities), next to the trained arms' counts from the same artefacts.

Usage: python scripts/analysis/random_policy_floor.py [--rollouts 1000]
Output: runs/thesis_support/item_i_random_policy_floor.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qwarm.env.dynamic_graph import DynamicGraph  # noqa: E402

SCALES = {
    "25x25": dict(artefact="runs/sweep_phase3_masked.json", n_iterations=5,
                  grid=dict(grid_width=25, grid_height=25, extra_edges=2, deactivate_prob=0.15)),
    "50x50": dict(artefact="runs/sweep_50x50_masked_gpu/sweep_v1_50x50_1x.json", n_iterations=10,
                  grid=dict(grid_width=50, grid_height=50, extra_edges=3, deactivate_prob=0.22,
                            node_deactivate_prob=0.05)),
    "100x100": dict(artefact="runs/sweep_100x100_masked.json", n_iterations=10,
                    grid=dict(grid_width=100, grid_height=100, extra_edges=4, deactivate_prob=0.30)),
}
BUDGET = 300


def dijkstra(graph, nodes, src, dst):
    """Same cost as the sweeps: distance + 0.1*time + node_penalty of the entered node."""
    q, best, vis = [(0.0, src)], {src: 0.0}, set()
    while q:
        c, n = heapq.heappop(q)
        if n in vis:
            continue
        vis.add(n)
        if n == dst:
            return c
        for nb, e in graph[n].items():
            if not e["active"] or not nodes[nb]["active"]:
                continue
            nc = c + e["distance"] + 0.1 * e["time"] + nodes[nb]["node_penalty"]
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(q, (nc, nb))
    return float("inf")


def eval_state(cfg, grid_seed, n_iterations):
    g = DynamicGraph(**cfg, seed=grid_seed)
    for it in range(n_iterations):
        g.update_graph(iteration=it + 1)
    return g


def random_reach(g, src, dst, rollouts, rng):
    # adjacency restricted to active edges into active nodes (fixed within the evaluation state)
    adj = {n: [nb for nb, e in g.graph[n].items() if e["active"] and g.nodes[nb]["active"]]
           for n in g.graph}
    hits = 0
    for _ in range(rollouts):
        cur, visited = src, {src}
        for _ in range(BUDGET):
            valid = [nb for nb in adj[cur] if nb not in visited]
            if not valid:
                break
            cur = valid[int(rng.integers(len(valid)))]
            visited.add(cur)
            if cur == dst:
                hits += 1
                break
    return hits / rollouts


def load_cells(path):
    d = json.load(open(ROOT / path))
    if isinstance(d, dict):
        d = next(v for v in d.values() if isinstance(v, list))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=1000)
    args = ap.parse_args()
    out = {"item": "i", "description": __doc__.strip().splitlines()[0], "budget": BUDGET,
           "rollouts_per_cell": args.rollouts, "python": platform.python_version(),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "scales": {}}
    for scale, spec in SCALES.items():
        cells = load_cells(spec["artefact"])
        rows, mismatches = [], 0
        for c in cells:
            grid_seed = int(c.get("grid_seed") or c["scenario_id"].split("_")[0].replace("seed", ""))
            g = eval_state(spec["grid"], grid_seed, spec["n_iterations"])
            dij = dijkstra(g.graph, g.nodes, c["source"], c["destination"])
            stored = c.get("dijkstra_cost")
            match = (stored is None and dij == float("inf")) or (
                stored is not None and dij != float("inf") and abs(dij - stored) < 1e-6)
            mismatches += 0 if match else 1
            rng = np.random.default_rng([grid_seed, int(c["seed"])])
            p = random_reach(g, c["source"], c["destination"], args.rollouts, rng) if dij != float("inf") else 0.0
            rows.append({"seed": c["seed"], "scenario_id": c["scenario_id"], "solvable": dij != float("inf"),
                         "dijkstra_rebuilt": None if dij == float("inf") else dij, "dijkstra_stored": stored,
                         "state_matches_artefact": match, "random_reach_prob": p,
                         "warm_strict": c.get("warm_strict"), "cold_strict": c.get("cold_strict")})
        probs = [r["random_reach_prob"] for r in rows]
        out["scales"][scale] = {
            "artefact": spec["artefact"], "cells": len(rows), "state_mismatches": mismatches,
            "mean_random_reach_prob": float(np.mean(probs)),
            "expected_cells_reached_by_one_random_rollout": float(np.sum(probs)),
            "cells_with_random_prob_ge_0.5": int(sum(p >= 0.5 for p in probs)),
            "warm_cells_reached": int(sum(bool(r["warm_strict"]) for r in rows)),
            "cold_cells_reached": int(sum(bool(r["cold_strict"]) for r in rows)),
            "rows": rows,
        }
        s = out["scales"][scale]
        print(f"{scale}: mismatches={mismatches}  random mean p={s['mean_random_reach_prob']:.3f}  "
              f"expected cells={s['expected_cells_reached_by_one_random_rollout']:.2f}/25  "
              f"warm={s['warm_cells_reached']} cold={s['cold_cells_reached']}", flush=True)
    dst = ROOT / "runs" / "thesis_support" / "item_i_random_policy_floor.json"
    json.dump(out, open(dst, "w"), indent=2)
    print("written", dst)


if __name__ == "__main__":
    main()
