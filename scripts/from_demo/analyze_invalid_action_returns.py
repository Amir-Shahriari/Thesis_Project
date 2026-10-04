"""Reward-design diagnosis: discounted return of reaching the goal vs. an
immediate invalid action (-5, terminal) under the ACTUAL environment parameters.

For each grid scale (25x25, 50x50, 100x100) this script:
  1. Samples 5 scenarios with the project's sampler (outer seed 42 -- identical
     scenario set to the headline sweeps at each scale's declared grid config).
  2. Rebuilds the DynamicGraph from scenario.grid_seed and reconstructs the
     Dijkstra-optimal path under the env cost  c(u,v) = dist + 0.1*time +
     node_penalty[v]  on two graph states:
       - "fresh":     iteration-0 graph (all node_penalty = 0, all edges active)
       - "perturbed": after one update_graph() call (node penalties ~U(0,5),
                      edge/node deactivations) -- the state rollouts actually
                      see from iteration 1 onward.
  3. Replays the optimal path through the env reward (terminal step gets +100)
     and computes the discounted return  G = sum_t gamma^t r_t  for
     gamma in {0.95, 0.99, 0.997}.
  4. Prints a table comparing G(optimal) against the invalid-action return (-5).

Read-only analysis: no training code is touched, nothing is written to runs/.

Usage:
    uv run python scripts/analyze_invalid_action_returns.py
"""
from __future__ import annotations

import heapq

import numpy as np

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.eval.scenario_sampler import sample_scenarios

GAMMAS = [0.95, 0.99, 0.997]
INVALID_RETURN = -5.0
GOAL_BONUS = 100.0
OUTER_SEED = 42
N_SCENARIOS = 5

# Declared per-scale grid configs (match the headline sweep scripts).
SCALE_CFGS = {
    "25x25": dict(grid_width=25, grid_height=25, extra_edges=2, deactivate_prob=0.10),
    "50x50": dict(grid_width=50, grid_height=50, extra_edges=3, deactivate_prob=0.22),
    "100x100": dict(grid_width=100, grid_height=100, extra_edges=4, deactivate_prob=0.30),
}


def dijkstra_path(graph: dict, nodes: dict, src: str, dst: str) -> list[str]:
    """Dijkstra under the env cost formula; returns node path ([] if unreachable)."""
    q: list[tuple[float, str]] = [(0.0, src)]
    best: dict[str, float] = {src: 0.0}
    prev: dict[str, str] = {}
    vis: set[str] = set()
    while q:
        c, n = heapq.heappop(q)
        if n in vis:
            continue
        vis.add(n)
        if n == dst:
            path = [dst]
            while path[-1] != src:
                path.append(prev[path[-1]])
            return path[::-1]
        for nb, e in graph[n].items():
            if not e["active"] or not nodes[nb]["active"]:
                continue
            nc = c + e["distance"] + 0.1 * e["time"] + nodes[nb]["node_penalty"]
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                prev[nb] = n
                heapq.heappush(q, (nc, nb))
    return []


def path_returns(graph: dict, nodes: dict, path: list[str]) -> tuple[float, list[float]]:
    """(total env cost, per-step reward sequence incl. +100 on the final step)."""
    rewards: list[float] = []
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = graph[u][v]
        step_cost = e["distance"] + 0.1 * e["time"] + nodes[v]["node_penalty"]
        cost += step_cost
        rewards.append(-step_cost)
    rewards[-1] += GOAL_BONUS
    return cost, rewards


def discounted(rewards: list[float], gamma: float) -> float:
    return float(sum(r * gamma**t for t, r in enumerate(rewards)))


def main() -> None:
    hdr = (f"{'scale':<8} {'state':<9} {'scenario':<22} {'T':>4} {'opt_cost':>9} "
           f"{'undisc':>8}" + "".join(f" {'G(g=' + str(g) + ')':>11}" for g in GAMMAS)
           + f" {'invalid':>8}")
    print(hdr)
    print("-" * len(hdr))

    summary: dict[tuple[str, str], list[list[float]]] = {}

    for scale, cfg in SCALE_CFGS.items():
        rng = np.random.default_rng(OUTER_SEED)
        scenarios = sample_scenarios(cfg, n_scenarios=N_SCENARIOS, rng=rng)
        for sc in scenarios:
            for state_name in ("fresh", "perturbed"):
                g = DynamicGraph(**cfg, seed=sc.grid_seed)
                if state_name == "perturbed":
                    g.update_graph(iteration=1)
                path = dijkstra_path(g.graph, g.nodes, sc.source_node, sc.destination_node)
                if not path:
                    print(f"{scale:<8} {state_name:<9} {sc.scenario_id:<22} "
                          f"{'--':>4} {'UNREACHABLE':>9}")
                    continue
                cost, rewards = path_returns(g.graph, g.nodes, path)
                T = len(rewards)
                gs = [discounted(rewards, gamma) for gamma in GAMMAS]
                summary.setdefault((scale, state_name), []).append([T, cost] + gs)
                row = (f"{scale:<8} {state_name:<9} {sc.scenario_id:<22} {T:>4} "
                       f"{cost:>9.1f} {GOAL_BONUS - cost:>8.1f}"
                       + "".join(f" {v:>11.2f}" for v in gs)
                       + f" {INVALID_RETURN:>8.1f}")
                print(row)

    print()
    print("MEANS over reachable scenarios "
          "(G > -5 means reaching the goal beats an immediate invalid action):")
    hdr2 = (f"{'scale':<8} {'state':<9} {'n':>3} {'T':>6} {'opt_cost':>9}"
            + "".join(f" {'G(g=' + str(g) + ')':>11}" for g in GAMMAS)
            + "  beats -5?")
    print(hdr2)
    print("-" * len(hdr2))
    for (scale, state_name), rows in summary.items():
        arr = np.array(rows)
        m = arr.mean(axis=0)
        beats = " ".join(
            f"g={g}:{'YES' if m[2 + i] > INVALID_RETURN else 'NO'}"
            for i, g in enumerate(GAMMAS)
        )
        print(f"{scale:<8} {state_name:<9} {len(rows):>3} {m[0]:>6.1f} {m[1]:>9.1f}"
              + "".join(f" {m[2 + i]:>11.2f}" for i in range(len(GAMMAS)))
              + f"  {beats}")


if __name__ == "__main__":
    main()
