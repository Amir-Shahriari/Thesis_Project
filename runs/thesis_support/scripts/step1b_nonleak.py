"""STEP 1b -- fixes that do NOT leak the optimal solution.

lambda_shape works but is built from Dijkstra distance-to-goal, i.e. the
answer. Chapter 3 disables it for exactly that reason. Three alternatives
that keep the reward solution-agnostic:

  (a) raise the terminal bonus so it is commensurate with path cost
  (b) normalise edge weights so per-step cost is O(1) (bonus unchanged)
  (c) invalid_penalty_mode='scaled', which removes the cheap escape hatch
"""
import heapq
import sys

sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph

STEP = lambda g, u, v: (g.graph[u][v]["distance"]
                        + 0.1 * g.graph[u][v]["time"]
                        + g.nodes[v]["node_penalty"])


def dijkstra(g, src, dst):
    dist, prev, pq, seen = {src: 0.0}, {}, [(0.0, src)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, e in g.graph[u].items():
            if not (e["active"] and g.nodes[v]["active"]):
                continue
            nd = d + STEP(g, u, v)
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None
    path, u = [dst], dst
    while u != src:
        u = prev[u]
        path.append(u)
    return path[::-1]


def ret(g, path, gamma, bonus, scale):
    R, disc = 0.0, 1.0
    for u, v in zip(path, path[1:]):
        r = -STEP(g, u, v) / scale
        if v == path[-1]:
            r += bonus
        R += disc * r
        disc *= gamma
    return R


def run(label, W, H, extra, gamma=0.95, n=20, seed=191664964):
    g = DynamicGraph(grid_width=W, grid_height=H, extra_edges=extra, seed=seed)
    nodes = list(g.nodes)
    rng = np.random.default_rng(7)
    paths, tries = [], 0
    while len(paths) < n and tries < 400:
        tries += 1
        s, d = rng.choice(nodes, 2, replace=False)
        p = dijkstra(g, str(s), str(d))
        if p:
            paths.append(p)
    mean_step = np.mean([STEP(g, u, v) for p in paths
                         for u, v in zip(p, p[1:])])
    print(f"\n=== {label} ===")
    print(f"  mean per-step cost {mean_step:.1f};  "
          f"median hops {np.median([len(p) - 1 for p in paths]):.0f}")

    print("  (a) raise terminal bonus, weights unchanged:")
    for bonus in (100, 250, 500, 1000, 2000):
        r = [ret(g, p, gamma, bonus, 1.0) for p in paths]
        print(f"        bonus {bonus:>5}: median return {np.median(r):>8.1f}"
              f"   beats -5 on {sum(x > -5 for x in r):>2d}/{len(r)}")

    print("  (b) normalise step cost to O(1), bonus stays +100:")
    r = [ret(g, p, gamma, 100.0, mean_step) for p in paths]
    print(f"        divide costs by {mean_step:.1f}: median return "
          f"{np.median(r):>8.1f}   beats -5 on "
          f"{sum(x > -5 for x in r):>2d}/{len(r)}")

    print("  (c) invalid_penalty_mode='scaled' (floor -50): the escape hatch")
    r0 = [ret(g, p, gamma, 100.0, 1.0) for p in paths]
    print(f"        optimal return {np.median(r0):>8.1f} vs escape -50: "
          f"beats it on {sum(x > -50 for x in r0):>2d}/{len(r0)}")


run("25x25 extra_edges=2  (AS USED)", 25, 25, 2)
run("25x25 extra_edges=0  (PURE GRID)", 25, 25, 0)
run("50x50 extra_edges=0  (PURE GRID)", 50, 50, 0)
