"""STEP 1 -- return arithmetic.

Does the Dijkstra-optimal path actually earn more than the -5 invalid-action
escape hatch? On the as-used small-world graphs paths are ~4 hops, so the
+100 terminal bonus survives discounting. On a pure grid they are ~17-25
hops and gamma^T crushes it while step costs accumulate linearly.

If optimal return < -5, an optimal agent should terminate immediately and
NO amount of training or warm-starting can fix it. That decides whether the
high-diameter sweep is worth starting at all.
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
    """Min composite-cost path, the same cost the reward negates."""
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    seen = set()
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
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, None
    path, u = [dst], dst
    while u != src:
        u = prev[u]
        path.append(u)
    return path[::-1], dist[dst]


def path_return(g, path, gamma, lam, d_to_goal):
    """Discounted return the env would actually pay for this path."""
    R, disc = 0.0, 1.0
    for u, v in zip(path, path[1:]):
        r = -STEP(g, u, v)
        if lam > 0.0:
            r += lam * (d_to_goal[u] - d_to_goal[v])
        if v == path[-1]:
            r += 100.0
        R += disc * r
        disc *= gamma
    return R


def cost_to_goal(g, dst):
    dist = {dst: 0.0}
    pq = [(0.0, dst)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, e in g.graph[u].items():
            if not (e["active"] and g.nodes[v]["active"]):
                continue
            nd = d + STEP(g, v, u)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def run(label, W, H, extra, gammas=(0.95, 0.99), n_pairs=20, seed=191664964):
    g = DynamicGraph(grid_width=W, grid_height=H, extra_edges=extra, seed=seed)
    nodes = list(g.nodes)
    rng = np.random.default_rng(7)
    pairs, tries = [], 0
    while len(pairs) < n_pairs and tries < 400:
        tries += 1
        s, d = rng.choice(nodes, 2, replace=False)
        p, c = dijkstra(g, str(s), str(d))
        if p:
            pairs.append((str(s), str(d), p, c))
    print(f"\n{'=' * 74}\n{label}   ({len(pairs)} solvable pairs)\n{'=' * 74}")
    hops = [len(p) - 1 for _, _, p, _ in pairs]
    costs = [c for _, _, _, c in pairs]
    print(f"  optimal path: median {np.median(hops):.0f} hops "
          f"(max {max(hops)}), median composite cost {np.median(costs):.0f}")
    print(f"\n  {'lambda_shape':>12} | " + " | ".join(
        f"gamma={gm}  ret / beats -5" for gm in gammas))
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        cells = []
        for gm in gammas:
            rets = []
            for s, d, p, _ in pairs:
                d2g = cost_to_goal(g, d) if lam > 0 else {}
                rets.append(path_return(g, p, gm, lam, d2g))
            med = float(np.median(rets))
            win = sum(r > -5 for r in rets)
            cells.append(f"{med:>10.1f} / {win:>2d}/{len(rets)}")
        print(f"  {lam:>12.1f} | " + " | ".join(cells))


run("25x25, extra_edges=2   (AS USED -- small world, diameter 5)", 25, 25, 2)
run("25x25, extra_edges=0   (PURE GRID -- diameter 47)", 25, 25, 0)
run("50x50, extra_edges=0   (PURE GRID -- diameter 96)", 50, 50, 0)
