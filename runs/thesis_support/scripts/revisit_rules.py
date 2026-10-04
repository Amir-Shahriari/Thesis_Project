"""Three revisit rules x exploration temperature.

The 98.2% figure was measured at eps=1 (uniform random). Training uses
eps-greedy decaying 0.40 -> 0.05, so the honest question is how the death
rate behaves as the policy becomes greedy. A greedy proxy is unavailable
without a checkpoint, so we bracket it: uniform, and eps-greedy where the
greedy action follows the Dijkstra-optimal next hop (an ORACLE upper bound
on how good the learned policy could be).

Rules:
  reactive  -- visited nodes stay in the candidate set; picking one = -5, end
  mask      -- visited nodes removed; empty set = dead end, episode ends
  fallback  -- visited nodes removed; if that empties the set, allow a
               visited neighbour and continue without penalty
"""
import heapq
import sys

sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph

STEP = lambda g, u, v: (g.graph[u][v]["distance"]
                        + 0.1 * g.graph[u][v]["time"]
                        + g.nodes[v]["node_penalty"])


def cost_to(g, dst):
    dist, pq, seen = {dst: 0.0}, [(0.0, dst)], set()
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


def rollout(g, src, dst, rule, eps, d2g, rng, max_steps=400):
    cur, visited, steps = src, {src}, 0
    while steps < max_steps:
        act = [n for n, e in g.graph[cur].items()
               if e["active"] and g.nodes[n]["active"]]
        if not act:
            return "dead-end"
        pool = act
        if rule in ("mask", "fallback"):
            unv = [a for a in act if a not in visited]
            if unv:
                pool = unv
            elif rule == "mask":
                return "dead-end"
            else:
                pool = act                      # fallback: allow revisit
        if rng.random() < eps:
            a = pool[rng.integers(len(pool))]
        else:
            a = min(pool, key=lambda n: d2g.get(n, float("inf")))
        revisit = a in visited
        if revisit and rule == "reactive":
            return "revisit"
        steps += 1
        cur = a
        visited.add(a)
        if cur == dst:
            return "goal"
    return "timeout"


for nc, lab in ((0, "n_chords=0     degree 3.8, diam 47"),
                (1250, "n_chords=1250  degree 7.8, diam 5")):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=0,
                     n_chords=nc, seed=191664964)
    nodes = list(g.nodes)
    rng = np.random.default_rng(3)
    pairs, d2gs = [], {}
    while len(pairs) < 15:
        s, d = rng.choice(nodes, 2, replace=False)
        d2g = d2gs.setdefault(str(d), cost_to(g, str(d)))
        if str(s) in d2g:
            pairs.append((str(s), str(d)))
    print(f"\n{'=' * 72}\n{lab}\n{'=' * 72}")
    print(f"  {'rule':<10}" + "".join(f"{f'eps={e}':>13}" for e in
                                      (1.0, 0.4, 0.05)))
    for rule in ("reactive", "mask", "fallback"):
        cells = []
        for eps in (1.0, 0.4, 0.05):
            out = {}
            for s, d in pairs:
                for _ in range(80):
                    o = rollout(g, s, d, rule, eps, d2gs[d], rng)
                    out[o] = out.get(o, 0) + 1
            n = sum(out.values())
            cells.append(f"{100 * out.get('goal', 0) / n:>11.1f}%")
        print(f"  {rule:<10}" + "".join(f"{c:>13}" for c in cells))
    print("  (eps=0.05 greedy action = Dijkstra next hop, an ORACLE upper "
          "bound\n   on the learned policy, not a measurement of it)")
