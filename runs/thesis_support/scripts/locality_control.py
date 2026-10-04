"""Positive control for the locality probe.

If the probe is sound, node COORDINATES -- which trivially encode grid
position -- must correlate strongly with hop distance. On a real grid they
should reach rho ~ 0.9. If they do not on the as-used graph, the probe is
measuring the generator, not the representation.
"""
import sys
from collections import deque

sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import numpy as np
from scipy.stats import spearmanr
from qwarm.env.dynamic_graph import DynamicGraph


def bfs(g, source):
    seen = {source: 0}
    q = deque([source])
    while q:
        n = q.popleft()
        for m, e in g.graph[n].items():
            if e["active"] and m not in seen:
                seen[m] = seen[n] + 1
                q.append(m)
    return seen


def run(label, W, H, extra, seed=191664964, n_src=80, per_src=60):
    g = DynamicGraph(grid_width=W, grid_height=H, extra_edges=extra, seed=seed)
    nodes = list(g.nodes.keys())
    # node coordinates, exactly the positional part of the 4-d input feature
    xy = np.array([g.nodes[n]["coords"] for n in nodes], dtype=float)
    rng = np.random.default_rng(0)
    srcs = rng.choice(len(nodes), size=min(n_src, len(nodes)), replace=False)
    hops, euc = [], []
    for si in srcs:
        h = bfs(g, nodes[si])
        cand = [n for n in h if n != nodes[si]]
        if not cand:
            continue
        pick = rng.choice(cand, size=min(per_src, len(cand)), replace=False)
        for p in pick:
            pi = nodes.index(str(p))
            hops.append(h[str(p)])
            euc.append(np.linalg.norm(xy[si] - xy[pi]))
    hops, euc = np.array(hops), np.array(euc)
    rho, p = spearmanr(euc, hops)
    uniq = len(np.unique(hops))
    print(f"  {label:44s} rho={rho:+.3f}  p={p:.1e}  "
          f"[{uniq} distinct hop values, sd={hops.std():.2f}]")
    return rho


print("Spearman rho between EUCLIDEAN COORDINATE distance and HOP distance")
print("(a sound probe must score high here -- coordinates ARE the grid)\n")
for W, H, e in [(25, 25, 2), (50, 50, 4), (100, 100, 4)]:
    run(f"{W}x{H}, extra_edges={e}  AS USED", W, H, e)
print()
for W, H in [(25, 25), (50, 50), (100, 100)]:
    run(f"{W}x{H}, extra_edges=0  pure grid control", W, H, 0)
