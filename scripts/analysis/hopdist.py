"""Is the locality probe variance-starved?

The reviewer's claim: extra_edges is per-node and uniformly sampled, so the
graph is a rewired small-world, hop distances concentrate, and Spearman rho
against hop distance is near zero for ANY embedding. Measure the hop
distribution directly, with a pure-grid control.
"""
import sys
from collections import deque, Counter

sys.path.insert(0, r"C:\Users\amirh\Desktop\qwarm-gnn-rl\src")
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph


def bfs_all(g, source):
    seen = {source: 0}
    q = deque([source])
    while q:
        n = q.popleft()
        for m, e in g.graph[n].items():
            if e["active"] and m not in seen:
                seen[m] = seen[n] + 1
                q.append(m)
    return seen


def report(label, W, H, extra, seed=191664964, n_src=60):
    g = DynamicGraph(grid_width=W, grid_height=H, extra_edges=extra, seed=seed)
    n_edges = sum(len(v) for v in g.graph.values()) // 2
    grid_edges = 2 * W * H - W - H
    rng = np.random.default_rng(0)
    nodes = list(g.nodes.keys())
    srcs = rng.choice(nodes, size=min(n_src, len(nodes)), replace=False)
    d = []
    ecc = []
    for s in srcs:
        hops = bfs_all(g, str(s))
        vals = [v for k, v in hops.items() if k != s]
        d.extend(vals)
        if vals:
            ecc.append(max(vals))
    d = np.array(d)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    print(f"\n=== {label} ===")
    print(f"  nodes {g.num_nodes}, edges {n_edges} "
          f"(grid-only would be {grid_edges}); "
          f"extra chords ~= {n_edges - grid_edges}")
    print(f"  mean degree {2 * n_edges / g.num_nodes:.1f}")
    print(f"  hop distance over {len(d):,} pairs: "
          f"mean {d.mean():.2f}  median {med:.0f}  "
          f"IQR [{q1:.0f}, {q3:.0f}] = {q3 - q1:.0f} hops  max {d.max()}")
    print(f"  eccentricity (max over sampled sources): {max(ecc)}")
    frac = Counter(d.tolist())
    top = sorted(frac.items())
    print("  distribution: " + "  ".join(
        f"{k}:{100 * v / len(d):.0f}%" for k, v in top if 100 * v / len(d) >= 1))
    return g, d


for label, W, H, extra in [
    ("25x25, extra_edges=2  (AS USED, smoke)", 25, 25, 2),
    ("25x25, extra_edges=0  (pure grid control)", 25, 25, 0),
    ("50x50, extra_edges=3  (AS USED: run_sweep_50x50.py, run_fleet_mode.py)",
     50, 50, 3),
    ("50x50, extra_edges=4  (NOT USED by any experiment -- kept for reference)",
     50, 50, 4),
    ("50x50, extra_edges=0  (pure grid control)", 50, 50, 0),
    ("100x100, extra_edges=4 (AS USED, headline)", 100, 100, 4),
    ("100x100, extra_edges=0 (pure grid control)", 100, 100, 0),
]:
    report(label, W, H, extra)
