"""STEPS 1c, 2, 3 -- everything that must be settled before any GPU time.

1c. Reward: does normalised step cost + a longer discount horizon make the
    optimal path beat the -5 escape hatch WITHOUT leaking the solution?
2.  Density calibration: pick three chord densities spanning diameter 47/15/5.
3.  Solvability: at low degree, p_deact=0.30 fragments the graph. Find the
    p_deact holding solvable-query fraction fixed across densities, and size
    T_max to the resulting path lengths.
"""
import heapq
import sys
from collections import deque

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


def sample_paths(g, n=25, rs=7):
    rng = np.random.default_rng(rs)
    nodes = list(g.nodes)
    out, tries = [], 0
    while len(out) < n and tries < 800:
        tries += 1
        s, d = rng.choice(nodes, 2, replace=False)
        p = dijkstra(g, str(s), str(d))
        if p:
            out.append(p)
    return out


print("#" * 78)
print("# STEP 1c -- non-leaking reward fix: normalise cost, lengthen horizon")
print("#" * 78)
print(f"\n  {'config':<26}" + "".join(f"{f'g={g}':>14}" for g in
                                      (0.95, 0.99, 0.995)))
for label, W, H, extra in [("25x25 as-used", 25, 25, 2),
                           ("25x25 pure grid", 25, 25, 0),
                           ("50x50 pure grid", 50, 50, 0)]:
    g = DynamicGraph(grid_width=W, grid_height=H, extra_edges=extra,
                     seed=191664964)
    paths = sample_paths(g, 20)
    ms = np.mean([STEP(g, u, v) for p in paths for u, v in zip(p, p[1:])])
    cells = []
    for gam in (0.95, 0.99, 0.995):
        rets = []
        for p in paths:
            R, disc = 0.0, 1.0
            for u, v in zip(p, p[1:]):
                r = -STEP(g, u, v) / ms + (100.0 if v == p[-1] else 0.0)
                R += disc * r
                disc *= gam
            rets.append(R)
        cells.append(f"{np.median(rets):>7.1f} {sum(r > -5 for r in rets):>2d}/20")
    print(f"  {label:<26}" + "".join(f"{c:>14}" for c in cells))
print("\n  (normalised step cost, +100 terminal, no shaping, no leak)")

print("\n" + "#" * 78)
print("# STEP 2 -- density calibration (n_chords = TOTAL long-range edges)")
print("#" * 78)


def diam_and_hops(g, n_src=50, rs=0):
    rng = np.random.default_rng(rs)
    nodes = list(g.nodes)
    srcs = rng.choice(nodes, size=min(n_src, len(nodes)), replace=False)
    allh, ecc = [], 0
    for s in srcs:
        seen = {str(s): 0}
        q = deque([str(s)])
        while q:
            n = q.popleft()
            for m, e in g.graph[n].items():
                if e["active"] and m not in seen:
                    seen[m] = seen[n] + 1
                    q.append(m)
        v = [x for k, x in seen.items() if k != str(s)]
        allh += v
        ecc = max(ecc, max(v) if v else 0)
    a = np.array(allh)
    return ecc, a.mean(), np.percentile(a, 75) - np.percentile(a, 25)


V = 625
print(f"\n  25x25 ({V} nodes), grid edges 1200. Legacy extra_edges=2 "
      f"=> ~1236 chords.\n")
print(f"  {'n_chords':>9} {'edges':>7} {'degree':>7} {'diam':>5} "
      f"{'meanhop':>8} {'hopIQR':>7}")
picks = {}
for nc in (0, 60, 125, 250, 400, 625, 1250):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=0,
                     n_chords=nc, seed=191664964)
    ne = sum(len(v) for v in g.graph.values()) // 2
    d, mh, iqr = diam_and_hops(g)
    print(f"  {nc:>9} {ne:>7} {2 * ne / V:>7.2f} {d:>5} {mh:>8.2f} {iqr:>7.1f}")
    picks[nc] = d

print("\n" + "#" * 78)
print("# STEP 3 -- solvability under perturbation, by density")
print("#" * 78)
print("\n  fraction of 60 random s-t pairs still solvable after 5 "
      "perturbation rounds\n")
print(f"  {'n_chords':>9} {'diam':>5} | " + "".join(
    f"{f'p={p}':>10}" for p in (0.05, 0.10, 0.15, 0.30)))
for nc in (0, 125, 400, 1250):
    row = []
    for pd in (0.05, 0.10, 0.15, 0.30):
        fr = []
        for sd in (1, 2, 3):
            g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=0,
                             n_chords=nc, deactivate_prob=pd, seed=sd)
            for k in range(5):
                g.update_graph(iteration=k + 1)
            rng = np.random.default_rng(11)
            nodes = list(g.nodes)
            ok = tot = 0
            for _ in range(60):
                s, d = rng.choice(nodes, 2, replace=False)
                if not (g.nodes[str(s)]["active"] and g.nodes[str(d)]["active"]):
                    continue
                tot += 1
                ok += dijkstra(g, str(s), str(d)) is not None
            fr.append(ok / max(tot, 1))
        row.append(f"{100 * np.mean(fr):>9.0f}%")
    print(f"  {nc:>9} {picks[nc]:>5} | " + "".join(row))

print("\n" + "#" * 78)
print("# T_max sizing (95th percentile optimal hop count, unperturbed)")
print("#" * 78)
for nc in (0, 125, 400, 1250):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=0,
                     n_chords=nc, seed=191664964)
    h = [len(p) - 1 for p in sample_paths(g, 40)]
    print(f"  n_chords {nc:>5}: median {np.median(h):>5.0f}  p95 "
          f"{np.percentile(h, 95):>5.0f}  max {max(h):>4}"
          f"   -> suggest T_max >= {int(4 * np.percentile(h, 95))}")
