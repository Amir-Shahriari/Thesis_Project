"""Item (e): exact all-source graph diameters.

Exact diameter = max over ALL sources of BFS eccentricity on the active
(iteration-0) graph, computed with scipy.sparse.csgraph.shortest_path
(unweighted). Cross-checked against a pure-Python BFS (same as
step23_calibrate.diam_and_hops, but over every source) at 25x25. Also reports
the sampled 50-source value of step23_calibrate.diam_and_hops /
hopdist.report for comparison.
"""
import pathlib
import sys
import time
from collections import deque

sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from qwarm.env.dynamic_graph import DynamicGraph

from prov import REPO, provenance, write, sha256

THESIS_SEEDS = [191664964, 2029167941, 1830621171, 1173222464, 518677876]


def to_csr(g):
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    r, c = [], []
    for u, nb in g.graph.items():
        for v, e in nb.items():
            if e["active"]:
                r.append(idx[u]); c.append(idx[v])
    n = len(nodes)
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), nodes


def exact(g, chunk=1000):
    A, nodes = to_csr(g)
    sym = (A != A.T).nnz == 0
    ncomp, _ = connected_components(A, directed=False)
    n = A.shape[0]
    ecc = np.zeros(n, dtype=int)
    hist = np.zeros(1, dtype=np.int64)
    for s in range(0, n, chunk):
        D = shortest_path(A, method="D", directed=False, unweighted=True,
                          indices=np.arange(s, min(n, s + chunk)))
        if not np.isfinite(D).all():
            return {"connected": False, "n_components": int(ncomp)}
        D = D.astype(int)
        ecc[s:s + D.shape[0]] = D.max(axis=1)
        b = np.bincount(D.ravel())
        if len(b) > len(hist):
            hist = np.pad(hist, (0, len(b) - len(hist)))
        hist[:len(b)] += b
    hist[0] -= n                                    # drop self-pairs
    # ordered pairs (u != v)
    vals = np.arange(len(hist))
    cdf = np.cumsum(hist) / hist.sum()
    pct = lambda q: int(np.searchsorted(cdf, q / 100.0) )
    return {"connected": True, "n_components": int(ncomp), "symmetric_adjacency": bool(sym),
            "nodes": n, "edges": int(A.nnz // 2), "mean_degree": round(A.nnz / n, 4),
            "diameter_exact": int(ecc.max()), "radius_exact": int(ecc.min()),
            "n_nodes_attaining_diameter_ecc": int((ecc == ecc.max()).sum()),
            "mean_hops_all_pairs": round(float((vals * hist).sum() / hist.sum()), 4),
            "hop_q1_all_pairs": pct(25), "hop_median_all_pairs": pct(50),
            "hop_q3_all_pairs": pct(75)}


def sampled_ecc(g, n_src=50, rs=0):
    """step23_calibrate.diam_and_hops eccentricity (50 sampled sources)."""
    rng = np.random.default_rng(rs)
    nodes = list(g.nodes)
    srcs = rng.choice(nodes, size=min(n_src, len(nodes)), replace=False)
    ecc = 0
    for s in srcs:
        seen = {str(s): 0}; q = deque([str(s)])
        while q:
            n = q.popleft()
            for m, e in g.graph[n].items():
                if e["active"] and m not in seen:
                    seen[m] = seen[n] + 1; q.append(m)
        ecc = max(ecc, max(v for k, v in seen.items() if k != str(s)))
    return ecc


def python_bfs_diam(g):
    best = 0
    for s in g.nodes:
        seen = {s: 0}; q = deque([s])
        while q:
            n = q.popleft()
            for m, e in g.graph[n].items():
                if e["active"] and m not in seen:
                    seen[m] = seen[n] + 1; q.append(m)
        best = max(best, max(seen.values()))
    return best


t0 = time.time()
results = {"as_used_thesis_seeds": [], "pure_grids": [], "density_axis_25x25": []}

for W, e in ((25, 2), (50, 3), (100, 4)):
    for sd in THESIS_SEEDS:
        g = DynamicGraph(grid_width=W, grid_height=W, extra_edges=e, seed=sd)
        r = exact(g)
        r.update({"grid": f"{W}x{W}", "extra_edges": e, "seed": sd})
        if sd == 191664964:
            r["diameter_sampled_50_sources"] = sampled_ecc(g)
        results["as_used_thesis_seeds"].append(r)
        print(r, round(time.time() - t0, 1), flush=True)

for W in (25, 50, 100):
    g = DynamicGraph(grid_width=W, grid_height=W, extra_edges=0, seed=191664964)
    r = exact(g)
    r.update({"grid": f"{W}x{W}", "extra_edges": 0, "n_chords": 0, "seed": 191664964,
              "analytic_2(W-1)": 2 * (W - 1),
              "diameter_sampled_50_sources_rng0": sampled_ecc(g)})
    results["pure_grids"].append(r)
    print(r, flush=True)

for nc in (0, 60, 125, 250, 400, 625, 1250):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=0, n_chords=nc, seed=191664964)
    r = exact(g)
    r.update({"grid": "25x25", "n_chords": nc, "seed": 191664964,
              "diameter_sampled_50_sources (tab:density method)": sampled_ecc(g),
              "diameter_python_bfs_all_sources_crosscheck": python_bfs_diam(g)})
    results["density_axis_25x25"].append(r)
    print(r, flush=True)

au = {(r["grid"], r["seed"]): r["diameter_exact"] for r in results["as_used_thesis_seeds"]}
summary = {
    "as_used_exact_diameter_by_scale": {
        s: sorted({au[(s, sd)] for sd in THESIS_SEEDS}) for s in ("25x25", "50x50", "100x100")},
    "pure_grid_exact": {r["grid"]: r["diameter_exact"] for r in results["pure_grids"]},
    "density_exact": {r["n_chords"]: r["diameter_exact"] for r in results["density_axis_25x25"]},
    "density_sampled": {r["n_chords"]: r["diameter_sampled_50_sources (tab:density method)"]
                        for r in results["density_axis_25x25"]},
}
print(summary)
thesis = {"as_used_diameter_every_scale": 5,
          "pure_grid_diameters": {"25x25": 47, "50x50": 96, "100x100": 194},
          "tab_density_diameter": {0: 47, 60: 15, 125: 13, 250: 10, 400: 8, 625: 7, 1250: 5},
          "ch5_density_points": {0: 47, 60: 15, 1250: 5}}
obj = {
    "item": "e",
    "description": "Exact all-source graph diameters (as-used scales at the thesis grid seeds, pure grids, tab:density points).",
    "thesis_locations": ["ch1-introduction.tex:13", "ch3-methodology.tex:36", "ch3-methodology.tex:47-58 (tab:density)",
                         "ch4-experiments.tex:19", "ch4-experiments.tex:126", "ch5-conclusion.tex:50",
                         "figures-generated.tex:634"],
    "audit_findings": ["N9-02"],
    "provenance": provenance([REPO / "src" / "qwarm" / "env" / "dynamic_graph.py",
                              REPO / "scripts" / "analysis" / "step23_calibrate.py",
                              REPO / "scripts" / "analysis" / "hopdist.py"],
                             [f"{sys.executable} {pathlib.Path(__file__)}"],
                             extra={"this_script": {"path": str(pathlib.Path(__file__)), "sha256": sha256(__file__)},
                                    "note": ("No existing script computes exact diameters (step23_calibrate.py and "
                                             "hopdist.py take the max eccentricity over 50/60 sampled BFS sources, "
                                             "a lower bound), so this new deterministic analysis builds the same "
                                             "graphs with Thesis_Project/src DynamicGraph and runs BFS from every node.")}),
    "parameters": {"thesis_grid_seeds": THESIS_SEEDS,
                   "as_used_configs": {"25x25": "extra_edges=2", "50x50": "extra_edges=3", "100x100": "extra_edges=4"},
                   "graph_state": "iteration-0 (construction-time) graph, all nodes/edges active",
                   "density_axis": "25x25, extra_edges=0, n_chords in {0,60,125,250,400,625,1250}, seed 191664964",
                   "method": "scipy.sparse.csgraph.shortest_path(method='D', unweighted=True, directed=False), all sources",
                   "hop_quantiles": "over all ordered pairs u != v"},
    "results": results,
    "summary": summary,
    "thesis_values": thesis,
    "runtime_s": round(time.time() - t0, 1),
}
write("item_e_exact_diameters.json", obj)
