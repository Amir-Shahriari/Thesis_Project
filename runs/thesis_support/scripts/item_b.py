"""Item (b): inflation of the Dijkstra denominator in the within-trajectory study.

run_v3_realtime_eval.py (Desktop/Demo/scripts) divides the agent's env cost
(distance + 0.1*time + node_penalty per step) by ClassicalDijkstra's
find_optimized_route cost, which adds a path-length (0.05*len(path)) and a
node-degree (0.1*degree) penalty per step. This script recomputes, for the six
cells of the study, on the same fresh graph the study used:
  D_ref   = ClassicalDijkstra cost (the denominator actually used)
  D_env   = minimum env-cost path (pure composite-cost Dijkstra)
  C_refp  = env cost of the ClassicalDijkstra path
and reports D_ref / D_env - 1 (and D_ref / C_refp - 1).
No checkpoint is loaded.
"""
import heapq
import json
import pathlib
import sys

sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.oracles.classical_dijkstra import ClassicalDijkstra

from prov import REPO, provenance, write, sha256, git

DEMO = pathlib.Path(r"C:\Users\amirh\Desktop\Demo")
CELLS = DEMO / "runs" / "v1_realtime_test_grid_v2.json"
PER_RUN = DEMO / "runs" / "v3_realtime_25x25.json"
STUDY_SCRIPT = DEMO / "scripts" / "run_v3_realtime_eval.py"

STEP = lambda g, u, v: (g.graph[u][v]["distance"] + 0.1 * g.graph[u][v]["time"]
                        + g.nodes[v]["node_penalty"])


def env_dijkstra(g, src, dst):
    dist, pq, seen = {src: 0.0}, [(0.0, src)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            return d
        if not g.nodes[u]["active"]:
            continue
        for v, e in g.graph[u].items():
            if not (e["active"] and g.nodes[v]["active"]):
                continue
            nd = d + STEP(g, u, v)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float("inf")


cells = json.loads(CELLS.read_text())
stored = {}
for r in json.loads(PER_RUN.read_text()):
    stored[r["cell_idx"]] = r["dijkstra_cost"]

rows = []
for c in cells:
    cfg = c["grid"]
    g = DynamicGraph(grid_width=cfg["grid_width"], grid_height=cfg["grid_height"],
                     extra_edges=cfg.get("extra_edges", 2),
                     deactivate_prob=cfg.get("deactivate_prob", 0.10),
                     seed=c["grid_seed"])
    d_ref, path, _ = ClassicalDijkstra(g.nodes, g.graph).find_optimized_route(
        c["source"], c["destination"])
    d_env = env_dijkstra(g, c["source"], c["destination"])
    c_refp = sum(STEP(g, u, v) for u, v in zip(path, path[1:]))
    rows.append({
        "cell_idx": c["cell_idx"], "outer_seed": c["outer_seed"],
        "grid_seed": c["grid_seed"], "scenario_id": c["scenario_id"],
        "source": c["source"], "destination": c["destination"],
        "dijkstra_ref_cost_recomputed": round(d_ref, 4),
        "dijkstra_ref_cost_stored_in_v3_realtime_25x25": stored.get(c["cell_idx"]),
        "ref_path_hops": len(path) - 1,
        "env_optimal_cost": round(d_env, 4),
        "env_cost_of_ref_path": round(c_refp, 4),
        "inflation_vs_env_optimum_pct": round(100 * (d_ref / d_env - 1), 2),
        "inflation_vs_env_cost_of_same_path_pct": round(100 * (d_ref / c_refp - 1), 2),
    })

inf1 = np.array([r["inflation_vs_env_optimum_pct"] for r in rows])
inf2 = np.array([r["inflation_vs_env_cost_of_same_path_pct"] for r in rows])
summary = {
    "vs_env_optimum": {"min_pct": float(inf1.min()), "max_pct": float(inf1.max()),
                       "mean_pct": round(float(inf1.mean()), 2)},
    "vs_env_cost_of_same_path": {"min_pct": float(inf2.min()), "max_pct": float(inf2.max()),
                                 "mean_pct": round(float(inf2.mean()), 2)},
    "stored_denominators_match": all(
        abs(r["dijkstra_ref_cost_recomputed"] - r["dijkstra_ref_cost_stored_in_v3_realtime_25x25"]) < 1e-3
        for r in rows),
}
print(json.dumps(rows, indent=1)); print(summary)

lo, hi, mean = summary["vs_env_optimum"]["min_pct"], summary["vs_env_optimum"]["max_pct"], summary["vs_env_optimum"]["mean_pct"]
reproduced = round(lo) == 10 and round(hi) == 26 and round(mean) == 15
obj = {
    "item": "b",
    "description": ("Within-trajectory perturbation study (V3 realtime, six "
                    "25x25 cells): inflation of the Dijkstra reference cost by "
                    "the path-length and node-degree penalties of "
                    "ClassicalDijkstra/_cost.calculate_cost, which the agent's "
                    "env cost does not include."),
    "thesis_locations": ["appendix-a.tex:345 (sec:app-transfer-realtime)"],
    "audit_findings": ["N8-01"],
    "provenance": provenance(
        [REPO / "src" / "qwarm" / "oracles" / "classical_dijkstra.py",
         REPO / "src" / "qwarm" / "oracles" / "_cost.py",
         REPO / "src" / "qwarm" / "env" / "dynamic_graph.py"],
        [f"{sys.executable} {pathlib.Path(__file__)}"],
        extra={"this_script": {"path": str(pathlib.Path(__file__)), "sha256": sha256(__file__)},
               "study_script_read_only": {"path": str(STUDY_SCRIPT), "sha256": sha256(STUDY_SCRIPT),
                                          "demo_git_revision": git(DEMO, "rev-parse", "HEAD")},
               "inputs_read_only": [{"path": str(CELLS), "sha256": sha256(CELLS)},
                                    {"path": str(PER_RUN), "sha256": sha256(PER_RUN)}],
               "note": ("No existing script reports this inflation, so a new "
                        "deterministic analysis was written (this file). It "
                        "rebuilds each cell's fresh graph exactly as "
                        "run_v3_realtime_eval._build_fresh_graph does and "
                        "computes the denominator with the same "
                        "ClassicalDijkstra. The Demo and Thesis_Project copies "
                        "of classical_dijkstra.py are identical, and the legacy "
                        "extra_edges path of dynamic_graph.py is unchanged.")}),
    "parameters": {"cells": "Demo/runs/v1_realtime_test_grid_v2.json (6 cells)",
                   "grid": "25x25, extra_edges=2, deactivate_prob=0.10, fresh (iteration-0) graph",
                   "grid_seeds": sorted({c['grid_seed'] for c in cells}),
                   "env_step_cost": "distance + 0.1*time + node_penalty[v]",
                   "reference_cost": "env step cost + 0.05*len(path_so_far) + 0.1*degree(v)"},
    "results": {"per_cell": rows, "summary": summary},
    "thesis_values": {"inflation_range_pct": [10, 26], "mean_pct_approx": 15},
    "status": "REPRODUCED" if reproduced else "DIFFERENT",
}
write("item_b_realtime_dijkstra_inflation.json", obj)
print(obj["status"])
