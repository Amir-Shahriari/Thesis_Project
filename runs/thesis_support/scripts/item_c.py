"""Item (c): locality positive control and hop-distance IQR."""
import re
import sys

from prov import REPO, SCRATCH, provenance, run_copy, write

A = REPO / "scripts" / "analysis"
cmd_l, out_l = run_copy("locality_control.py")
cmd_h, out_h = run_copy("hopdist.py")

loc = []
for m in re.finditer(r"(\d+)x(\d+), extra_edges=(\d+)\s+(AS USED|pure grid control)\s+"
                     r"rho=([+-][\d.]+)\s+p=(\S+)\s+\[(\d+) distinct hop values, sd=([\d.]+)\]", out_l):
    loc.append({"grid": f"{m[1]}x{m[2]}", "extra_edges": int(m[3]),
                "label": m[4], "rho": float(m[5]), "p": float(m[6]),
                "distinct_hop_values": int(m[7]), "hop_sd": float(m[8])})

hop = []
for blk in re.split(r"\n=== ", out_h)[1:]:
    head = blk.split(" ===")[0]
    g = re.search(r"(\d+)x(\d+), extra_edges=(\d+)", head)
    nodes, edges = map(int, re.search(r"nodes (\d+), edges (\d+)", blk).groups())
    md = float(re.search(r"mean degree ([\d.]+)", blk)[1])
    hd = re.search(r"over ([\d,]+) pairs: mean ([\d.]+)\s+median (\d+)\s+IQR \[(\d+), (\d+)\] = (\d+) hops\s+max (\d+)", blk)
    ecc = int(re.search(r"eccentricity \(max over sampled sources\): (\d+)", blk)[1])
    dist = {int(k): int(v) for k, v in re.findall(r"(\d+):(\d+)%", blk.split("distribution:")[1])}
    hop.append({"label": head, "grid": f"{g[1]}x{g[2]}", "extra_edges": int(g[3]),
                "nodes": nodes, "edges": edges, "mean_degree": md,
                "n_pairs": int(hd[1].replace(",", "")), "mean_hops": float(hd[2]),
                "median_hops": int(hd[3]), "q1": int(hd[4]), "q3": int(hd[5]),
                "iqr_hops": int(hd[6]), "max_hops_sampled": int(hd[7]),
                "sampled_eccentricity_max": ecc,
                "hop_distribution_pct_ge1": dist})

# Supplementary: locality control at the 50x50 density actually used (extra_edges=3);
# locality_control.py's "AS USED" 50x50 row uses extra_edges=4.
sys.argv = ["locality_control_supp"]
import importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("lc", SCRATCH / "locality_control.py")
lc = importlib.util.module_from_spec(spec)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    spec.loader.exec_module(lc)          # re-runs the script body (same output)
    supp_rho = lc.run("50x50, extra_edges=3  (as used by run_sweep_50x50.py)", 50, 50, 3)
supp_line = buf.getvalue().strip().splitlines()[-1]

as_used = {r["grid"]: r["rho"] for r in loc if r["label"] == "AS USED"}
pure = {r["grid"]: r["rho"] for r in loc if r["label"] == "pure grid control"}
iqr_used = {h["grid"]: h["iqr_hops"] for h in hop
            if (h["grid"], h["extra_edges"]) in {("25x25", 2), ("50x50", 3), ("100x100", 4)}}
pct4_100 = next(h for h in hop if h["grid"] == "100x100" and h["extra_edges"] == 4)["hop_distribution_pct_ge1"][4]

thesis = {"rho_generated": [0.091, 0.034, -0.012], "rho_pure_grid": [0.978, 0.980, 0.978],
          "hop_iqr_as_used": [1, 1, 0], "pct_pairs_4_hops_100x100": 66,
          "mean_degree": [7.8, 9.9, 12.0]}
regen = {"rho_generated": [as_used["25x25"], as_used["50x50"], as_used["100x100"]],
         "rho_pure_grid": [pure["25x25"], pure["50x50"], pure["100x100"]],
         "hop_iqr_as_used": [iqr_used["25x25"], iqr_used["50x50"], iqr_used["100x100"]],
         "pct_pairs_4_hops_100x100": pct4_100,
         "mean_degree": [next(h["mean_degree"] for h in hop if (h["grid"], h["extra_edges"]) == k)
                         for k in (("25x25", 2), ("50x50", 3), ("100x100", 4))]}
ok = all(abs(a - b) < 1e-9 for a, b in zip(regen["rho_generated"] + regen["rho_pure_grid"],
                                            thesis["rho_generated"] + thesis["rho_pure_grid"])) \
    and regen["hop_iqr_as_used"] == thesis["hop_iqr_as_used"] \
    and regen["pct_pairs_4_hops_100x100"] == 66 and regen["mean_degree"] == thesis["mean_degree"]

obj = {
    "item": "c",
    "description": ("Locality positive control: Spearman rho between Euclidean "
                    "coordinate distance and hop distance on the generated graphs "
                    "vs pure grids; hop-distance IQR at the three scales."),
    "thesis_locations": ["ch1-introduction.tex:117", "ch4-experiments.tex:191",
                         "ch5-conclusion.tex:19", "appendix-a.tex:320 (tab:geometry)",
                         "ch3-methodology.tex:36"],
    "audit_findings": ["N8-03"],
    "provenance": provenance([A / "locality_control.py", A / "hopdist.py"],
                             [cmd_l, cmd_h],
                             [SCRATCH / "locality_control.py", SCRATCH / "hopdist.py"],
                             {"supplementary_command": "item_c.py imports the scratch copy of "
                              "locality_control.py and calls run('...', 50, 50, 3)"}),
    "parameters": {
        "graph_seed": 191664964,
        "locality_control.py": {"configs_as_used": [[25, 25, 2], [50, 50, 4], [100, 100, 4]],
                                "configs_pure_grid": [[25, 25, 0], [50, 50, 0], [100, 100, 0]],
                                "rng": "np.random.default_rng(0)", "n_src": 80, "per_src": 60},
        "hopdist.py": {"rng": "np.random.default_rng(0)", "n_src": 60,
                       "configs": [[25, 25, 2], [25, 25, 0], [50, 50, 3], [50, 50, 4],
                                   [50, 50, 0], [100, 100, 4], [100, 100, 0]]},
    },
    "results": {"locality_control": loc, "hopdist": hop,
                "supplementary_locality_50x50_extra_edges_3": {"rho": float(supp_rho),
                                                               "line": supp_line},
                "regenerated_thesis_quantities": regen},
    "thesis_values": thesis,
    "status": "REPRODUCED" if ok else "DIFFERENT",
    "notes": ("All values reproduce exactly. Caveat: locality_control.py's 50x50 "
              "'AS USED' row is built with extra_edges=4, but hopdist.py labels "
              "50x50 extra_edges=4 'NOT USED by any experiment' (the 50x50 "
              "sweeps use extra_edges=3). The thesis's 0.034 is therefore the "
              "extra_edges=4 graph; the supplementary row gives rho at "
              "extra_edges=3. The hop IQR 1/1/0 is taken from hopdist.py at "
              "the as-used densities (25x25 e=2, 50x50 e=3, 100x100 e=4)."),
    "raw_stdout": {"locality_control.py": out_l, "hopdist.py": out_h},
}
write("item_c_locality_control.json", obj)
print(obj["status"], regen, supp_rho)
