"""Item (d): optimal discounted returns and reward calibration."""
import pathlib
import re

from prov import REPO, SCRATCH, provenance, run_copy, write, git

A = REPO / "scripts" / "analysis"
DEMO = pathlib.Path(r"C:\Users\amirh\Desktop\Demo")
cmd1, out1 = run_copy("step1_returns.py")
cmd1b, out1b = run_copy("step1b_nonleak.py")
cmd23, out23 = run_copy("step23_calibrate.py")
cmdb, outb = run_copy("blockers.py")
cmda, outa = run_copy("analyze_invalid_action_returns.py")

# ---- step1_returns ---------------------------------------------------------
s1 = {}
for blk in out1.split("=" * 74)[1::2]:
    pass
for m in re.finditer(r"\n(\d+x\d+, extra_edges=\d+)[^\n]*\((\d+) solvable pairs\)\n=+\n(.*?)(?=\n=+\n|\Z)", out1, re.S):
    label, body = m[1], m[3]
    rows = {}
    for r in re.finditer(r"^\s+([\d.]+) \|\s+([-\d.]+) /\s*(\d+)/(\d+) \|\s+([-\d.]+) /\s*(\d+)/(\d+)", body, re.M):
        rows[r[1]] = {"gamma=0.95": {"median_return": float(r[2]), "beats_minus5": f"{r[3]}/{r[4]}"},
                      "gamma=0.99": {"median_return": float(r[5]), "beats_minus5": f"{r[6]}/{r[7]}"}}
    hops = re.search(r"median (\d+) hops \(max (\d+)\), median composite cost (\d+)", body)
    s1[label] = {"n_pairs": int(m[2]), "median_hops": int(hops[1]), "max_hops": int(hops[2]),
                 "median_composite_cost": int(hops[3]), "by_lambda_shape": rows}

# ---- step23 1c / 3 ---------------------------------------------------------
c1 = {}
for r in re.finditer(r"^\s+(25x25 as-used|25x25 pure grid|50x50 pure grid)\s+([-\d.]+)\s+(\d+)/20\s+([-\d.]+)\s+(\d+)/20\s+([-\d.]+)\s+(\d+)/20", out23, re.M):
    c1[r[1]] = {g: {"median_return": float(r[2 + 2 * i]), "beats_minus5": f"{r[3 + 2 * i]}/20"}
                for i, g in enumerate(("gamma=0.95", "gamma=0.99", "gamma=0.995"))}
dens = []
for r in re.finditer(r"^\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$",
                     out23.split("STEP 2")[1].split("STEP 3")[0], re.M):
    dens.append({"n_chords": int(r[1]), "edges": int(r[2]), "mean_degree": float(r[3]),
                 "diam_sampled_50_sources": int(r[4]), "mean_hops": float(r[5]), "hop_iqr": float(r[6])})
solv = {}
for r in re.finditer(r"^\s+(\d+)\s+(\d+) \|\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%", out23, re.M):
    solv[r[1]] = {"diam": int(r[2]), "p=0.05": int(r[3]), "p=0.10": int(r[4]),
                  "p=0.15": int(r[5]), "p=0.30": int(r[6])}

# ---- blockers BLOCKER 2 ----------------------------------------------------
b2 = []
for r in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)x\s+(\d+)\s+([-\d.]+)\s+(\d+)/25", outb, re.M):
    b2.append({"n_chords": int(r[1]), "mean_opt_cost_normalised": float(r[2]),
               "fixed_bonus100_to_opt_cost_ratio": float(r[3]), "beta5_bonus": int(r[4]),
               "median_return_beta5": float(r[5]), "beats_minus5": f"{r[6]}/25"})

# ---- Demo analyze_invalid_action_returns ----------------------------------
per_scen, means = [], []
for r in re.finditer(r"^(\d+x\d+)\s+(fresh|perturbed)\s+(seed\d+_s\d)\s+(\d+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+-5\.0$", outa, re.M):
    per_scen.append({"scale": r[1], "state": r[2], "scenario": r[3], "T": int(r[4]),
                     "opt_cost": float(r[5]), "G_0.95": float(r[7]), "G_0.99": float(r[8]),
                     "G_0.997": float(r[9])})
for r in re.finditer(r"^(\d+x\d+)\s+(fresh|perturbed)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+g=", outa, re.M):
    means.append({"scale": r[1], "state": r[2], "n": int(r[3]), "mean_T": float(r[4]),
                  "mean_opt_cost": float(r[5]), "mean_G_0.95": float(r[6]),
                  "mean_G_0.99": float(r[7]), "mean_G_0.997": float(r[8])})
g95 = [m["mean_G_0.95"] for m in means]
gall = [m[k] for m in means for k in ("mean_G_0.95", "mean_G_0.99", "mean_G_0.997")]
g099 = [m["mean_G_0.99"] for m in means]
m100 = [m[k] for m in means if m["scale"] == "100x100" for k in ("mean_G_0.95", "mean_G_0.99", "mean_G_0.997")]
min_individual = min(min(p["G_0.95"], p["G_0.99"], p["G_0.997"]) for p in per_scen)

regen = {
    "optimal_return_as_used_25x25_gamma0.95_median": s1["25x25, extra_edges=2"]["by_lambda_shape"]["0.0"]["gamma=0.95"],
    "optimal_return_pure_grid_25x25_gamma0.95_median": s1["25x25, extra_edges=0"]["by_lambda_shape"]["0.0"]["gamma=0.95"],
    "normalised_cost_gamma0.99_beats_minus5": {k: v["gamma=0.99"]["beats_minus5"] for k, v in c1.items()},
    "normalised_cost_gamma0.95_50x50_pure": c1["50x50 pure grid"]["gamma=0.95"]["beats_minus5"],
    "bonus_to_opt_cost_ratio_range": [min(b["fixed_bonus100_to_opt_cost_ratio"] for b in b2),
                                      max(b["fixed_bonus100_to_opt_cost_ratio"] for b in b2)],
    "solvable_pct_p0.15_by_n_chords": {k: v["p=0.15"] for k, v in solv.items()},
    "solvable_pct_p0.30_by_n_chords": {k: v["p=0.30"] for k, v in solv.items()},
    "demo_script_scale_means_gamma0.95_range": [min(g95), max(g95)],
    "demo_script_scale_means_gamma0.99_range": [min(g099), max(g099)],
    "demo_script_scale_means_all_gammas_range": [min(gall), max(gall)],
    "demo_script_100x100_means_all_gammas_range": [min(m100), max(m100)],
    "demo_script_min_individual_return_any_gamma": min_individual,
}
sub = {
    "+50.3 (Ch5)": ["REPRODUCED", "step1_returns.py: median over 20 sampled pairs, 25x25 extra_edges=2, unperturbed graph, lambda_shape=0, gamma=0.95 -> 50.3 (20/20 beat -5)"],
    "-96.4 at E_extra=0 (Ch4, Ch5)": ["REPRODUCED", "step1_returns.py: 25x25 extra_edges=0, gamma=0.95 median -96.4, 2/20 beat -5"],
    "20/20 sampled queries at every density (Ch5)": ["REPRODUCED", "step23_calibrate.py STEP 1c: normalised cost, gamma=0.99 -> 20/20 for 25x25 as-used, 25x25 pure grid, 50x50 pure grid (gamma=0.95 gives 12/20 at 50x50, as RUN_LOG says). 'Every density' = these three configurations, not the n_chords axis."],
    "4.7x to 21.9x (Ch5)": ["REPRODUCED", "blockers.py BLOCKER 2 ('fixed bonus 100' column = 100 / mean normalised optimal cost), n_chords 0 -> 1250 at 25x25: 4.7x, 10.2x, 11.8x, 16.2x, 21.9x"],
    "98-99% at p_deact=0.15; 91% vs 100% at p_deact=0.30 (Ch5)": ["REPRODUCED", "step23_calibrate.py STEP 3: p=0.15 -> 99/98/99/99% (n_chords 0/125/400/1250); p=0.30 -> 91% at n_chords=0 vs 100% at 1250"],
    "+13 to +48 at every scale and every gamma (Ch4)": ["DIFFERENT", "Demo/scripts/analyze_invalid_action_returns.py (5 scenarios per scale, outer seed 42, grid seed 191664964, fresh and perturbed states): the per-scale/state MEAN optimal returns span +13.39 to +48.03 at gamma=0.95 ONLY. At gamma=0.99 they span +28.48 to +61.21, and over all three gammas +13.39 to +63.76. At 100x100 alone (reward_variants.md's wording) they span +13.39 to +57.92 (+13.39 to +37.38 at gamma=0.95). The qualitative claim holds: every individual optimal return exceeds -5 at every scale and gamma (minimum -1.24, 100x100 perturbed s3, gamma=0.95). The numeric range is the gamma=0.95 range over all three scales, not 'every gamma' and not '100x100'."],
}
obj = {
    "item": "d",
    "description": "Optimal discounted returns and reward calibration (desk calibration 2026-08-09).",
    "thesis_locations": ["ch4-experiments.tex:126", "ch5-conclusion.tex:50"],
    "audit_findings": ["N5-01", "N5-06", "N9-10", "N10-15", "N10-17"],
    "provenance": provenance(
        [A / "step1_returns.py", A / "step1b_nonleak.py", A / "step23_calibrate.py", A / "blockers.py"],
        [cmd1, cmd1b, cmd23, cmdb, cmda],
        [SCRATCH / n for n in ("step1_returns.py", "step1b_nonleak.py", "step23_calibrate.py",
                               "blockers.py", "analyze_invalid_action_returns.py")],
        {"demo_script": {"path": str(DEMO / "scripts" / "analyze_invalid_action_returns.py"),
                         "demo_git_revision": git(DEMO, "rev-parse", "HEAD"),
                         "scratch_copy_modification": "inserted 'import sys; sys.path.insert(0, r\"C:\\Users\\amirh\\Desktop\\Thesis_Project\\src\")' after 'import heapq'; scenario_sampler.py is identical in Demo and Thesis_Project"}}),
    "parameters": {
        "graph_seed": 191664964,
        "step1_returns.py": {"configs": ["25x25 e=2", "25x25 e=0", "50x50 e=0"], "pair_rng": "default_rng(7)", "n_pairs": 20, "gammas": [0.95, 0.99], "lambda_shape": [0, 0.5, 1, 2, 4, 8], "goal_bonus": 100},
        "step1b_nonleak.py": {"gamma": 0.95, "n_pairs": 20, "pair_rng": "default_rng(7)"},
        "step23_calibrate.py": {"step1c_gammas": [0.95, 0.99, 0.995], "density_n_chords": [0, 60, 125, 250, 400, 625, 1250], "diam_n_src": 50, "diam_rng": "default_rng(0)", "solvability": {"n_chords": [0, 125, 400, 1250], "p_deact": [0.05, 0.10, 0.15, 0.30], "graph_seeds": [1, 2, 3], "perturbation_rounds": 5, "pairs": 60, "pair_rng": "default_rng(11)"}},
        "blockers.py BLOCKER 2": {"n_chords": [0, 60, 125, 400, 1250], "pair_rng": "default_rng(7)", "n_paths": 25, "beta": 5.0, "gamma": 0.99},
        "analyze_invalid_action_returns.py": {"outer_seed": 42, "n_scenarios": 5, "gammas": [0.95, 0.99, 0.997], "scale_cfgs": {"25x25": "e=2, p=0.10", "50x50": "e=3, p=0.22", "100x100": "e=4, p=0.30"}, "states": ["fresh", "perturbed (one update_graph)"]},
    },
    "results": {
        "step1_returns": s1,
        "step23_step1c_normalised": c1,
        "step23_density_table": dens,
        "step23_solvability_pct": solv,
        "blockers_blocker2": b2,
        "analyze_invalid_action_returns": {"per_scenario": per_scen, "means": means},
        "regenerated_thesis_quantities": regen,
        "sub_item_status": sub,
    },
    "thesis_values": {"optimal_return_as_used": 50.3, "optimal_return_pure_grid": -96.4,
                      "optimal_return_range_every_scale_every_gamma": [13, 48],
                      "bonus_ratio_range": [4.7, 21.9], "solvable_p0.15_pct": [98, 99],
                      "solvable_p0.30_sparse_vs_dense_pct": [91, 100], "queries_beating_minus5": "20/20"},
    "status": "DIFFERENT",
    "status_note": "All sub-items reproduce exactly except '+13 to +48 at every scale and every gamma', which holds only for gamma=0.95 (see sub_item_status).",
    "raw_stdout": {"step1_returns.py": out1, "step1b_nonleak.py": out1b, "step23_calibrate.py": out23,
                   "blockers.py": outb, "analyze_invalid_action_returns.py": outa},
}
write("item_d_reward_calibration.json", obj)
import json
print(json.dumps(regen, indent=1))
