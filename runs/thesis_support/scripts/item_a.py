"""Item (a): uniform-rollout exploration statistics, reactive vs masked rule."""
import re
import sys

from prov import REPO, SCRATCH, PY, provenance, run_copy, write

A = REPO / "scripts" / "analysis"
cmd_b, out_b = run_copy("blockers.py")
cmd_r, out_r = run_copy("revisit_rules.py")

# --- blockers.py, BLOCKER 1 -------------------------------------------------
b1 = []
for m in re.finditer(r"n_chords=\s*(\d+)\s+degree ([\d.]+), diam (\d+) \| (REACTIVE|MASKED)\s+"
                     r"goal\s+([\d.]+)%\s+revisit-death\s+([\d.]+)%\s+dead-end\s+([\d.]+)%\s+"
                     r"median steps (\d+)", out_b):
    b1.append({"n_chords": int(m[1]), "rule": m[4].lower(),
               "goal_pct": float(m[5]), "revisit_death_pct": float(m[6]),
               "dead_end_pct": float(m[7]), "median_steps": int(m[8])})

# --- revisit_rules.py -------------------------------------------------------
rr = {}
for block in re.split(r"={72}\n(n_chords=\d+)", out_r)[1:]:
    pass
blocks = re.findall(r"(n_chords=\d+)[^\n]*\n=+\n\s+rule[^\n]*\n((?:\s+\w+\s+[\d.%\s]+\n)+)", out_r)
for lab, body in blocks:
    nc = int(lab.split("=")[1])
    rr[nc] = {}
    for line in body.strip().splitlines():
        parts = line.split()
        rr[nc][parts[0]] = {"eps=1.0": float(parts[1].rstrip("%")),
                            "eps=0.4": float(parts[2].rstrip("%")),
                            "eps=0.05": float(parts[3].rstrip("%"))}

dense_reactive = next(r for r in b1 if r["n_chords"] == 1250 and r["rule"] == "reactive")
regen = {
    "reactive_revisit_death_pct_uniform_nchords1250": dense_reactive["revisit_death_pct"],
    "reactive_median_steps_uniform_nchords1250": dense_reactive["median_steps"],
    "goal_rate_reactive_eps1_nchords1250_pct": rr[1250]["reactive"]["eps=1.0"],
    "goal_rate_masked_eps1_nchords1250_pct": rr[1250]["mask"]["eps=1.0"],
    "goal_rate_reactive_eps0.05_dijkstra_greedy_nchords1250_pct": rr[1250]["reactive"]["eps=0.05"],
}
thesis = {
    "reactive_revisit_death_pct_uniform": 98.2,
    "median_steps": 5,
    "goal_rate_reactive_eps1_pct": 1.5,
    "goal_rate_masked_eps1_pct": 51.2,
    "goal_rate_reactive_eps0.05_pct": 89.2,
}
ok = (regen["reactive_revisit_death_pct_uniform_nchords1250"] == 98.2
      and regen["reactive_median_steps_uniform_nchords1250"] == 5
      and regen["goal_rate_reactive_eps1_nchords1250_pct"] == 1.5
      and regen["goal_rate_masked_eps1_nchords1250_pct"] == 51.2
      and regen["goal_rate_reactive_eps0.05_dijkstra_greedy_nchords1250_pct"] == 89.2)

obj = {
    "item": "a",
    "description": ("Uniform-rollout exploration statistics under the reactive "
                    "vs masked action rule (thesis: 98.2% of uniform rollouts "
                    "end on a revisit within a median of five steps; goal rate "
                    "1.5% reactive vs 51.2% masked at eps=1; 89.2% at eps=0.05 "
                    "with the Dijkstra next hop as greedy action)."),
    "thesis_locations": ["front-1-abstract.tex:7", "ch1-introduction.tex:115",
                         "ch3-methodology.tex:68", "ch4-experiments.tex:84",
                         "ch4-experiments.tex:109", "ch5-conclusion.tex:9"],
    "audit_findings": ["N4-01"],
    "provenance": provenance([A / "blockers.py", A / "revisit_rules.py"],
                             [cmd_b, cmd_r],
                             [SCRATCH / "blockers.py", SCRATCH / "revisit_rules.py"]),
    "parameters": {
        "graph": "DynamicGraph(grid_width=25, grid_height=25, extra_edges=0, n_chords=<nc>, seed=191664964)",
        "n_chords": {"blockers.py": [1250, 60, 0], "revisit_rules.py": [0, 1250]},
        "graph_seed": 191664964,
        "blockers.py": {"pair_rng": "np.random.default_rng(3)", "n_pairs": 20,
                        "rollouts_per_pair": 100, "max_steps": 400,
                        "policy": "uniform random (eps=1)",
                        "median_steps": "median over ALL rollouts of steps taken before termination"},
        "revisit_rules.py": {"pair_rng": "np.random.default_rng(3)", "n_pairs": 15,
                             "rollouts_per_pair_per_cell": 80, "max_steps": 400,
                             "eps": [1.0, 0.4, 0.05],
                             "greedy_action": "Dijkstra-optimal next hop (oracle upper bound)",
                             "rules": ["reactive", "mask", "fallback"]},
    },
    "source_mapping": {
        "98.2% / median 5 steps": "blockers.py BLOCKER 1, n_chords=1250, REACTIVE row",
        "1.5% -> 51.2%": "revisit_rules.py, n_chords=1250, eps=1.0, reactive vs mask rows",
        "89.2%": "revisit_rules.py, n_chords=1250, eps=0.05, reactive row",
        "note": ("The 'as-used density' here is n_chords=1250 total chords "
                 "(the density-axis analogue of the legacy extra_edges=2 "
                 "preset, ~1,236 chords), not the extra_edges=2 graph itself. "
                 "blockers.py's own uniform goal rates at n_chords=1250 are "
                 "1.9% reactive / 49.8% masked (different pair sample, 20x100 "
                 "rollouts); the thesis's 1.5% / 51.2% are the revisit_rules.py "
                 "values (15x80 rollouts)."),
    },
    "results": {
        "blockers_blocker1": b1,
        "revisit_rules_goal_rate_pct": {str(k): v for k, v in rr.items()},
        "regenerated_thesis_quantities": regen,
    },
    "thesis_values": thesis,
    "status": "REPRODUCED" if ok else "DIFFERENT",
    "raw_stdout": {"blockers.py": out_b, "revisit_rules.py": out_r},
}
write("item_a_exploration_rules.json", obj)
print(obj["status"], regen)
