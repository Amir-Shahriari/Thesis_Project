"""Compare the recorded-stack 25x25 A/B re-run (this folder) with the original artefacts cited in the thesis.

Re-run 2026-10-10/11: scripts/run_multi_seed_warm_vs_cold.py --oracle-pool classical_only [--mask-visited],
torch 2.11.0+cu128 on an RTX 5080 (provenance_*.json). Run from the repo root: python runs/thesis_support/item_g_ab25_rerun/compare_ab25.py
"""
import json, math, pathlib, statistics

R = pathlib.Path(__file__).parent
TP = R.parents[1]
pairs = {
    "masked": (TP / "sweep_phase3_masked.json", R / "sweep_phase3_masked_rerun.json"),
    "reactive": (TP / "sweep_phase3_unmasked_ctrl.json", R / "sweep_phase3_unmasked_ctrl_rerun.json"),
}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def summary(rows):
    w = sum(bool(r["warm_strict"]) for r in rows)
    c = sum(bool(r["cold_strict"]) for r in rows)
    b = sum(bool(r["warm_strict"]) and not r["cold_strict"] for r in rows)
    cc = sum(bool(r["cold_strict"]) and not r["warm_strict"] for r in rows)
    both = [r for r in rows if r["warm_strict"] and r["cold_strict"]]
    wr = [r["warm_cost_ratio"] for r in both]
    cr = [r["cold_cost_ratio"] for r in both]
    return {"n": len(rows), "warm": w, "cold": c, "b": b, "c": cc, "mcnemar_p": mcnemar(b, cc),
            "both": len(both), "warm_med_ratio": round(statistics.median(wr), 2) if wr else None,
            "cold_med_ratio": round(statistics.median(cr), 2) if cr else None,
            "warm_cheaper": sum(x < y for x, y in zip(wr, cr))}


out = {}
for arm, (orig_p, new_p) in pairs.items():
    o = json.loads(orig_p.read_text())
    n = json.loads(new_p.read_text())
    key = lambda r: (r["seed"], r["scenario_id"])
    om, nm = {key(r): r for r in o}, {key(r): r for r in n}
    same_cells = sorted(set(om) & set(nm))
    dij_same = sum(abs((om[k]["dijkstra_cost"] or 0) - (nm[k]["dijkstra_cost"] or 0)) < 1e-6 for k in same_cells)
    agree_w = sum(bool(om[k]["warm_strict"]) == bool(nm[k]["warm_strict"]) for k in same_cells)
    agree_c = sum(bool(om[k]["cold_strict"]) == bool(nm[k]["cold_strict"]) for k in same_cells)
    out[arm] = {"original": summary(o), "rerun": summary(n), "cells_matched": len(same_cells),
                "dijkstra_identical": dij_same, "warm_reach_agree": agree_w, "cold_reach_agree": agree_c}
prov = {a: json.loads((R / f"provenance_{a}.json").read_text()) for a in ("masked", "reactive")}
out["stack"] = {a: {k: prov[a][k] for k in ("torch_version", "resolved_device", "gpu_name", "git_rev")} for a in prov}
(R / "ab25_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
