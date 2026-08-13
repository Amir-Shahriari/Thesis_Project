"""Sweep: does the warm-vs-cold contrast survive off-training perturbation?

The demo's explore tab exposes two off-training perturbation levels (0.5x and
2x). This script measures warm vs cold across a wider range than the UI offers,
so the claim "warm reaches more reliably than cold" can be checked against the
whole perturbation curve rather than the two levels the demo happens to show.

Method (mirrors demo_app/server.py::post_reach for perturb_multiplier != 1.0):
  - Same trained checkpoints and same grid_seed as the demo.
  - deactivate_prob and node_deactivate_prob are both scaled by the multiplier
    (capped at 0.95), then the graph is rebuilt via _rebuild_graph and the two
    agents are rolled out greedily (epsilon=0).
  - Node failure is scaled at BOTH scales: 25x25 omits node_deactivate_prob from
    its manifest grid, so the default (_DEFAULT_NODE_DEACTIVATE_PROB) is scaled
    rather than left untouched. Keying off key-presence would make "2x" mean
    edges+nodes at 50x50 but edges only at 25x25.

Attribution: every cell is Dijkstra-checked on the perturbed graph, so a
non-reaching agent is separated into "no path exists" (structurally unsolvable
-- not an agent failure) vs "agent did not reach". All reach rates below are
reported over SOLVABLE cells only, matching the convention in
scripts/audit_eval_reachability.py.

Caveat on reading the output: each (cell, level) is ONE random realisation, and
reach is not monotonic in the failure rate -- a cell can fail at 0.5x and
succeed at 2x by luck of the draw. With 5 cells per (scale, level), individual
fractions carry wide error bars. The robust signal is the direction, not the
exact counts.

Read-only: loads checkpoints and writes one JSON report. No training.

Usage:
    uv run python scripts/sweep_ood_warm_vs_cold.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from demo_app.server import (  # noqa: E402
    _DEFAULT_NODE_DEACTIVATE_PROB,
    _DEMO_AGENTS_DIR,
    _dijkstra,
    _load_checkpoint,
    _rebuild_graph,
    _rollout,
)

MANIFEST_PATH = pathlib.Path("demo_agents/manifest.json")
OUT_PATH = pathlib.Path("runs/ood_warm_vs_cold.json")

# 1.0 is the training level the paper's numbers are reported on. 0.5 and 2.0 are
# the two levels the demo UI exposes; 1.5 and 3.0 are included to show the curve
# either side of them (3.0 is deliberately NOT offered in the UI -- most cells
# have no path at all there).
LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
UI_LEVELS = {0.5, 1.0, 2.0}

# _rollout's own default; mirrored here so "wandered to the budget" is explicit.
STEP_BUDGET = 500


def rebuild_at(scale_entry: dict, grid_seed: int, mult: float):
    """Rebuild the eval graph with failure rates scaled by `mult`.

    Mirrors post_reach's OOD branch exactly.
    """
    grid = dict(scale_entry["grid"])
    grid["deactivate_prob"] = min(0.95, grid.get("deactivate_prob", 0.15) * mult)
    base_node = grid.get("node_deactivate_prob", _DEFAULT_NODE_DEACTIVATE_PROB)
    grid["node_deactivate_prob"] = min(0.95, base_node * mult)
    return _rebuild_graph({**scale_entry, "grid": grid}, grid_seed)


def main() -> None:
    device = torch.device("cpu")
    manifest = json.loads(MANIFEST_PATH.read_text())
    records: list[dict] = []

    for scale_key, scale_entry in manifest.items():
        for sc_idx, scenario in enumerate(scale_entry.get("scenarios", [])):
            cell_id = f"{scale_key}_s{sc_idx}"
            warm_pt = _DEMO_AGENTS_DIR / scenario["warm"]
            cold_pt = _DEMO_AGENTS_DIR / scenario["cold"]
            if not warm_pt.exists() or not cold_pt.exists():
                print(f"SKIP {cell_id}: checkpoint missing", flush=True)
                continue

            warm_agent = _load_checkpoint(warm_pt, device)
            cold_agent = _load_checkpoint(cold_pt, device)
            src, dst = scenario["source"], scenario["destination"]

            for mult in LEVELS:
                g = rebuild_at(scale_entry, scenario["grid_seed"], mult)

                dij_cost, _ = _dijkstra(g, src, dst)
                solvable = dij_cost != float("inf")

                w = _rollout(warm_agent, g, src, dst)
                c = _rollout(cold_agent, g, src, dst)

                records.append({
                    "cell_id": cell_id,
                    "scale": scale_key,
                    "mult": mult,
                    "in_ui": mult in UI_LEVELS,
                    "solvable": solvable,
                    "dijkstra_cost": dij_cost if solvable else None,
                    "warm_reached": w["reached"],
                    "cold_reached": c["reached"],
                    "warm_steps": w["steps"],
                    "cold_steps": c["steps"],
                    "warm_ratio": (w["cost"] / dij_cost) if (w["reached"] and solvable) else None,
                    "cold_ratio": (c["cost"] / dij_cost) if (c["reached"] and solvable) else None,
                    "warm_hit_budget": w["hit_step_budget"],
                    "cold_hit_budget": c["hit_step_budget"],
                })
                print(
                    f"{cell_id:<12} {mult:>4}x  solvable={str(solvable):<5} "
                    f"warm={'R' if w['reached'] else '.'}({w['steps']:>3}) "
                    f"cold={'R' if c['reached'] else '.'}({c['steps']:>3})",
                    flush=True,
                )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} records to {OUT_PATH}")

    _report(records, manifest)


def _report(records: list[dict], manifest: dict) -> None:
    # ── Reach, over solvable cells only ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("REACH over SOLVABLE cells  (* = level offered in the demo UI)")
    print(f"{'level':>7} {'scale':<8} {'solvable':>9} {'warm':>8} {'cold':>8} {'both fail':>10}")
    print("-" * 78)
    for mult in LEVELS:
        for scale in manifest:
            rows = [r for r in records if r["mult"] == mult and r["scale"] == scale]
            if not rows:
                continue
            solv = [r for r in rows if r["solvable"]]
            star = "*" if mult in UI_LEVELS else " "
            if not solv:
                print(f"{star}{mult:>5}x {scale:<8} {0:>4}/{len(rows):<4} "
                      f"{'--':>8} {'--':>8} {'--':>10}")
                continue
            wr = sum(r["warm_reached"] for r in solv)
            cr = sum(r["cold_reached"] for r in solv)
            both = sum(not r["warm_reached"] and not r["cold_reached"] for r in solv)
            print(f"{star}{mult:>5}x {scale:<8} {len(solv):>4}/{len(rows):<4} "
                  f"{wr:>4}/{len(solv):<3} {cr:>4}/{len(solv):<3} {both:>5}/{len(solv):<4}")

    # ── Unsolvable share: these are NOT agent failures ───────────────────────
    print("\n" + "=" * 78)
    print("STRUCTURALLY UNSOLVABLE (no path exists -- not an agent failure)")
    for mult in LEVELS:
        rows = [r for r in records if r["mult"] == mult]
        n_un = sum(not r["solvable"] for r in rows)
        star = "*" if mult in UI_LEVELS else " "
        print(f"{star}{mult:>5}x  {n_un}/{len(rows)}")

    # ── Wandering: budget exhaustion is the cold agent's signature failure ────
    print("\n" + "=" * 78)
    print(f"BUDGET EXHAUSTION (>= {STEP_BUDGET} steps) over solvable cells")
    for mult in LEVELS:
        for scale in manifest:
            solv = [r for r in records
                    if r["mult"] == mult and r["scale"] == scale and r["solvable"]]
            if not solv:
                continue
            wb = sum(r["warm_steps"] >= STEP_BUDGET for r in solv)
            cb = sum(r["cold_steps"] >= STEP_BUDGET for r in solv)
            wm = sum(r["warm_steps"] for r in solv) / len(solv)
            cm = sum(r["cold_steps"] for r in solv) / len(solv)
            star = "*" if mult in UI_LEVELS else " "
            print(f"{star}{mult:>5}x {scale:<8} warm {wb}/{len(solv)} "
                  f"(mean {wm:6.1f} steps) | cold {cb}/{len(solv)} (mean {cm:6.1f} steps)")

    # ── Head-to-head route quality, only where BOTH reached ──────────────────
    # This is the only situation where an attendee can compare optimality ratios
    # side by side, so it is where a warm-worse-than-cold cell becomes visible.
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD ROUTE QUALITY (both agents reached -- ratio comparison visible)")
    any_row = False
    for r in records:
        if not (r["warm_reached"] and r["cold_reached"]):
            continue
        any_row = True
        w, c = r["warm_ratio"], r["cold_ratio"]
        flag = "  <-- COLD CHEAPER" if c < w else ""
        star = "*" if r["in_ui"] else " "
        print(f"{star}{r['cell_id']:<12} {r['mult']:>4}x  "
              f"warm {w:8.2f}x ({r['warm_steps']:>3} steps) | "
              f"cold {c:8.2f}x ({r['cold_steps']:>3} steps){flag}")
    if not any_row:
        print("  (none)")
    print("=" * 78)


if __name__ == "__main__":
    main()
