"""Compare the density-robust pilot checkpoints (demo_agents_ood/) against the
existing demo checkpoints (demo_agents/) for 50x50_s1, across the same
perturbation-multiplier levels as runs/ood_warm_vs_cold.json.

Same methodology as scripts/sweep_ood_warm_vs_cold.py (identical
_rebuild_graph / _dijkstra / _rollout), just run twice per level -- once per
checkpoint set -- so the two are directly comparable under the same graph
realisation.

Read-only: loads checkpoints and writes one JSON report. No training.

Usage:
    uv run python scripts/eval_ood_robust_pilot.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from demo_app.server import (  # noqa: E402
    _DEFAULT_NODE_DEACTIVATE_PROB,
    _dijkstra,
    _load_checkpoint,
    _rebuild_graph,
    _rollout,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "demo_agents" / "manifest.json"
BASELINE_DIR = ROOT / "demo_agents"
PILOT_DIR = ROOT / "demo_agents_ood"
OUT_PATH = ROOT / "runs" / "ood_robust_pilot.json"

CELL_ID = "50x50_s1"
SCENARIO_INDEX = 1
LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
UI_LEVELS = {0.5, 1.0, 2.0}


def rebuild_at(scale_entry: dict, grid_seed: int, mult: float):
    grid = dict(scale_entry["grid"])
    grid["deactivate_prob"] = min(0.95, grid.get("deactivate_prob", 0.15) * mult)
    base_node = grid.get("node_deactivate_prob", _DEFAULT_NODE_DEACTIVATE_PROB)
    grid["node_deactivate_prob"] = min(0.95, base_node * mult)
    return _rebuild_graph({**scale_entry, "grid": grid}, grid_seed)


def main() -> None:
    device = torch.device("cpu")
    manifest = json.loads(MANIFEST_PATH.read_text())
    scale_entry = manifest["50x50"]
    scenario = scale_entry["scenarios"][SCENARIO_INDEX]
    src, dst, grid_seed = scenario["source"], scenario["destination"], scenario["grid_seed"]

    checkpoint_sets = {
        "baseline": (BASELINE_DIR / "warm_50x50_s1.pt", BASELINE_DIR / "cold_50x50_s1.pt"),
        "pilot": (PILOT_DIR / "warm_50x50_s1.pt", PILOT_DIR / "cold_50x50_s1.pt"),
    }
    for label, (w, c) in checkpoint_sets.items():
        if not w.exists() or not c.exists():
            print(f"ABORT: {label} checkpoint missing ({w} / {c})")
            return

    records: list[dict] = []
    for label, (warm_path, cold_path) in checkpoint_sets.items():
        warm_agent = _load_checkpoint(warm_path, device)
        cold_agent = _load_checkpoint(cold_path, device)

        for mult in LEVELS:
            g = rebuild_at(scale_entry, grid_seed, mult)
            dij_cost, _ = _dijkstra(g, src, dst)
            solvable = dij_cost != float("inf")

            w = _rollout(warm_agent, g, src, dst)
            c = _rollout(cold_agent, g, src, dst)

            records.append({
                "checkpoint_set": label,
                "cell_id": CELL_ID,
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
            ratio = records[-1]["warm_ratio"]
            ratio_str = f", ratio={ratio:.2f}x" if ratio else ""
            print(
                f"{label:<9} {mult:>4}x  solvable={str(solvable):<5} "
                f"warm={'R' if w['reached'] else '.'}({w['steps']:>3} steps{ratio_str}) "
                f"cold={'R' if c['reached'] else '.'}({c['steps']:>3} steps)",
                flush=True,
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} records to {OUT_PATH}")

    _report(records)


def _report(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'level':>7} {'set':<10} {'warm reach':>10} {'warm steps':>10} "
          f"{'warm ratio':>10} {'cold reach':>10}")
    print("-" * 78)
    for mult in LEVELS:
        star = "*" if mult in UI_LEVELS else " "
        for label in ("baseline", "pilot"):
            rows = [r for r in records if r["mult"] == mult and r["checkpoint_set"] == label]
            if not rows:
                continue
            r = rows[0]
            wratio = f"{r['warm_ratio']:.2f}x" if r["warm_ratio"] else "--"
            print(f"{star}{mult:>5}x {label:<10} {str(r['warm_reached']):>10} "
                  f"{r['warm_steps']:>10} {wratio:>10} {str(r['cold_reached']):>10}")
    print("=" * 78)


if __name__ == "__main__":
    main()
