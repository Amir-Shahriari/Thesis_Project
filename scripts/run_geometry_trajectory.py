"""Geometry over training: does the representation degrade while reward rises?

The snapshot checkpoints written by scripts/run_fleet_mode.py capture the
encoder at the end of every training iteration. This script walks them, computes
the geometry panel at each one, and joins it against that iteration's
goal_reach_rate from the same run's training log.

That join is the whole point. A single end-of-training snapshot can show that
the representation is collapsed; only the time series can show the collapse
happening *while the reward curve is still climbing*, which is what "silent
failure" actually claims. No published GNN-RL work reports this pairing.

GRAPH STATE (easy to get wrong)
------------------------------
In train_gnn_dqn the callback for iteration k fires AFTER
update_graph(iteration=k+1), so the snapshot at iteration k corresponds to a
graph that has been perturbed k+1 times from its initial state. The final
checkpoint, saved after the loop, has n_iterations perturbations. Rebuilding a
snapshot against the wrong perturbation count measures the geometry on a graph
the encoder never saw.

Usage
-----
    python scripts/run_geometry_trajectory.py \
        --manifest runs/fleet/agents/manifest_fleet_25x25_seed42.json \
        --results  runs/fleet/fleet_results_seed42_25x25.json \
        --out      runs/geometry_trajectory_seed42_25x25.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.geometry import all_metrics, locality_spearman

from run_geometry_audit import load_agent, locality_pairs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_NODE_DEACT = 0.05
_SNAP_RE = re.compile(r"^(?P<tag>.+)_(?P<arm>warm|cold)_it(?P<it>\d+)\.pt$")


def graph_at_iteration(grid: dict, grid_seed: int, n_perturbations: int) -> DynamicGraph:
    """Rebuild the graph after exactly n_perturbations update_graph() calls."""
    g = DynamicGraph(
        grid_width=grid["grid_width"],
        grid_height=grid["grid_height"],
        extra_edges=grid.get("extra_edges", 2),
        deactivate_prob=grid.get("deactivate_prob", 0.15),
        node_deactivate_prob=grid.get("node_deactivate_prob", _DEFAULT_NODE_DEACT),
        seed=grid_seed,
    )
    for i in range(1, n_perturbations + 1):
        g.update_graph(iteration=i)
    return g


def panel_for(ckpt: pathlib.Path, grid: dict, grid_seed: int, hidden_dim: int,
              n_perturbations: int, rng) -> dict:
    g = graph_at_iteration(grid, grid_seed, n_perturbations)
    data = dynamic_graph_to_pyg(g)
    agent = load_agent(ckpt, hidden_dim)
    with torch.no_grad():
        H = agent._encoder_raw(data.x, data.edge_index)
    panel = all_metrics(H, rng=rng)
    pairs, hops = locality_pairs(g, data, rng=rng)
    panel["locality_rho"] = locality_spearman(H, hops, pairs)["rho"]
    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="manifest written by run_fleet_mode.py")
    ap.add_argument("--results", default=None,
                    help="fleet_results_*.json for the same run (supplies the "
                         "per-iteration goal_reach_rate to join against)")
    ap.add_argument("--snapshots-dir", default=None,
                    help="default: <manifest dir>/snapshots")
    ap.add_argument("--out", default="runs/geometry_trajectory.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man_path = pathlib.Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    manifest = json.loads(man_path.read_text())
    entry = next(iter(manifest.values()))
    grid, grid_seed = entry["grid"], entry["grid_seed"]
    hidden_dim = entry["hidden_dim"]
    n_iterations = entry["train"]["n_iterations"]

    snap_dir = (pathlib.Path(args.snapshots_dir) if args.snapshots_dir
                else man_path.parent / "snapshots")
    if not snap_dir.is_absolute():
        snap_dir = ROOT / snap_dir
    if not snap_dir.is_dir():
        raise SystemExit(f"no snapshots directory at {snap_dir}")

    # Per-iteration goal-reach from the training log, keyed by arm.
    reach: dict[str, list] = {}
    if args.results:
        rp = pathlib.Path(args.results)
        if not rp.is_absolute():
            rp = ROOT / rp
        if rp.exists():
            res = json.loads(rp.read_text())
            for arm, blk in res.get("arms", {}).items():
                reach[arm] = blk.get("training_log", {}).get("goal_reach_rate", [])

    snaps: dict[str, list[tuple[int, pathlib.Path]]] = {}
    for p in sorted(snap_dir.glob("*.pt")):
        m = _SNAP_RE.match(p.name)
        if m:
            snaps.setdefault(m.group("arm"), []).append((int(m.group("it")), p))
    if not snaps:
        raise SystemExit(f"no snapshots matching *_it###.pt in {snap_dir}")

    rng = np.random.default_rng(args.seed)
    out: dict = {
        "manifest": str(man_path), "grid_seed": grid_seed,
        "n_iterations": n_iterations, "series": {},
    }

    for arm, items in snaps.items():
        items.sort()
        rows = []
        print(f"\n=== {arm} ({len(items)} snapshots) ===", flush=True)
        print(f"{'iter':>5} {'reach':>7} {'IsoScore':>9} {'erank':>7} "
              f"{'cos':>7} {'loc_rho':>8}")
        for it, path in items:
            # callback for iteration k fires after update_graph(k+1)
            panel = panel_for(path, grid, grid_seed, hidden_dim, it + 1, rng)
            gr = reach.get(arm, [])
            panel["iteration"] = it
            panel["goal_reach_rate"] = float(gr[it]) if it < len(gr) else None
            rows.append(panel)
            g_str = ("  n/a" if panel["goal_reach_rate"] is None
                     else f"{panel['goal_reach_rate']:.3f}")
            print(f"{it:>5} {g_str:>7} {panel['isoscore']:>9.4f} "
                  f"{panel['effective_rank']:>7.1f} "
                  f"{panel['mean_pair_cosine']:>7.3f} "
                  f"{panel['locality_rho']:>8.3f}", flush=True)
        out["series"][arm] = rows

        # The claim this script exists to test, stated as a number.
        gr = [r["goal_reach_rate"] for r in rows if r["goal_reach_rate"] is not None]
        iso = [r["isoscore"] for r in rows if r["goal_reach_rate"] is not None]
        er = [r["effective_rank"] for r in rows if r["goal_reach_rate"] is not None]
        if len(gr) >= 3:
            d_reach = gr[-1] - gr[0]
            d_iso = iso[-1] - iso[0]
            d_er = er[-1] - er[0]
            silent = d_reach > 0 and (d_iso < 0 or d_er < 0)
            out["series"][arm + "_summary"] = {
                "delta_goal_reach": d_reach,
                "delta_isoscore": d_iso,
                "delta_effective_rank": d_er,
                "silent_degradation": bool(silent),
            }
            verdict = ("SILENT DEGRADATION: reward up, geometry down" if silent
                       else "no silent degradation on this run")
            print(f"  Delta over training: reach {d_reach:+.3f}  "
                  f"IsoScore {d_iso:+.4f}  erank {d_er:+.1f}")
            print(f"  -> {verdict}")

    op = pathlib.Path(args.out)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {op}")


if __name__ == "__main__":
    main()
