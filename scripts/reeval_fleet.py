"""Re-evaluate trained fleet checkpoints on a MUCH larger held-out query set.

Why
---
The seed-42 rows evaluate on 20 held-out queries at 25-35% reach, so the set of
cells that BOTH arms solve -- the only cells a paired cost comparison can use --
comes out at 1 to 4. Every warm-vs-cold conclusion is then being drawn from
1-4 samples, which no amount of tuning will rescue.

Evaluation is a greedy rollout costing milliseconds per query, and it needs no
retraining, so widening the held-out set is nearly free. This script reloads
existing checkpoints and re-evaluates them on as many held-out queries as you
like.

Query identity is preserved: sample_scenarios draws grid_seed first and then
scenarios in order from the same generator, so asking for more scenarios
extends the list rather than changing it. The first n_train queries are exactly
the ones the agent trained on, and the original 20 held-out queries remain a
prefix of the widened set.

Usage
-----
    python scripts/reeval_fleet.py --seed 42 --scale 25x25 --n-eval-queries 150
    python scripts/reeval_fleet.py --seed 42 --scale 25x25 --tag-suffix _gr \
        --n-eval-queries 150
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.eval.scenario_sampler import sample_scenarios

ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_NODE_DEACT = 0.05


def load_agent(ckpt_path: pathlib.Path, device: str) -> GNNDQN:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    agent = GNNDQN(
        node_in_dim=int(ck.get("node_in_dim", 4)),
        hidden_dim=int(ck["hidden_dim"]),
        device=device, seed=0,
        goal_relative=bool(ck.get("goal_relative", False)),
    )
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    return agent


def rebuild_graph(grid: dict, grid_seed: int, n_iterations: int) -> DynamicGraph:
    g = DynamicGraph(
        grid_width=grid["grid_width"], grid_height=grid["grid_height"],
        extra_edges=grid.get("extra_edges", 2),
        deactivate_prob=grid.get("deactivate_prob", 0.15),
        node_deactivate_prob=grid.get("node_deactivate_prob", _DEFAULT_NODE_DEACT),
        seed=grid_seed,
    )
    for i in range(1, n_iterations + 1):
        g.update_graph(iteration=i)
    return g


def evaluate(agent, g, queries, k_threshold):
    data = dynamic_graph_to_pyg(g, device=agent.device)
    rows = []
    for src, dst in queries:
        v = evaluate_with_reasonableness(
            g, agent, src, dst, k_threshold=k_threshold, data=data
        )
        rows.append({
            "source": src, "destination": dst,
            "reached": bool(v.reached_goal_strict),
            "reasonable": bool(v.reached_goal_reasonable),
            "cost": None if v.cost == float("inf") else float(v.cost),
            "cost_ratio": v.cost_ratio,
            "dijkstra_cost": v.dijkstra_reference_cost,
            "solvable": v.dijkstra_reference_cost not in (None, float("inf")),
        })
    return rows


def paired_report(warm_rows, cold_rows, label):
    """Warm-vs-cold on the cells both arms solved -- the only fair cost test."""
    W = {(r["source"], r["destination"]): r for r in warm_rows}
    C = {(r["source"], r["destination"]): r for r in cold_rows}
    keys = sorted(set(W) & set(C))
    solvable = [k for k in keys if W[k]["solvable"]]
    wr = {k for k in solvable if W[k]["reached"]}
    cr = {k for k in solvable if C[k]["reached"]}
    both = sorted(wr & cr)

    print(f"\n  --- {label} ---")
    print(f"  queries {len(keys)}  solvable {len(solvable)}  "
          f"(unsolvable at eval state: {len(keys) - len(solvable)})")
    print(f"  reach   warm {len(wr):3d}/{len(solvable)}  cold {len(cr):3d}/{len(solvable)}"
          f"   both {len(both):3d}  warm-only {len(wr - cr):3d}  cold-only {len(cr - wr):3d}")

    out = {
        "n_queries": len(keys), "n_solvable": len(solvable),
        "warm_reach": len(wr), "cold_reach": len(cr),
        "n_both": len(both), "warm_only": len(wr - cr), "cold_only": len(cr - wr),
    }

    # McNemar on discordant reach pairs
    b, c = len(wr - cr), len(cr - wr)
    if b + c > 0:
        from scipy.stats import binomtest
        p = binomtest(b, b + c, 0.5).pvalue
        print(f"  reach McNemar (exact): warm-only {b} vs cold-only {c}  p={p:.4f}")
        out["reach_mcnemar_p"] = float(p)

    if len(both) >= 3:
        w = np.array([W[k]["cost_ratio"] for k in both], dtype=float)
        c_ = np.array([C[k]["cost_ratio"] for k in both], dtype=float)
        ok = np.isfinite(w) & np.isfinite(c_)
        w, c_ = w[ok], c_[ok]
        if w.size >= 3:
            try:
                p = float(wilcoxon(w, c_).pvalue)
            except ValueError:
                p = float("nan")
            print(f"  PAIRED cost on {w.size} commonly-reached: "
                  f"warm median {np.median(w):7.2f}  cold median {np.median(c_):7.2f}  "
                  f"warm better {int((w < c_).sum())}/{w.size}  wilcoxon p={p:.4f}")
            out.update({
                "paired_n": int(w.size),
                "warm_median_ratio": float(np.median(w)),
                "cold_median_ratio": float(np.median(c_)),
                "warm_better": int((w < c_).sum()),
                "paired_wilcoxon_p": p,
            })
    else:
        print(f"  PAIRED cost: only {len(both)} commonly-reached cells -- "
              f"too few to test. Raise --n-eval-queries.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scale", default="25x25")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--n-train-queries", type=int, default=20)
    ap.add_argument("--n-eval-queries", type=int, default=150)
    ap.add_argument("--k-threshold", type=float, default=3.0)
    ap.add_argument("--fleet-dir", default="runs/fleet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fleet = pathlib.Path(args.fleet_dir)
    if not fleet.is_absolute():
        fleet = ROOT / fleet
    tag = f"fleet_{args.scale}_seed{args.seed}{args.tag_suffix}"
    res_path = fleet / f"fleet_results_seed{args.seed}_{args.scale}{args.tag_suffix}.json"
    if not res_path.exists():
        raise SystemExit(f"no results file at {res_path}")
    orig = json.loads(res_path.read_text())
    cfg = orig["config"]
    grid_seed = orig["grid_seed"]
    n_iters = cfg["train"]["n_iterations"]

    # Re-derive the query list. Same seed + same grid template => identical
    # prefix, so training queries are unchanged and the original held-out set
    # is a prefix of the widened one.
    rng = np.random.default_rng(args.seed)
    total = args.n_train_queries + args.n_eval_queries
    scenarios = sample_scenarios(cfg["grid"], n_scenarios=total, rng=rng,
                                 min_euclidean_fraction=0.6)
    assert scenarios[0].grid_seed == grid_seed, (
        f"grid_seed mismatch ({scenarios[0].grid_seed} vs {grid_seed}); "
        "the scenario stream is not reproducing the trained configuration"
    )
    train_q = [(s.source_node, s.destination_node) for s in scenarios[:args.n_train_queries]]
    eval_q = [(s.source_node, s.destination_node) for s in scenarios[args.n_train_queries:]]

    g = rebuild_graph(cfg["grid"], grid_seed, n_iters)
    print(f"=== re-eval {tag} ===")
    print(f"    ablation : {orig.get('ablation', {})}")
    print(f"    train queries {len(train_q)}   held-out {len(eval_q)} "
          f"(was {orig['n_eval_queries']})")

    rows: dict = {}
    for arm in ("warm", "cold"):
        ck = fleet / "agents" / f"{tag}_{arm}.pt"
        if not ck.exists():
            print(f"    missing checkpoint {ck}; skipping {arm}")
            continue
        agent = load_agent(ck, args.device)
        rows[arm] = {
            "train": evaluate(agent, g, train_q, args.k_threshold),
            "held_out": evaluate(agent, g, eval_q, args.k_threshold),
        }

    summary: dict = {"tag": tag, "ablation": orig.get("ablation", {}),
                     "grid_seed": grid_seed,
                     "n_train_queries": len(train_q), "n_eval_queries": len(eval_q)}
    if "warm" in rows and "cold" in rows:
        summary["train"] = paired_report(rows["warm"]["train"], rows["cold"]["train"],
                                         "TRAIN QUERIES")
        summary["held_out"] = paired_report(rows["warm"]["held_out"],
                                            rows["cold"]["held_out"], "HELD-OUT")

    out = pathlib.Path(args.out) if args.out else (
        fleet / f"reeval_seed{args.seed}_{args.scale}{args.tag_suffix}.json")
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
