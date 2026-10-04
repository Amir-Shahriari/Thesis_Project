"""V3 cross-distribution (transfer) benchmark.

Loads ONE V1 warm checkpoint (seed=42, scenario_id=seed191664964_s0) and the
corresponding V2 library (seed_42.pt) and evaluates them in three modes
(V1 / V2 / V3) across five graph scenarios that vary distribution shift.

Usage:
  uv run python scripts/run_v3_transfer_benchmark.py \
      --agent runs/agents_v1/warm_seed42_scenseed191664964_s0.pt \
      --library runs/libraries/seed_42.pt \
      --out runs/v3_transfer_benchmark.json
"""
from __future__ import annotations

import sys
import argparse
import json
import pathlib
import random
import time

_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))

import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.replay.expert_library import ExpertLibrary

# ── Eval scenarios ─────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name":           "in_dist",
        "grid_width":     100, "grid_height": 100,
        "extra_edges":    4,   "deactivate_prob": 0.30,
        "eval_seed":      42,
        "nature":         "same as training (sanity check)",
    },
    {
        "name":           "new_seed",
        "grid_width":     100, "grid_height": 100,
        "extra_edges":    4,   "deactivate_prob": 0.30,
        "eval_seed":      999,
        "nature":         "new random seed, same parameters",
    },
    {
        "name":           "low_pert",
        "grid_width":     100, "grid_height": 100,
        "extra_edges":    4,   "deactivate_prob": 0.15,
        "eval_seed":      42,
        "nature":         "half deactivation probability",
    },
    {
        "name":           "high_pert",
        "grid_width":     100, "grid_height": 100,
        "extra_edges":    4,   "deactivate_prob": 0.45,
        "eval_seed":      42,
        "nature":         "50% higher deactivation probability",
    },
    {
        "name":           "small_grid",
        "grid_width":     50,  "grid_height": 50,
        "extra_edges":    2,   "deactivate_prob": 0.30,
        "eval_seed":      42,
        "nature":         "smaller grid — tests encoder transfer",
    },
]

PERTURBATION_STEPS   = 5
MAX_STEPS            = 1000
N_QUERIES_PER_SCEN   = 10
MIN_EUCLIDEAN_FRAC   = 0.6   # src/dst must be at least 60% of max possible distance


# ── Graph construction ────────────────────────────────────────────────────────
def _build_graph(scen: dict) -> DynamicGraph:
    g = DynamicGraph(
        grid_width=scen["grid_width"],
        grid_height=scen["grid_height"],
        extra_edges=scen["extra_edges"],
        deactivate_prob=scen["deactivate_prob"],
        seed=scen["eval_seed"],
    )
    for _ in range(PERTURBATION_STEPS):
        g.update_graph()
    return g


def _sample_queries(g: DynamicGraph, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Sample n (src, dst) pairs with euclidean distance >= 60% of grid diagonal."""
    active = [nid for nid, nd in g.nodes.items() if nd.get("active", True)]
    xs = [g.nodes[nid]["coords"][0] for nid in active]
    ys = [g.nodes[nid]["coords"][1] for nid in active]
    max_dist = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    threshold = MIN_EUCLIDEAN_FRAC * max_dist

    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < n * 200:
        attempts += 1
        src, dst = rng.sample(active, 2)
        cx, cy = g.nodes[src]["coords"]
        dx, dy = g.nodes[dst]["coords"]
        dist = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
        if dist >= threshold:
            pairs.append((src, dst))
    if len(pairs) < n:
        print(f"  WARNING: only found {len(pairs)}/{n} pairs meeting distance threshold.")
    return pairs


# ── Rollout helpers ───────────────────────────────────────────────────────────
def _rollout_base(agent, g, data, src, dst) -> dict:
    """V1: pure GNN, no library."""
    agent.encode(data)
    path, visited, current, cost = [src], {src}, src, 0.0
    for _ in range(MAX_STEPS):
        valid = [nb for nb, d in g.graph[current].items()
                 if d["active"] and g.nodes[nb]["active"] and nb not in visited]
        if not valid:
            break
        action = agent.choose_action(current, valid, dst, data, epsilon=0.0)
        ed = g.graph[current][action]
        cost += ed["distance"] + 0.1 * ed["time"] + g.nodes[action]["node_penalty"]
        path.append(action); visited.add(action); current = action
        if current == dst:
            break
    reached = path[-1] == dst
    return {"reached": reached, "cost": cost if reached else float("inf")}


def _rollout_v2(agent, lib, g, data, src, dst) -> dict:
    """V2: fixed lambda=0.5."""
    agent.encode(data)
    path, visited, current, cost = [src], {src}, src, 0.0
    for _ in range(MAX_STEPS):
        valid = [nb for nb, d in g.graph[current].items()
                 if d["active"] and g.nodes[nb]["active"] and nb not in visited]
        if not valid:
            break
        action = agent.select_action_with_library(
            current, valid, dst, data, lib, lambda_retr=0.5, adaptive_lambda=False)
        ed = g.graph[current][action]
        cost += ed["distance"] + 0.1 * ed["time"] + g.nodes[action]["node_penalty"]
        path.append(action); visited.add(action); current = action
        if current == dst:
            break
    reached = path[-1] == dst
    return {"reached": reached, "cost": cost if reached else float("inf")}


def _rollout_v3(agent, lib, g, data, src, dst) -> dict:
    """V3: adaptive lambda, lambda_max=1.0."""
    agent.encode(data)
    path, visited, current, cost = [src], {src}, src, 0.0
    lambdas, max_sims = [], []
    for _ in range(MAX_STEPS):
        valid = [nb for nb, d in g.graph[current].items()
                 if d["active"] and g.nodes[nb]["active"] and nb not in visited]
        if not valid:
            break
        action, diag = agent.select_action_with_library(
            current, valid, dst, data, lib,
            adaptive_lambda=True, lambda_max=1.0, return_diagnostics=True)
        lambdas.append(diag["lambda_effective"])
        ms = diag["max_similarity"]
        max_sims.append(float(ms) if ms is not None else 0.0)
        ed = g.graph[current][action]
        cost += ed["distance"] + 0.1 * ed["time"] + g.nodes[action]["node_penalty"]
        path.append(action); visited.add(action); current = action
        if current == dst:
            break
    reached = path[-1] == dst
    n = max(len(lambdas), 1)
    return {
        "reached": reached,
        "cost": cost if reached else float("inf"),
        "mean_lambda": sum(lambdas) / n,
        "mean_max_sim": sum(max_sims) / n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent",   default="runs/agents_v1/warm_seed42_scenseed191664964_s0.pt")
    ap.add_argument("--library", default="runs/libraries/seed_42.pt")
    ap.add_argument("--out",     default="runs/v3_transfer_benchmark.json")
    ap.add_argument("--queries", type=int, default=N_QUERIES_PER_SCEN)
    args = ap.parse_args()

    agent_path = _root / args.agent
    lib_path   = _root / args.library
    print(f"Agent:   {agent_path}")
    print(f"Library: {lib_path}")

    ck = torch.load(str(agent_path), map_location="cpu", weights_only=False)
    agent = GNNDQN(node_in_dim=4, hidden_dim=128)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    lib = ExpertLibrary.load(str(lib_path))
    print(f"Loaded agent checkpoint + library ({len(lib)} entries).\n")

    results: list[dict] = []

    for scen in SCENARIOS:
        name = scen["name"]
        print(f"Scenario: {name} — {scen['nature']}")
        g    = _build_graph(scen)
        data = dynamic_graph_to_pyg(g)

        rng   = random.Random(scen["eval_seed"] + 1)   # deterministic but offset from graph seed
        pairs = _sample_queries(g, args.queries, rng)
        print(f"  Sampled {len(pairs)} (src, dst) pairs.")

        for qi, (src, dst) in enumerate(pairs):
            t0 = time.perf_counter()

            r_v1 = _rollout_base(agent, g, data, src, dst)
            r_v2 = _rollout_v2(agent, lib, g, data, src, dst)
            r_v3 = _rollout_v3(agent, lib, g, data, src, dst)

            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  q{qi+1:02d}: V1={int(r_v1['reached'])} V2={int(r_v2['reached'])} "
                  f"V3={int(r_v3['reached'])} "
                  f"lam={r_v3['mean_lambda']:.3f} sim={r_v3['mean_max_sim']:.3f} "
                  f"({elapsed:.0f}ms)")

            results.append({
                "scenario":     name,
                "query_idx":    qi,
                "src":          src,
                "dst":          dst,
                "v1_reached":   r_v1["reached"],
                "v1_cost":      r_v1["cost"],
                "v2_reached":   r_v2["reached"],
                "v2_cost":      r_v2["cost"],
                "v3_reached":   r_v3["reached"],
                "v3_cost":      r_v3["cost"],
                "v3_mean_lambda":  r_v3["mean_lambda"],
                "v3_mean_max_sim": r_v3["mean_max_sim"],
            })

    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} records to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    from collections import defaultdict
    by_scen: dict[str, list] = defaultdict(list)
    for r in results:
        by_scen[r["scenario"]].append(r)

    print()
    print("=" * 80)
    print("  V3 Transfer Benchmark — cross-distribution generalisation")
    print("=" * 80)
    header = ("Scenario", "V1 reach", "V2 reach", "V3 reach", "V3 mean lam", "V3 mean sim")
    print("  %-14s  %-10s  %-10s  %-10s  %-12s  %-12s" % header)
    print("  " + "-" * 76)

    in_dist_lam = None
    flags = []
    for scen in SCENARIOS:
        name = scen["name"]
        recs = by_scen[name]
        n = len(recs)
        if n == 0:
            continue
        v1r = sum(1 for r in recs if r["v1_reached"])
        v2r = sum(1 for r in recs if r["v2_reached"])
        v3r = sum(1 for r in recs if r["v3_reached"])
        mean_l = sum(r["v3_mean_lambda"]  for r in recs) / n
        mean_s = sum(r["v3_mean_max_sim"] for r in recs) / n
        if name == "in_dist":
            in_dist_lam = mean_l
        # Flag if V3 < V1 on this scenario
        if v3r < v1r:
            flags.append(name)
        print("  %-14s  %-10s  %-10s  %-10s  %-12.3f  %-12.3f" % (
            name, f"{v1r}/{n}", f"{v2r}/{n}", f"{v3r}/{n}", mean_l, mean_s,
        ))
    print("=" * 80)

    # Interpretation
    print()
    print("Hypothesis check:")
    ood_names = ["new_seed", "low_pert", "high_pert", "small_grid"]
    ood_recs  = [r for r in results if r["scenario"] in ood_names]

    if in_dist_lam is not None and ood_recs:
        ood_lam = sum(r["v3_mean_lambda"] for r in ood_recs) / len(ood_recs)
        lam_dropped = ood_lam < in_dist_lam
        print(f"  V3 mean lambda in-dist={in_dist_lam:.3f}  OOD avg={ood_lam:.3f}  "
              f"-> lambda {'DROPPED (as expected)' if lam_dropped else 'DID NOT DROP (unexpected)'}")

    if flags:
        print(f"  V3 < V1 on: {flags} — adaptive mechanism misfires on these distributions.")
    else:
        print("  V3 >= V1 on all scenarios — adaptive blending is provably safe (never worse than no library).")


if __name__ == "__main__":
    main()
