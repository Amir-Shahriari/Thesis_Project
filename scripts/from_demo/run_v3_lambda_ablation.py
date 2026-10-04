"""V3 lambda_max ablation script.

For each cell in the 10-cell subset (first 2 scenarios per seed) and each
lambda_max in {0.5, 1.5, 2.0}, run the V3 adaptive-lambda rollout using the
V1 warm checkpoint + V2 library.  Also reads lambda_max=1.0 results from the
existing sweep_v3_adaptive.json for a complete four-point curve.

Usage:
  uv run python scripts/run_v3_lambda_ablation.py \
      --cells 10 --lambda-max 0.5 1.5 2.0 \
      --out runs/v3_lambda_ablation.json
"""
from __future__ import annotations

import sys
import argparse
import json
import pathlib
import time
from collections import defaultdict
from typing import List

_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))

import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.replay.expert_library import ExpertLibrary

# ── Constants (must match V3 eval) ────────────────────────────────────────────
GRID_TEMPLATE = {
    "grid_width":     100,
    "grid_height":    100,
    "extra_edges":    4,
    "deactivate_prob": 0.30,
}
PERTURBATION_STEPS = 5
MAX_STEPS          = 1000
AGENTS_DIR         = _root / "runs" / "agents_v1"
LIBRARIES_DIR      = _root / "runs" / "libraries"
V3_SWEEP           = _root / "runs" / "sweep_v3_adaptive.json"
V1_SWEEP           = _root / "runs" / "sweep_v1_on_100x100.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _build_graph(grid_seed: int) -> DynamicGraph:
    g = DynamicGraph(**GRID_TEMPLATE, seed=grid_seed)
    for _ in range(PERTURBATION_STEPS):
        g.update_graph()
    return g


def _load_agent_and_lib(seed: int, scenario_id: str):
    ck_path  = AGENTS_DIR   / f"warm_seed{seed}_scen{scenario_id}.pt"
    lib_path = LIBRARIES_DIR / f"seed_{seed}.pt"
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    agent = GNNDQN(node_in_dim=4, hidden_dim=128)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    lib = ExpertLibrary.load(str(lib_path))
    return agent, lib


def run_rollout(
    agent: GNNDQN,
    g: DynamicGraph,
    data,
    source: str,
    destination: str,
    library: ExpertLibrary,
    lambda_max: float,
) -> dict:
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0
    lambdas: list[float] = []
    max_sims: list[float] = []

    for _ in range(MAX_STEPS):
        valid = [
            nb for nb, d in g.graph[current].items()
            if d["active"] and g.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break

        action, diag = agent.select_action_with_library(
            current, valid, destination, data, library,
            adaptive_lambda=True,
            lambda_max=lambda_max,
            return_diagnostics=True,
        )
        lambdas.append(diag["lambda_effective"])
        ms = diag["max_similarity"]
        max_sims.append(float(ms) if ms is not None else 0.0)

        ed = g.graph[current][action]
        cost += ed["distance"] + 0.1 * ed["time"] + g.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    n = max(len(lambdas), 1)
    return {
        "reached_strict": reached,
        "cost": cost if reached else float("inf"),
        "mean_lambda": sum(lambdas) / n,
        "mean_max_sim": sum(max_sims) / n,
    }


def select_subset(all_cells: list[dict], n_cells: int) -> list[dict]:
    """First n_cells/n_seeds scenarios per seed, balanced."""
    by_seed = defaultdict(list)
    for c in all_cells:
        by_seed[c["seed"]].append(c)
    seeds = sorted(by_seed)
    per_seed = max(1, n_cells // len(seeds))
    subset = []
    for seed in seeds:
        subset.extend(by_seed[seed][:per_seed])
        if len(subset) >= n_cells:
            break
    return subset[:n_cells]


def load_v3_baseline(subset: list[dict]) -> dict[tuple, dict]:
    """Load lambda_max=1.0 results from existing sweep_v3_adaptive.json."""
    v3 = json.loads(V3_SWEEP.read_text())
    index = {(r["seed"], r["scenario_id"]): r for r in v3}
    out = {}
    for cell in subset:
        key = (cell["seed"], cell["scenario_id"])
        if key in index:
            r = index[key]
            out[key] = {
                "reached_strict": r["v3_strict"],
                "cost": r["v3_cost"] if r["v3_strict"] else float("inf"),
                "mean_lambda": r["v3_mean_lambda"],
                "mean_max_sim": r["v3_mean_max_sim"],
            }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells",      type=int,   default=10)
    ap.add_argument("--lambda-max", type=float, nargs="+", default=[0.5, 1.5, 2.0])
    ap.add_argument("--out",        type=str,   default="runs/v3_lambda_ablation.json")
    args = ap.parse_args()

    all_cells = json.loads(V1_SWEEP.read_text())
    subset    = select_subset(all_cells, args.cells)
    print(f"Selected {len(subset)} cells across {len({c['seed'] for c in subset})} seeds.")

    lambda_values = sorted(args.lambda_max)
    v3_baseline   = load_v3_baseline(subset)
    print(f"Loaded {len(v3_baseline)} lambda_max=1.0 baseline entries from sweep_v3_adaptive.json.")

    results: list[dict] = []
    total = len(subset) * len(lambda_values)
    done  = 0

    for cell in subset:
        seed        = cell["seed"]
        scenario_id = cell["scenario_id"]
        source      = cell["source"]
        destination = cell["destination"]
        grid_seed   = _parse_grid_seed(scenario_id)

        print(f"\n[{done+1}/{total}..] seed={seed} scen={scenario_id} "
              f"src={source} dst={destination}")

        g    = _build_graph(grid_seed)
        data = dynamic_graph_to_pyg(g)
        agent, lib = _load_agent_and_lib(seed, scenario_id)

        for lm in lambda_values:
            t0 = time.perf_counter()
            r  = run_rollout(agent, g, data, source, destination, lib, lambda_max=lm)
            elapsed = (time.perf_counter() - t0) * 1000
            done += 1
            print(f"  lambda_max={lm:.1f}: reached={r['reached_strict']} "
                  f"cost={r['cost']:.1f} mean_lam={r['mean_lambda']:.3f} "
                  f"mean_sim={r['mean_max_sim']:.3f} ({elapsed:.0f}ms)")
            results.append({
                "seed": seed,
                "scenario_id": scenario_id,
                "source": source,
                "destination": destination,
                "lambda_max": lm,
                **r,
            })

    # Save new results
    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} records to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    # Aggregate by lambda_max (ablation values + 1.0 from baseline)
    def _agg(records):
        n = len(records)
        n_reach = sum(1 for r in records if r["reached_strict"])
        costs   = [r["cost"] for r in records if r["cost"] < 1e15]
        mean_c  = sum(costs) / len(costs) if costs else float("inf")
        mean_l  = sum(r["mean_lambda"]  for r in records) / max(n, 1)
        mean_s  = sum(r["mean_max_sim"] for r in records) / max(n, 1)
        return n_reach, n, mean_c, mean_l, mean_s

    # Build aggregation dict: lambda_max -> list of result dicts
    by_lm: dict[float, list] = defaultdict(list)
    for r in results:
        by_lm[r["lambda_max"]].append(r)

    # Add 1.0 baseline entries for same subset cells
    baseline_records = []
    for cell in subset:
        key = (cell["seed"], cell["scenario_id"])
        if key in v3_baseline:
            baseline_records.append(v3_baseline[key])
    by_lm[1.0] = baseline_records

    all_lm = sorted(by_lm.keys())

    print()
    print("=" * 70)
    print("  lambda_max sensitivity summary (10-cell subset)")
    print("=" * 70)
    header = ("lambda_max", "goal-reach", "mean cost (reached)", "mean lam_eff", "mean max-sim")
    print("  %-10s  %-12s  %-20s  %-12s  %-12s" % header)
    print("  " + "-" * 68)
    for lm in all_lm:
        recs = by_lm[lm]
        if not recs:
            continue
        n_reach, n, mean_c, mean_l, mean_s = _agg(recs)
        suffix = " <- (from sweep_v3_adaptive.json)" if lm == 1.0 else ""
        cost_str = f"{mean_c:.1f}" if mean_c < 1e15 else "inf"
        print("  %-10.1f  %-12s  %-20s  %-12.3f  %-12.3f%s" % (
            lm, f"{n_reach}/{n}", cost_str, mean_l, mean_s, suffix,
        ))
    print("=" * 70)

    # Interpretation
    reaches = [_agg(by_lm[lm])[0] for lm in all_lm if by_lm[lm]]
    if len(set(reaches)) == 1:
        note = ("FLAT across lambda_max: adaptive mechanism is self-regulating "
                "(max-sim already caps effective lambda).")
    elif reaches[-1] < reaches[0]:
        note = ("Goal-reach DEGRADES at high lambda_max: library is over-influential "
                "when given headroom above its natural similarity ceiling.")
    else:
        best_lm = all_lm[reaches.index(max(reaches))]
        note = f"Goal-reach peaks at lambda_max={best_lm:.1f}: that is the optimal trust ceiling."

    print()
    print("Interpretation:", note)


if __name__ == "__main__":
    main()
