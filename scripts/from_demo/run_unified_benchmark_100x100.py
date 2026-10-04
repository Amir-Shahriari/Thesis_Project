"""Unified 100×100 benchmark — V1, V2, V3, A*, Cold.

25 cells × 5 conditions × 4 perturbation regimes = 500 records.

Tasks covered:
  Task 1  raw records  → runs/unified_benchmark_100x100.json
  Task 2  aggregates   → runs/unified_benchmark_100x100_aggregate.json
  Task 3  Pareto CSV   → runs/unified_pareto_100x100.csv

Usage:
  uv run python scripts/run_unified_benchmark_100x100.py
"""
from __future__ import annotations

import csv
import gc
import heapq
import json
import math
import pathlib
import sys
import time
from typing import Any

import torch
from scipy import stats as scipy_stats

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.replay.expert_library import ExpertLibrary

# ── Constants ────────────────────────────────────────────────────────────────
GRID = dict(grid_width=100, grid_height=100, extra_edges=4, deactivate_prob=0.30)
PERTURB_STEPS_INIT = 5
MAX_STEPS = 1000

REGIMES: dict[str, int] = {
    "static": 10**9,
    "light":  50,
    "medium": 30,
    "heavy":  15,
}
CONDITIONS = ["V1", "V2", "V3", "Astar", "Cold"]

AGENTS_DIR    = pathlib.Path("runs/agents_v1")
LIBRARIES_DIR = pathlib.Path("runs/libraries")
SWEEP_PATH    = pathlib.Path("runs/sweep_v1_on_100x100.json")
OUT_RAW       = pathlib.Path("runs/unified_benchmark_100x100.json")
OUT_AGG       = pathlib.Path("runs/unified_benchmark_100x100_aggregate.json")
OUT_PARETO    = pathlib.Path("runs/unified_pareto_100x100.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def load_v1_ckpt(agent: GNNDQN, path: pathlib.Path) -> None:
    ck = torch.load(str(path), map_location=agent.device, weights_only=False)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None


def dijkstra_ref(graph: dict, nodes: dict, src: str, dst: str) -> float:
    """Simple Dijkstra using env cost formula: dist + 0.1*time + node_penalty."""
    q: list[tuple[float, str]] = [(0.0, src)]
    best: dict[str, float] = {src: 0.0}
    vis: set[str] = set()
    while q:
        c, n = heapq.heappop(q)
        if n in vis:
            continue
        vis.add(n)
        if n == dst:
            return c
        for nb, e in graph[n].items():
            if not e["active"] or not nodes[nb]["active"]:
                continue
            nc = c + e["distance"] + 0.1 * e["time"] + nodes[nb]["node_penalty"]
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(q, (nc, nb))
    return float("inf")


# ── Rollout ───────────────────────────────────────────────────────────────────

def run_rollout(
    condition: str,
    warm_agent: GNNDQN,
    cold_agent: GNNDQN,
    dyn_graph: DynamicGraph,
    library: "ExpertLibrary | None",
    source: str,
    destination: str,
    perturb_every_n_steps: int,
    dij_cost: float,
) -> dict[str, Any]:
    """Single trajectory.  Caller must restore dyn_graph to base state first."""

    is_static = (perturb_every_n_steps > MAX_STEPS)

    env = PathfindingEnv(
        graph=dyn_graph.graph,
        nodes=dyn_graph.nodes,
        source=source,
        destination=destination,
        max_steps=MAX_STEPS,
        realtime_perturb=(not is_static),
        perturb_every_n_steps=perturb_every_n_steps,
        perturb_n_nodes=2,
    )
    env.reset()

    data = dynamic_graph_to_pyg(dyn_graph)

    agent: "GNNDQN | None" = None
    if condition in ("V1", "V2", "V3"):
        agent = warm_agent
        agent.encode(data)
    elif condition == "Cold":
        agent = cold_agent
        agent.encode(data)

    # A* initial plan
    astar_path: list[str] = []
    path_idx = 1
    decision_time_ms = 0.0

    if condition == "Astar":
        t0 = time.perf_counter()
        astar_obj = ClassicalAStar(dyn_graph.nodes, dyn_graph.graph)
        _, astar_path, _ = astar_obj.find_optimized_route(source, destination)
        decision_time_ms += (time.perf_counter() - t0) * 1000
        path_idx = 1

    current = source
    visited: set[str] = {source}
    total_cost = 0.0
    n_decisions = 0
    lambdas: list[float] = []
    max_sims: list[float] = []
    done = False

    while not done:
        # Active unvisited neighbours
        valid = [
            nb for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break

        # ── Choose action ─────────────────────────────────────────────────────
        t_decide = time.perf_counter()
        action: str

        if condition in ("V1", "Cold"):
            assert agent is not None
            action = agent.choose_action(current, valid, destination, data, epsilon=0.0)

        elif condition == "V2":
            assert agent is not None
            action = agent.select_action_with_library(
                current, valid, destination, data, library,
                lambda_retr=0.5, adaptive_lambda=False,
            )

        elif condition == "V3":
            assert agent is not None
            action, diag = agent.select_action_with_library(
                current, valid, destination, data, library,
                adaptive_lambda=True, lambda_max=1.0, return_diagnostics=True,
            )
            lambdas.append(float(diag["lambda_effective"]))
            if diag["max_similarity"] is not None:
                max_sims.append(float(diag["max_similarity"]))

        elif condition == "Astar":
            next_planned = (
                astar_path[path_idx]
                if astar_path and path_idx < len(astar_path)
                else None
            )
            if next_planned and next_planned in valid:
                action = next_planned
            else:
                # Path broken or exhausted — replan
                t_rp = time.perf_counter()
                astar_rp = ClassicalAStar(dyn_graph.nodes, dyn_graph.graph)
                _, astar_path, _ = astar_rp.find_optimized_route(current, destination)
                decision_time_ms += (time.perf_counter() - t_rp) * 1000
                path_idx = 1
                next_planned = (
                    astar_path[path_idx]
                    if astar_path and path_idx < len(astar_path)
                    else None
                )
                if next_planned and next_planned in valid:
                    action = next_planned
                else:
                    break  # completely stuck
        else:
            break  # unknown condition

        decision_time_ms += (time.perf_counter() - t_decide) * 1000
        n_decisions += 1

        # ── Step env ──────────────────────────────────────────────────────────
        edge = dyn_graph.graph[current][action]
        step_cost = (
            edge["distance"]
            + 0.1 * edge["time"]
            + dyn_graph.nodes[action]["node_penalty"]
        )
        total_cost += step_cost

        next_node, _reward, done = env.step(action)
        visited.add(action)
        current = next_node

        if condition == "Astar":
            path_idx += 1

        # ── Handle perturbation ───────────────────────────────────────────────
        if env.graph_changed:
            env.graph_changed = False

            if condition != "Astar":
                # Refresh embeddings to reflect the updated graph
                data = dynamic_graph_to_pyg(dyn_graph)
                assert agent is not None
                agent.encode(data)
            else:
                # Replan A* from current position
                t_rp = time.perf_counter()
                astar_rp = ClassicalAStar(dyn_graph.nodes, dyn_graph.graph)
                _, astar_path, _ = astar_rp.find_optimized_route(current, destination)
                decision_time_ms += (time.perf_counter() - t_rp) * 1000
                path_idx = 1

    reached = (current == destination)
    n_perturb = env.perturb_count

    dij_safe = dij_cost if (dij_cost > 0 and dij_cost < 1e14) else None
    if reached and dij_safe is not None:
        ratio = total_cost / dij_safe
    else:
        ratio = float("inf")

    return {
        "reached_goal": reached,
        "total_cost": total_cost if reached else float("inf"),
        "cost_ratio_vs_dijkstra": ratio,
        "decision_time_ms": decision_time_ms,
        "n_decisions": n_decisions,
        "n_perturbations_encountered": n_perturb,
        "n_perturbations_survived": n_perturb,  # current_node is protected
        "mean_lambda": (sum(lambdas) / len(lambdas)) if lambdas else None,
        "mean_max_similarity": (sum(max_sims) / len(max_sims)) if max_sims else None,
    }


# ── Task 1: collect raw records ───────────────────────────────────────────────

def run_benchmark(cells: list[dict]) -> list[dict]:
    records: list[dict] = []
    total_cells = len(cells)

    wall_start = time.perf_counter()

    for cell_i, cell in enumerate(cells, 1):
        seed         = cell["seed"]
        scenario_id  = cell["scenario_id"]
        source       = cell["source"]
        destination  = cell["destination"]
        grid_seed    = parse_grid_seed(scenario_id)
        cell_id      = f"seed{seed}_{scenario_id}"

        print(
            f"\n[{cell_i:>2}/{total_cells}]  cell={cell_id}  "
            f"{source} -> {destination}",
            flush=True,
        )

        # ── Per-cell setup ────────────────────────────────────────────────────
        try:
            dyn_graph = DynamicGraph(
                grid_width=GRID["grid_width"],
                grid_height=GRID["grid_height"],
                extra_edges=GRID["extra_edges"],
                deactivate_prob=GRID["deactivate_prob"],
                seed=grid_seed,
            )
            for _ in range(PERTURB_STEPS_INIT):
                dyn_graph.update_graph()

            base_state = dyn_graph.save_state()
            dij_cost   = dijkstra_ref(dyn_graph.graph, dyn_graph.nodes, source, destination)

            # Load warm agent
            ckpt_warm = AGENTS_DIR / f"warm_seed{seed}_scen{scenario_id}.pt"
            warm_agent = GNNDQN(node_in_dim=4, hidden_dim=128, seed=seed)
            load_v1_ckpt(warm_agent, ckpt_warm)

            # Load cold agent
            ckpt_cold = AGENTS_DIR / f"cold_seed{seed}_scen{scenario_id}.pt"
            cold_agent = GNNDQN(node_in_dim=4, hidden_dim=128, seed=seed)
            load_v1_ckpt(cold_agent, ckpt_cold)

            # Load library
            lib_path = LIBRARIES_DIR / f"seed_{seed}.pt"
            library  = ExpertLibrary.load(str(lib_path))

        except Exception as exc:
            print(f"  SETUP ERROR: {exc}", flush=True)
            gc.collect()
            continue

        print(f"  dijkstra_ref={dij_cost:.2f}", flush=True)

        # ── Per-regime × per-condition rollout ────────────────────────────────
        for regime_name, perturb_every in REGIMES.items():
            for cond in CONDITIONS:
                try:
                    dyn_graph.restore_state(base_state)

                    t0 = time.perf_counter()
                    res = run_rollout(
                        condition=cond,
                        warm_agent=warm_agent,
                        cold_agent=cold_agent,
                        dyn_graph=dyn_graph,
                        library=library,
                        source=source,
                        destination=destination,
                        perturb_every_n_steps=perturb_every,
                        dij_cost=dij_cost,
                    )
                    elapsed = time.perf_counter() - t0

                    record = {
                        "cell_id":                    cell_id,
                        "seed":                       seed,
                        "scenario_id":                scenario_id,
                        "source":                     source,
                        "destination":                destination,
                        "condition":                  cond,
                        "regime":                     regime_name,
                        "reached_goal":               res["reached_goal"],
                        "total_cost":                 _json_float(res["total_cost"]),
                        "cost_ratio_vs_dijkstra":     _json_float(res["cost_ratio_vs_dijkstra"]),
                        "decision_time_ms":           res["decision_time_ms"],
                        "n_decisions":                res["n_decisions"],
                        "n_perturbations_encountered": res["n_perturbations_encountered"],
                        "n_perturbations_survived":   res["n_perturbations_survived"],
                        "mean_lambda":                res["mean_lambda"],
                        "mean_max_similarity":        res["mean_max_similarity"],
                        "dijkstra_ref":               _json_float(dij_cost),
                        "wall_s":                     elapsed,
                    }
                    records.append(record)

                    print(
                        f"    {regime_name:<8} {cond:<6} "
                        f"reach={str(res['reached_goal']):<5} "
                        f"cost={_fmt(res['total_cost'])} "
                        f"ratio={_fmt(res['cost_ratio_vs_dijkstra'])} "
                        f"dt={res['decision_time_ms']:.0f}ms "
                        f"npert={res['n_perturbations_encountered']}",
                        flush=True,
                    )

                except Exception as exc:
                    print(f"    ERROR {regime_name}/{cond}: {exc}", flush=True)

        # Incremental checkpoint
        _save_json(records, OUT_RAW)
        gc.collect()

    wall_total = time.perf_counter() - wall_start
    print(f"\nBenchmark complete: {len(records)} records in {wall_total:.0f}s")
    return records


def _json_float(v: float) -> "float | None":
    if v == float("inf") or v == float("-inf") or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _fmt(v: float) -> str:
    return "inf" if v >= 1e14 else f"{v:.2f}"


def _save_json(obj: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_float)


# ── Task 2: aggregates + statistics ──────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    """Build headline table, winner summary, stats, and V3 lambda trend."""

    # Group by (regime, condition)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["regime"], r["condition"])
        groups.setdefault(key, []).append(r)

    regime_order = list(REGIMES.keys())
    cond_order   = CONDITIONS

    # ── Headline table ────────────────────────────────────────────────────────
    headline: list[dict] = []
    for regime in regime_order:
        for cond in cond_order:
            recs = groups.get((regime, cond), [])
            if not recs:
                continue
            n = len(recs)
            n_reach   = sum(1 for r in recs if r["reached_goal"])
            costs     = [r["total_cost"] for r in recs if r["reached_goal"] and r["total_cost"] is not None]
            ratios    = [r["cost_ratio_vs_dijkstra"] for r in recs
                         if r["reached_goal"] and r["cost_ratio_vs_dijkstra"] is not None]
            dec_times = [r["decision_time_ms"] for r in recs]
            per_dec   = [r["decision_time_ms"] / r["n_decisions"] for r in recs
                         if r["n_decisions"] > 0]
            n_surv    = [r["n_perturbations_survived"] for r in recs]
            n_enc     = [r["n_perturbations_encountered"] for r in recs]

            row = {
                "regime":          regime,
                "condition":       cond,
                "n_cells":         n,
                "reach":           n_reach,
                "reach_rate":      n_reach / n,
                "mean_cost_reached": (sum(costs) / len(costs)) if costs else None,
                "mean_cost_ratio": (sum(ratios) / len(ratios)) if ratios else None,
                # mean of per-trajectory TOTAL decision time (length-confounded)
                "mean_trajectory_decision_ms": sum(dec_times) / max(len(dec_times), 1),
                # mean per-decision latency (decision_time_ms / n_decisions per cell)
                "mean_per_decision_ms": (sum(per_dec) / len(per_dec)) if per_dec else None,
                "mean_survival":   sum(n_surv) / max(len(n_enc), 1) if n_enc else None,
            }
            if cond == "V3":
                lam_vals = [r["mean_lambda"] for r in recs if r["mean_lambda"] is not None]
                sim_vals = [r["mean_max_similarity"] for r in recs if r["mean_max_similarity"] is not None]
                row["v3_mean_lambda"]      = (sum(lam_vals) / len(lam_vals)) if lam_vals else None
                row["v3_mean_max_sim"]     = (sum(sim_vals) / len(sim_vals)) if sim_vals else None
            headline.append(row)

    # ── Winner per regime per metric ──────────────────────────────────────────
    winners: list[dict] = []
    for regime in regime_order:
        rows = {cond: next((h for h in headline if h["regime"] == regime and h["condition"] == cond), None)
                for cond in cond_order}

        def best_reach():
            return max(cond_order, key=lambda c: rows[c]["reach_rate"] if rows[c] else -1)

        def best_cost():
            valid = [(c, rows[c]["mean_cost_reached"]) for c in cond_order
                     if rows[c] and rows[c]["mean_cost_reached"] is not None]
            return min(valid, key=lambda x: x[1])[0] if valid else None

        def best_latency():
            # Rank by per-decision latency, not trajectory total (which just
            # rewards short — often failed — trajectories).
            valid = [(c, rows[c]["mean_per_decision_ms"]) for c in cond_order
                     if rows[c] and rows[c]["mean_per_decision_ms"] is not None]
            return min(valid, key=lambda x: x[1])[0] if valid else None

        def best_survival():
            valid = [(c, rows[c]["mean_survival"]) for c in cond_order
                     if rows[c] and rows[c]["mean_survival"] is not None
                     and rows[c]["mean_survival"] > 0]
            return max(valid, key=lambda x: x[1])[0] if valid else "n/a"

        winners.append({
            "regime":          regime,
            "best_reach":      best_reach(),
            "best_cost":       best_cost(),
            "best_latency":    best_latency(),
            "best_survival":   "n/a" if regime == "static" else best_survival(),
        })

    # ── Paired statistics at MEDIUM regime ───────────────────────────────────
    def _costs_by_cond(regime: str, cond: str):
        return [r["total_cost"] for r in records
                if r["regime"] == regime and r["condition"] == cond]

    def _reach_by_cond(regime: str, cond: str):
        return [int(r["reached_goal"]) for r in records
                if r["regime"] == regime and r["condition"] == cond]

    def paired_ttest(a_vals, b_vals):
        pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None
                 and a < 1e14 and b < 1e14]
        if len(pairs) < 2:
            return None, None
        a, b = zip(*pairs)
        t, p = scipy_stats.ttest_rel(list(a), list(b))
        return float(t), float(p)

    def mcnemar(a_vals, b_vals):
        n_01 = sum(1 for a, b in zip(a_vals, b_vals) if a == 0 and b == 1)
        n_10 = sum(1 for a, b in zip(a_vals, b_vals) if a == 1 and b == 0)
        if n_01 + n_10 == 0:
            return None
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / max(n_01 + n_10, 1)
        p = float(1 - scipy_stats.chi2.cdf(chi2, df=1))
        return p

    v3_cost    = _costs_by_cond("medium", "V3")
    v1_cost    = _costs_by_cond("medium", "V1")
    v2_cost    = _costs_by_cond("medium", "V2")
    astar_cost = _costs_by_cond("medium", "Astar")
    v3_reach   = _reach_by_cond("medium", "V3")
    v1_reach   = _reach_by_cond("medium", "V1")

    t_v3_v1, p_v3_v1 = paired_ttest(v3_cost, v1_cost)
    t_v3_v2, p_v3_v2 = paired_ttest(v3_cost, v2_cost)
    t_v3_as, p_v3_as = paired_ttest(v3_cost, astar_cost)
    p_mcnemar        = mcnemar(v3_reach, v1_reach)

    stats = {
        "regime": "medium",
        "V3_vs_V1_cost_t":      t_v3_v1, "V3_vs_V1_cost_p":      p_v3_v1,
        "V3_vs_V2_cost_t":      t_v3_v2, "V3_vs_V2_cost_p":      p_v3_v2,
        "V3_vs_Astar_cost_t":   t_v3_as, "V3_vs_Astar_cost_p":   p_v3_as,
        "V3_vs_V1_reach_mcnemar_p": p_mcnemar,
    }

    # ── V3 lambda trend ───────────────────────────────────────────────────────
    lambda_trend: dict[str, "float | None"] = {}
    for regime in regime_order:
        v3_recs = [r for r in records if r["regime"] == regime and r["condition"] == "V3"
                   and r["mean_lambda"] is not None]
        lambda_trend[regime] = (
            sum(r["mean_lambda"] for r in v3_recs) / len(v3_recs)
            if v3_recs else None
        )

    lambda_vals = [lambda_trend[r] for r in regime_order if lambda_trend[r] is not None]
    monotone_decreasing = all(
        lambda_vals[i] >= lambda_vals[i + 1]
        for i in range(len(lambda_vals) - 1)
    ) if len(lambda_vals) >= 2 else False

    if monotone_decreasing:
        lambda_interp = "V3 self-attenuates under heavier perturbation regimes as predicted."
    else:
        lambda_interp = (
            "V3 lambda trend is not strictly monotone-decreasing across regimes; "
            "adaptive blending does not fully self-attenuate as predicted."
        )

    return {
        "headline_table": headline,
        "winner_summary": winners,
        "paired_stats":   stats,
        "v3_lambda_trend": lambda_trend,
        "v3_lambda_monotone_decreasing": monotone_decreasing,
        "v3_lambda_interpretation": lambda_interp,
    }


# ── Task 3: Pareto CSV ────────────────────────────────────────────────────────

def write_pareto_csv(headline: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "regime", "mean_cost_ratio",
                  "mean_trajectory_decision_ms", "mean_per_decision_ms", "reach_rate"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in headline:
            writer.writerow({
                "condition":            row["condition"],
                "regime":               row["regime"],
                "mean_cost_ratio":      row.get("mean_cost_ratio") if row.get("mean_cost_ratio") is not None else "",
                "mean_trajectory_decision_ms": row["mean_trajectory_decision_ms"],
                "mean_per_decision_ms": row["mean_per_decision_ms"] if row["mean_per_decision_ms"] is not None else "",
                "reach_rate":           row["reach_rate"],
            })
    print(f"Pareto CSV saved to {path}")


# ── Final report ──────────────────────────────────────────────────────────────

def print_report(agg: dict, sanity_path: pathlib.Path, wall_s: float) -> None:
    sanity_status = "NOT RUN"
    if sanity_path.exists():
        with open(sanity_path) as f:
            s = json.load(f)
        sanity_status = s.get("status", "unknown")

    print("\n" + "=" * 72)
    print("FINAL REPORT")
    print("=" * 72)

    print(f"\n1. SANITY CHECK\n   {sanity_status}")

    print("\n2. HEADLINE TABLE (25 cells each)")
    print(f"  {'Regime':<8} {'Cond':<6} {'Reach':>5} {'Cost(reach)':>12} {'Cost/Dij':>9} "
          f"{'TrajDec(ms)':>11} {'PerDec(ms)':>10} {'Survival':>9}")
    print("  " + "-" * 78)
    for row in agg["headline_table"]:
        print(
            f"  {row['regime']:<8} {row['condition']:<6} "
            f"{row['reach']:>2}/{row['n_cells']:<2}  "
            f"{_fmt_opt(row['mean_cost_reached']):>12} "
            f"{_fmt_opt(row['mean_cost_ratio']):>9} "
            f"{row['mean_trajectory_decision_ms']:>11.1f} "
            f"{_fmt_opt(row['mean_per_decision_ms']):>10} "
            f"{_fmt_opt(row['mean_survival']):>9}"
        )

    print("\n3. WINNER SUMMARY")
    print(f"  {'Regime':<8} {'BestReach':<10} {'BestCost':<10} {'BestLat':<10} {'BestSurv':<10}")
    print("  " + "-" * 50)
    for w in agg["winner_summary"]:
        print(
            f"  {w['regime']:<8} {str(w['best_reach']):<10} "
            f"{str(w['best_cost']):<10} {str(w['best_latency']):<10} "
            f"{str(w['best_survival']):<10}"
        )

    print("\n4. PAIRED STATS (medium regime, 25 cells)")
    st = agg["paired_stats"]
    _p = lambda v: f"{v:.4f}" if v is not None else "n/a"
    _t = lambda v: f"{v:.3f}" if v is not None else "n/a"
    print(f"  V3 cost vs V1 cost   : t={_t(st['V3_vs_V1_cost_t'])}  p={_p(st['V3_vs_V1_cost_p'])}")
    print(f"  V3 cost vs V2 cost   : t={_t(st['V3_vs_V2_cost_t'])}  p={_p(st['V3_vs_V2_cost_p'])}")
    print(f"  V3 cost vs Astar cost: t={_t(st['V3_vs_Astar_cost_t'])}  p={_p(st['V3_vs_Astar_cost_p'])}")
    print(f"  V3 vs V1 reach McNemar p={_p(st['V3_vs_V1_reach_mcnemar_p'])}")

    print("\n5. V3 LAMBDA TREND")
    for regime, lam in agg["v3_lambda_trend"].items():
        print(f"  {regime:<8}: {lam:.4f}" if lam is not None else f"  {regime:<8}: n/a")
    print(f"\n  {agg['v3_lambda_interpretation']}")

    print("\n6. OUTPUT FILES")
    for p in [OUT_RAW, OUT_AGG, OUT_PARETO, sanity_path]:
        exists = "OK" if p.exists() else "MISSING"
        print(f"  {exists}  {p}")

    print(f"\n7. WALL-CLOCK SUMMARY")
    m, s = divmod(int(wall_s), 60)
    h, m = divmod(m, 60)
    print(f"  Total benchmark time: {h:d}h {m:02d}m {s:02d}s")
    print("\n" + "=" * 72)


def _fmt_opt(v: "float | None") -> str:
    if v is None:
        return "n/a"
    return "inf" if v >= 1e14 else f"{v:.2f}"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("QWarm-RL Unified Benchmark — 100×100 grid")
    print(f"  {len(CONDITIONS)} conditions × {len(REGIMES)} regimes")

    with open(SWEEP_PATH) as f:
        cells = json.load(f)
    print(f"  {len(cells)} cells from {SWEEP_PATH}")
    print(f"  Output: {OUT_RAW}\n")

    wall_start = time.perf_counter()

    # ── Task 1: raw benchmark ─────────────────────────────────────────────────
    records = run_benchmark(cells)
    _save_json(records, OUT_RAW)
    print(f"\nTask 1 complete: {len(records)} records saved to {OUT_RAW}")

    # ── Task 2: aggregate ─────────────────────────────────────────────────────
    print("\nTask 2: computing aggregates …")
    agg = aggregate(records)
    _save_json(agg, OUT_AGG)
    print(f"Task 2 complete: aggregates saved to {OUT_AGG}")

    # ── Task 3: Pareto CSV ────────────────────────────────────────────────────
    print("\nTask 3: writing Pareto CSV …")
    write_pareto_csv(agg["headline_table"], OUT_PARETO)
    print(f"Task 3 complete: {OUT_PARETO}")

    wall_s = time.perf_counter() - wall_start

    # ── Final report ──────────────────────────────────────────────────────────
    sanity_path = pathlib.Path("runs/unified_benchmark_sanity.json")
    print_report(agg, sanity_path, wall_s)


if __name__ == "__main__":
    main()
