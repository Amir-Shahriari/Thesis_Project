"""V3 realtime evaluation: 6 cells × 4 perturbation frequencies × 3 conditions.

Conditions (all use STATIC-trained V1 agent):
  A: pure GNN (no library)
  B: fixed-lambda library (lambda_retr=0.5)
  C: adaptive-lambda library (lambda_max=1.0)

Frequencies: perturb_every_n_steps in [inf, 50, 30, 15].

Outputs:
  runs/v3_realtime_25x25.json          — per-run records
  runs/v3_realtime_25x25_aggregate.json — aggregated by (condition x frequency)
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
import time
from itertools import product

import numpy as np
import torch
from scipy import stats as scipy_stats

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.oracles.classical_dijkstra import ClassicalDijkstra
from qwarm.replay.expert_library import ExpertLibrary

AGENTS_DIR = pathlib.Path("runs/agents_v1_realtime_v2")
LIBRARIES_DIR = pathlib.Path("runs/libraries_realtime")
TEST_GRID_JSON = pathlib.Path("runs/v1_realtime_test_grid_v2.json")
OUT_JSON = pathlib.Path("runs/v3_realtime_25x25.json")
AGG_JSON = pathlib.Path("runs/v3_realtime_25x25_aggregate.json")

FREQUENCIES = [float("inf"), 50, 30, 15]
CONDITIONS = ["A", "B", "C"]
MAX_STEPS = 500
LAMBDA_RETR = 0.5
LAMBDA_MAX = 1.0


def _load_static_agent(cell_idx: int, outer_seed: int) -> GNNDQN:
    path = AGENTS_DIR / f"static_seed{outer_seed}_scen{cell_idx}.pt"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    hidden_dim = ck.get("hidden_dim", 64)
    node_in_dim = ck.get("node_in_dim", 4)
    agent = GNNDQN(node_in_dim=node_in_dim, hidden_dim=hidden_dim, seed=0)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    agent._cached_embeddings = None
    agent._cached_data_id = None
    return agent


def _load_library(cell: dict) -> ExpertLibrary:
    cell_idx = cell["cell_idx"]
    outer_seed = cell["outer_seed"]
    scenario_id = cell["scenario_id"]
    path = LIBRARIES_DIR / f"cell{cell_idx}_seed{outer_seed}_scen{scenario_id}.pt"
    return ExpertLibrary.load(str(path), device="cpu")


def _dijkstra_cost(g: DynamicGraph, src: str, dst: str) -> float:
    oracle = ClassicalDijkstra(g.nodes, g.graph)
    cost, _, _ = oracle.find_optimized_route(src, dst)
    return cost


def _build_fresh_graph(cell: dict) -> DynamicGraph:
    cfg = cell["grid"]
    return DynamicGraph(
        grid_width=cfg["grid_width"],
        grid_height=cfg["grid_height"],
        extra_edges=cfg.get("extra_edges", 2),
        deactivate_prob=cfg.get("deactivate_prob", 0.10),
        seed=cell["grid_seed"],
    )


def run_episode(
    agent: GNNDQN,
    library: ExpertLibrary | None,
    cell: dict,
    condition: str,
    freq: float,
    dijkstra_cost: float,
) -> dict:
    """Run one evaluation episode and return metrics."""
    # Fresh graph for this run
    g = _build_fresh_graph(cell)
    src = cell["source"]
    dst = cell["destination"]

    realtime = not math.isinf(freq)
    env = PathfindingEnv(
        g.graph,
        g.nodes,
        src,
        dst,
        max_steps=MAX_STEPS,
        lambda_shape=0.0,
        realtime_perturb=realtime,
        perturb_every_n_steps=int(freq) if realtime else 9999999,
        perturb_n_nodes=2,
    )

    data = dynamic_graph_to_pyg(g, device=agent.device)
    agent.encode(data)

    state = env.reset()
    total_reward = 0.0
    n_decisions = 0
    decision_time_ms = 0.0
    lambdas: list[float] = []
    sims: list[float] = []
    episode_failed = False

    while True:
        valid_actions = [a for a in env.get_valid_actions() if a not in env.visited_nodes]
        if not valid_actions:
            episode_failed = True
            break

        t0 = time.perf_counter()
        if condition == "A":
            action = agent.choose_action(state, valid_actions, dst, data, epsilon=0.0)
        elif condition == "B":
            action = agent.select_action_with_library(
                state, valid_actions, dst, data,
                library=library,
                lambda_retr=LAMBDA_RETR,
                adaptive_lambda=False,
            )
        else:  # C
            action, diag = agent.select_action_with_library(
                state, valid_actions, dst, data,
                library=library,
                adaptive_lambda=True,
                lambda_max=LAMBDA_MAX,
                return_diagnostics=True,
            )
            lambdas.append(float(diag["lambda_effective"]))
            if diag["max_similarity"] is not None:
                sims.append(float(diag["max_similarity"]))

        decision_time_ms += (time.perf_counter() - t0) * 1000.0
        n_decisions += 1

        next_state, reward, done = env.step(action)

        if reward == -5.0:
            episode_failed = True
            break

        total_reward += reward

        if env.graph_changed:
            env.graph_changed = False
            data = dynamic_graph_to_pyg(g, device=agent.device)
            agent.encode(data)

        state = next_state
        if done:
            break

    reached_goal = (state == dst) and not episode_failed

    # total_cost = sum of step costs (positive)
    # total_reward = sum(-step_cost) + 100 if goal reached
    # => total_cost = -total_reward + 100 if reached, -total_reward if not
    if reached_goal:
        total_cost = -total_reward + 100.0
    else:
        total_cost = -total_reward

    rec: dict = {
        "cell_idx": cell["cell_idx"],
        "outer_seed": cell["outer_seed"],
        "scenario_id": cell["scenario_id"],
        "condition": condition,
        "freq": freq if not math.isinf(freq) else "inf",
        "reached_goal": reached_goal,
        "total_cost": round(total_cost, 4),
        "dijkstra_cost": round(dijkstra_cost, 4),
        "cost_ratio": round(total_cost / dijkstra_cost, 4) if dijkstra_cost > 0 and reached_goal else None,
        "n_perturbations_encountered": env.perturb_count,
        "n_perturbations_survived": env.perturb_count,
        "decision_time_ms": round(decision_time_ms, 3),
        "n_decisions": n_decisions,
        "mean_lambda_effective": round(float(np.mean(lambdas)), 4) if lambdas else None,
        "mean_max_similarity": round(float(np.mean(sims)), 4) if sims else None,
    }
    return rec


def main() -> None:
    cells = json.loads(TEST_GRID_JSON.read_text())
    print(f"V3 realtime evaluation: {len(cells)} cells × {len(FREQUENCIES)} freqs × {len(CONDITIONS)} conditions")
    print(f"= {len(cells)*len(FREQUENCIES)*len(CONDITIONS)} total runs\n")

    wall_t0 = time.perf_counter()
    all_records: list[dict] = []

    for cell in cells:
        cell_idx = cell["cell_idx"]
        outer_seed = cell["outer_seed"]
        print(f"\n[Cell {cell_idx}] outer_seed={outer_seed}  scen={cell['scenario_id']}")

        # Load agent and library once per cell
        agent = _load_static_agent(cell_idx, outer_seed)
        library = _load_library(cell)

        # Dijkstra cost on initial graph (same for all runs of this cell)
        g_init = _build_fresh_graph(cell)
        dij_cost = _dijkstra_cost(g_init, cell["source"], cell["destination"])
        print(f"  dijkstra_cost={dij_cost:.2f}")

        for freq, cond in product(FREQUENCIES, CONDITIONS):
            t_run = time.perf_counter()
            rec = run_episode(agent, library, cell, cond, freq, dij_cost)
            elapsed = time.perf_counter() - t_run
            freq_label = str(int(freq)) if not math.isinf(freq) else "inf"
            goal_str = "GOAL" if rec["reached_goal"] else "FAIL"
            ratio_str = f"{rec['cost_ratio']:.3f}" if rec["cost_ratio"] is not None else "n/a"
            lam_str = f"{rec['mean_lambda_effective']:.3f}" if rec["mean_lambda_effective"] is not None else "n/a"
            print(
                f"  freq={freq_label:>3} cond={cond}  {goal_str}  "
                f"cost={rec['total_cost']:6.1f}  ratio={ratio_str}  "
                f"perturbs={rec['n_perturbations_encountered']}  "
                f"lam={lam_str}  t={elapsed:.2f}s"
            )
            all_records.append(rec)

    # Save per-run records
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_records, indent=2))
    print(f"\nSaved {len(all_records)} run records to {OUT_JSON}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    agg: dict = {}
    for freq in FREQUENCIES:
        freq_key = str(int(freq)) if not math.isinf(freq) else "inf"
        agg[freq_key] = {}
        for cond in CONDITIONS:
            recs = [r for r in all_records if str(r["freq"]) == freq_key and r["condition"] == cond]
            if not recs:
                continue
            reached = [r["reached_goal"] for r in recs]
            costs = [r["total_cost"] for r in recs]
            ratios = [r["cost_ratio"] for r in recs if r["cost_ratio"] is not None]
            dec_ms = [r["decision_time_ms"] for r in recs]
            lams = [r["mean_lambda_effective"] for r in recs if r["mean_lambda_effective"] is not None]
            sims_ = [r["mean_max_similarity"] for r in recs if r["mean_max_similarity"] is not None]
            agg[freq_key][cond] = {
                "n_cells": len(recs),
                "reach_count": int(sum(reached)),
                "mean_cost": round(float(np.mean(costs)), 4),
                "mean_cost_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
                "mean_decision_time_ms": round(float(np.mean(dec_ms)), 3),
                "mean_lambda_effective": round(float(np.mean(lams)), 4) if lams else None,
                "mean_max_similarity": round(float(np.mean(sims_)), 4) if sims_ else None,
            }

    # Paired t-tests at freq=30
    print("\n── Paired t-tests at freq=30 ──")
    recs_30 = {cond: [r for r in all_records if r["freq"] == 30 and r["condition"] == cond] for cond in CONDITIONS}
    t_tests: dict = {}

    def _paired_t(label: str, cond_x: str, cond_y: str, key: str = "total_cost") -> None:
        xs = sorted(recs_30[cond_x], key=lambda r: r["cell_idx"])
        ys = sorted(recs_30[cond_y], key=lambda r: r["cell_idx"])
        vals_x = [r[key] for r in xs]
        vals_y = [r[key] for r in ys]
        if len(vals_x) < 2:
            return
        t_stat, p_val = scipy_stats.ttest_rel(vals_x, vals_y)
        print(f"  {label}: t={t_stat:.3f}  p={p_val:.4f}")
        t_tests[label] = {"t": round(float(t_stat), 4), "p": round(float(p_val), 4)}

    _paired_t("V3_vs_V1_cost", "C", "A", "total_cost")
    _paired_t("V3_vs_V2_cost", "C", "B", "total_cost")

    # Lambda comparison: freq=30 vs freq=inf for condition C
    recs_inf_C = [r for r in all_records if r["freq"] == "inf" and r["condition"] == "C"]
    recs_30_C  = [r for r in all_records if r["freq"] == 30    and r["condition"] == "C"]
    lam_inf = [r["mean_lambda_effective"] for r in recs_inf_C if r["mean_lambda_effective"] is not None]
    lam_30  = [r["mean_lambda_effective"] for r in recs_30_C  if r["mean_lambda_effective"] is not None]
    lam_15_recs = [r for r in all_records if r["freq"] == 15 and r["condition"] == "C"]
    lam_15  = [r["mean_lambda_effective"] for r in lam_15_recs if r["mean_lambda_effective"] is not None]
    mean_lam_inf = float(np.mean(lam_inf)) if lam_inf else None
    mean_lam_30  = float(np.mean(lam_30))  if lam_30  else None
    mean_lam_15  = float(np.mean(lam_15))  if lam_15  else None
    print(f"  V3 mean lambda: inf={mean_lam_inf}  freq=30: {mean_lam_30}  freq=15: {mean_lam_15}")
    t_tests["mean_lambda_inf"] = mean_lam_inf
    t_tests["mean_lambda_30"]  = mean_lam_30
    t_tests["mean_lambda_15"]  = mean_lam_15

    agg["t_tests_at_freq30"] = t_tests

    AGG_JSON.write_text(json.dumps(agg, indent=2))
    print(f"\nSaved aggregates to {AGG_JSON}")

    wall_elapsed = time.perf_counter() - wall_t0
    print(f"\nTotal wall-clock: {wall_elapsed:.1f}s")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n── Summary table ──")
    header = f"{'freq':>5}  {'Cond':<22}  {'reach':>6}  {'mean_cost':>9}  {'cost/dij':>8}  {'dec_ms':>7}  {'mean_lam':>8}  {'mean_sim':>8}"
    print(header)
    print("-" * len(header))
    cond_labels = {"A": "(A) V1 static      ", "B": "(B) V2 fixed-lam   ", "C": "(C) V3 adaptive    "}
    for freq in FREQUENCIES:
        freq_key = str(int(freq)) if not math.isinf(freq) else "inf"
        for cond in CONDITIONS:
            d = agg.get(freq_key, {}).get(cond, {})
            reach = f"{d.get('reach_count','?')}/{d.get('n_cells','?')}"
            cost  = f"{d.get('mean_cost', float('nan')):.2f}"
            ratio = f"{d.get('mean_cost_ratio', float('nan')):.3f}" if d.get("mean_cost_ratio") is not None else " n/a "
            dec   = f"{d.get('mean_decision_time_ms', float('nan')):.1f}"
            lam   = f"{d.get('mean_lambda_effective', float('nan')):.3f}" if d.get("mean_lambda_effective") is not None else "  n/a"
            sim   = f"{d.get('mean_max_similarity', float('nan')):.3f}" if d.get("mean_max_similarity") is not None else "  n/a"
            print(f"{freq_key:>5}  {cond_labels[cond]}  {reach:>6}  {cost:>9}  {ratio:>8}  {dec:>7}  {lam:>8}  {sim:>8}")

    # ── Interpretation hints ───────────────────────────────────────────────
    print("\n── Interpretation hints ──")

    def _reach(cond: str, fkey: str) -> int | None:
        return agg.get(fkey, {}).get(cond, {}).get("reach_count")

    freq_keys = ["inf", "50", "30", "15"]

    # Hint 1: V3 reach == V1 reach across all frequencies
    v3_v1_same = all(
        _reach("C", fk) == _reach("A", fk)
        for fk in freq_keys
        if _reach("C", fk) is not None and _reach("A", fk) is not None
    )
    if v3_v1_same:
        print("  [HINT 1 FIRED] Adaptive blending preserves the GNN's intrinsic robustness to realtime "
              "perturbations — the safety property holds under realtime conditions.")

    # Hint 2: V3 mean lambda DROPS at higher perturbation frequencies
    if mean_lam_inf is not None and mean_lam_15 is not None:
        if mean_lam_15 < mean_lam_inf - 0.05:
            print("  [HINT 2 FIRED] Library similarity decreases as the graph diverges from the library's "
                  "stored states — adaptive mechanism correctly down-weights retrieval under realtime drift, "
                  "as predicted by the V3 hypothesis.")

    # Hint 3: V3 mean lambda approximately constant across frequencies
    if mean_lam_inf is not None and mean_lam_15 is not None:
        if abs(mean_lam_15 - mean_lam_inf) <= 0.05:
            print("  [HINT 3 FIRED] Library similarity is unaffected by within-trajectory perturbations — "
                  "likely because the encoder is permutation-equivariant and embeddings remain stable under "
                  "topology changes. The adaptive mechanism does not differentially activate on realtime vs "
                  "static conditions on this benchmark.")

    # Hint 4: V2 reach < V1 reach at high frequency
    v2_reach_15 = _reach("B", "15")
    v1_reach_15 = _reach("A", "15")
    if v2_reach_15 is not None and v1_reach_15 is not None and v2_reach_15 < v1_reach_15:
        print("  [HINT 4 FIRED] Fixed-lambda retrieval interferes with the GNN's natural robustness — "
              "adaptive blending (V3) is necessary to preserve safety under perturbation.")

    # Hint 5: V3 cost > V1 cost consistently
    v3_gt_v1 = all(
        agg.get(fk, {}).get("C", {}).get("mean_cost", 0) >
        agg.get(fk, {}).get("A", {}).get("mean_cost", 0)
        for fk in freq_keys
        if "C" in agg.get(fk, {}) and "A" in agg.get(fk, {})
    )
    if v3_gt_v1:
        print("  [HINT 5 FIRED] Adaptive retrieval introduces a path-quality penalty even at low residual "
              "lambda, consistent with the 100x100 long-train finding.")

    # ── Final paths summary ────────────────────────────────────────────────
    print(f"\n── Output files ──")
    libs = list(LIBRARIES_DIR.glob("*.pt"))
    print(f"  runs/libraries_realtime/  ({len(libs)} files)")
    print(f"  {OUT_JSON}")
    print(f"  {AGG_JSON}")


if __name__ == "__main__":
    main()
