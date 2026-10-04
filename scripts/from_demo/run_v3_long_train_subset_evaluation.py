"""V1 / V2 / V3 inference comparison across 10 long-trained cells (n=15 experiment).

For each cell where warm_strict=True (4x reached goal):
  1. Build DynamicGraph (100x100, grid_seed from scenario_id, 5 perturbation steps)
  2. Load the long-trained V1 agent from runs/agents_v1_long/
  3. Load the freshly-built library from runs/libraries_long/
  4. Run three greedy rollouts on the canonical (source, destination) — same agent, only
     the inference rule differs (apples-to-apples isolation):
       V1: pure network Q-values (no library)
       V2: fixed-lambda blending, lambda_retr=0.5
       V3: adaptive-lambda blending, lambda_max=1.0
  5. Record per-mode metrics and save incrementally.

Cross-check: compares seed=42/seed191664964_s0 against runs/v3_long_train_results.json
(the n=1 prior result). Flags if costs differ by >10%.

Output: runs/v3_long_train_subset_results.json
        (30 records — 10 cells x 3 modes — plus 5 skipped noted with status=skipped_4x_failed)

CLI:
    uv run python scripts/run_v3_long_train_subset_evaluation.py
"""
from __future__ import annotations

import datetime
import heapq
import json
import pathlib
import time

import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.replay.expert_library import ExpertLibrary

SUBSET_JSON    = pathlib.Path("runs/v1_long_train_subset_results.json")
AGENTS_DIR     = pathlib.Path("runs/agents_v1_long")
LIBRARIES_DIR  = pathlib.Path("runs/libraries_long")
RESULTS_PATH   = pathlib.Path("runs/v3_long_train_subset_results.json")
PRIOR_N1_PATH  = pathlib.Path("runs/v3_long_train_results.json")

PERTURBATION_STEPS = 5
MAX_STEPS      = 1000
K_THRESHOLD    = 3.0
LAMBDA_MAX     = 1.0
LAMBDA_RETR_V2 = 0.5
HIDDEN_DIM     = 128


def _parse_grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _dijkstra_cost(g: DynamicGraph, source: str, destination: str) -> float:
    queue: list[tuple[float, str]] = [(0.0, source)]
    best: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            return cost
        for nb, edge in g.graph[node].items():
            if not edge["active"] or not g.nodes[nb]["active"]:
                continue
            step_cost = edge["distance"] + 0.1 * edge["time"] + g.nodes[nb]["node_penalty"]
            nc = cost + step_cost
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(queue, (nc, nb))
    return float("inf")


def _load_agent(seed: int, scenario_id: str) -> GNNDQN:
    path = AGENTS_DIR / f"warm_seed{seed}_scen{scenario_id}_4x.pt"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    hidden_dim  = ck.get("hidden_dim", HIDDEN_DIM)
    node_in_dim = ck.get("node_in_dim", 4)
    agent = GNNDQN(node_in_dim=node_in_dim, hidden_dim=hidden_dim, seed=seed)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None
    return agent


def _build_graph(grid_seed: int) -> DynamicGraph:
    g = DynamicGraph(grid_width=100, grid_height=100, extra_edges=4,
                     deactivate_prob=0.30, seed=grid_seed)
    for _ in range(PERTURBATION_STEPS):
        g.update_graph()
    return g


def _rollout_v1(agent: GNNDQN, g: DynamicGraph, data, source: str, destination: str) -> dict:
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0
    t0 = time.perf_counter()

    for _ in range(MAX_STEPS):
        valid = [
            nb for nb, d in g.graph[current].items()
            if d["active"] and g.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break
        action = agent.choose_action(current, valid, destination, data, epsilon=0.0)
        edge = g.graph[current][action]
        cost += edge["distance"] + 0.1 * edge["time"] + g.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    return {
        "reached_goal": reached,
        "total_cost": cost if reached else None,
        "n_decisions": len(path) - 1,
        "decision_time_ms": (time.perf_counter() - t0) * 1000,
        "n_perturbations_encountered": 0,
    }


def _rollout_v2(agent: GNNDQN, g: DynamicGraph, data, source: str, destination: str,
                library: ExpertLibrary) -> dict:
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0
    t0 = time.perf_counter()

    for _ in range(MAX_STEPS):
        valid = [
            nb for nb, d in g.graph[current].items()
            if d["active"] and g.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break
        action = agent.select_action_with_library(
            current, valid, destination, data, library,
            adaptive_lambda=False,
            lambda_retr=LAMBDA_RETR_V2,
        )
        edge = g.graph[current][action]
        cost += edge["distance"] + 0.1 * edge["time"] + g.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    return {
        "reached_goal": reached,
        "total_cost": cost if reached else None,
        "n_decisions": len(path) - 1,
        "decision_time_ms": (time.perf_counter() - t0) * 1000,
        "n_perturbations_encountered": 0,
    }


def _rollout_v3(agent: GNNDQN, g: DynamicGraph, data, source: str, destination: str,
                library: ExpertLibrary) -> dict:
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0
    lambdas: list[float] = []
    max_sims: list[float] = []
    t0 = time.perf_counter()

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
            lambda_max=LAMBDA_MAX,
            return_diagnostics=True,
        )
        lambdas.append(diag["lambda_effective"])
        if diag["max_similarity"] is not None:
            max_sims.append(diag["max_similarity"])
        edge = g.graph[current][action]
        cost += edge["distance"] + 0.1 * edge["time"] + g.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    return {
        "reached_goal": reached,
        "total_cost": cost if reached else None,
        "n_decisions": len(path) - 1,
        "decision_time_ms": (time.perf_counter() - t0) * 1000,
        "n_perturbations_encountered": 0,
        "mean_lambda_effective": sum(lambdas) / max(len(lambdas), 1),
        "mean_max_similarity": sum(max_sims) / max(len(max_sims), 1),
    }


def _check_prior_n1(records: list[dict]) -> None:
    """Compare seed=42/seed191664964_s0 against the n=1 prior result."""
    if not PRIOR_N1_PATH.exists():
        print("  [cross-check] runs/v3_long_train_results.json not found — skipping.")
        return

    with open(PRIOR_N1_PATH) as fh:
        prior_list = json.load(fh)
    if not prior_list:
        return

    # The n=1 script appended records; take the most recent
    prior = prior_list[-1]
    prior_v3_cost = prior.get("v3_cost")

    # Find the matching cell in our results
    for r in records:
        if r.get("seed") == 42 and r.get("scenario_id") == "seed191664964_s0" and r.get("mode") == "V3":
            new_v3_cost = r.get("total_cost")
            if prior_v3_cost is not None and new_v3_cost is not None:
                ratio = abs(new_v3_cost - prior_v3_cost) / max(abs(prior_v3_cost), 1e-9)
                flag = "FLAG — >10% divergence" if ratio > 0.10 else "OK"
                print(f"\n  [cross-check seed=42/s0] prior V3 cost={prior_v3_cost:.1f}  "
                      f"new V3 cost={new_v3_cost:.1f}  diff={ratio:.1%}  {flag}")
            else:
                print(f"\n  [cross-check seed=42/s0] prior reached={prior.get('v3_strict')}  "
                      f"new reached={r.get('reached_goal')}  (one or both did not reach goal)")
            return


def _safe_json(obj):
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    return obj


def main() -> None:
    cells = json.loads(SUBSET_JSON.read_text())
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"V3 Long-Train Subset Evaluation")
    print(f"Cells total: {len(cells)}  |  expected skip=5, evaluate=10")
    print(f"Output: {RESULTS_PATH}")
    print()

    all_records: list[dict] = []
    wall_t0 = time.perf_counter()
    n_skipped = 0
    n_evaluated = 0

    for idx, cell in enumerate(cells):
        seed        = cell["seed"]
        scenario_id = cell["scenario_id"]
        source      = cell["source"]
        destination = cell["destination"]
        reached_4x  = cell.get("warm_strict", False)

        if not reached_4x:
            print(f"[{idx:02d}] seed={seed} scen={scenario_id} — [skipped: 4x failed]")
            all_records.append({
                "cell_idx": idx,
                "seed": seed,
                "scenario_id": scenario_id,
                "status": "skipped_4x_failed",
            })
            n_skipped += 1
            continue

        lib_path = LIBRARIES_DIR / f"cell{idx:02d}_seed{seed}_scen{scenario_id}.pt"
        if not lib_path.exists():
            print(f"[{idx:02d}] seed={seed} scen={scenario_id} — MISSING library {lib_path}, skipping")
            all_records.append({
                "cell_idx": idx, "seed": seed, "scenario_id": scenario_id,
                "status": "skipped_library_missing",
            })
            continue

        print(f"\n[{idx:02d}] seed={seed} scen={scenario_id}  {source} -> {destination}", flush=True)
        t_cell = time.perf_counter()

        grid_seed = _parse_grid_seed(scenario_id)
        g    = _build_graph(grid_seed)
        data = dynamic_graph_to_pyg(g)

        dijkstra = _dijkstra_cost(g, source, destination)
        print(f"  Dijkstra: {dijkstra:.2f}", flush=True)

        agent   = _load_agent(seed, scenario_id)
        library = ExpertLibrary.load(str(lib_path))
        print(f"  Library size: {len(library)}", flush=True)

        # V1
        print("  V1 ...", flush=True)
        v1 = _rollout_v1(agent, g, data, source, destination)
        v1_ratio = (v1["total_cost"] / max(dijkstra, 1e-9)) if v1["total_cost"] is not None else None
        all_records.append({
            "cell_idx": idx, "seed": seed, "scenario_id": scenario_id,
            "source": source, "destination": destination,
            "dijkstra_cost": dijkstra,
            "mode": "V1",
            "status": "ok",
            **v1,
            "cost_ratio_vs_dijkstra": v1_ratio,
        })
        print(f"     reached={v1['reached_goal']}  cost={v1['total_cost']}  ratio={v1_ratio}", flush=True)

        # V2
        print("  V2 ...", flush=True)
        v2 = _rollout_v2(agent, g, data, source, destination, library)
        v2_ratio = (v2["total_cost"] / max(dijkstra, 1e-9)) if v2["total_cost"] is not None else None
        all_records.append({
            "cell_idx": idx, "seed": seed, "scenario_id": scenario_id,
            "source": source, "destination": destination,
            "dijkstra_cost": dijkstra,
            "mode": "V2",
            "lambda_retr": LAMBDA_RETR_V2,
            "status": "ok",
            **v2,
            "cost_ratio_vs_dijkstra": v2_ratio,
        })
        print(f"     reached={v2['reached_goal']}  cost={v2['total_cost']}  ratio={v2_ratio}", flush=True)

        # V3
        print("  V3 ...", flush=True)
        v3 = _rollout_v3(agent, g, data, source, destination, library)
        v3_ratio = (v3["total_cost"] / max(dijkstra, 1e-9)) if v3["total_cost"] is not None else None
        all_records.append({
            "cell_idx": idx, "seed": seed, "scenario_id": scenario_id,
            "source": source, "destination": destination,
            "dijkstra_cost": dijkstra,
            "mode": "V3",
            "lambda_max": LAMBDA_MAX,
            "status": "ok",
            **v3,
            "cost_ratio_vs_dijkstra": v3_ratio,
        })
        print(f"     reached={v3['reached_goal']}  cost={v3['total_cost']}  ratio={v3_ratio}"
              f"  mean_lam={v3['mean_lambda_effective']:.3f}  mean_sim={v3['mean_max_similarity']:.3f}", flush=True)

        elapsed = time.perf_counter() - t_cell
        print(f"  cell time: {elapsed:.1f}s", flush=True)
        n_evaluated += 1

        # Incremental save
        with open(RESULTS_PATH, "w") as fh:
            json.dump(all_records, fh, indent=2, default=_safe_json)

    # Final save
    with open(RESULTS_PATH, "w") as fh:
        json.dump(all_records, fh, indent=2, default=_safe_json)

    wall_elapsed = time.perf_counter() - wall_t0
    print(f"\n{'='*60}")
    print(f"Done.  Wall-clock: {wall_elapsed:.1f}s")
    print(f"Skipped (4x failed): {n_skipped}")
    print(f"Evaluated: {n_evaluated} cells x 3 modes = {n_evaluated*3} records")
    print(f"Results: {RESULTS_PATH}")

    # Cross-check against n=1 prior result
    _check_prior_n1(all_records)


if __name__ == "__main__":
    main()
