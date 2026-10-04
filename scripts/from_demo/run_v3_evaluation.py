"""V3 evaluation — adaptive retrieval blending with pre-trained V1 agents.

No training is performed.  For each of the 25 (seed, scenario) cells:

  1. Read cell metadata from runs/sweep_v1_on_100x100.json.
  2. Rebuild the same DynamicGraph (grid_seed embedded in scenario_id) and
     apply 5 perturbation steps to approximate the trained-final graph state.
  3. Load the V1 warm-agent checkpoint from runs/agents_v1/.
  4. Load the V2 expert library for that seed from runs/libraries/.
  5. V3 rollout: adaptive lambda   lambda(s) = lambda_max * max(0, cos_sim(h_s, lib))
  6. V2 control rollout: fixed lambda=0.5, validates checkpoint loading is sound.

CLI:
    uv run python scripts/run_v3_evaluation.py --out runs/sweep_v3_adaptive.json
    uv run python scripts/run_v3_evaluation.py --single-cell --out runs/sweep_v3_smoke.json
"""
from __future__ import annotations

import argparse
import gc
import heapq
import json
import pathlib
import time
from typing import Optional

import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.path_evaluator import evaluate_agent
from qwarm.replay.expert_library import ExpertLibrary

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

GRID_TEMPLATE = {
    "grid_width": 100,
    "grid_height": 100,
    "extra_edges": 4,
    "deactivate_prob": 0.30,
}
PERTURBATION_STEPS = 5      # update_graph() calls after initial build
K_THRESHOLD = 3.0
LAMBDA_MAX = 1.0
LAMBDA_RETR_V2 = 0.5       # V2 fixed-lambda baseline for reproduction check
MAX_STEPS_PER_EVAL = 1000   # generous budget for 100x100 grid

V1_SWEEP_PATH = pathlib.Path("runs/sweep_v1_on_100x100.json")
V2_SWEEP_PATH = pathlib.Path("runs/sweep_v2_headline.json")
AGENTS_DIR    = pathlib.Path("runs/agents_v1")
LIBRARIES_DIR = pathlib.Path("runs/libraries")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dijkstra_cost(dyn_graph: DynamicGraph, source: str, destination: str) -> float:
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
        for nb, edge in dyn_graph.graph[node].items():
            if not edge["active"] or not dyn_graph.nodes[nb]["active"]:
                continue
            step_cost = edge["distance"] + 0.1 * edge["time"] + dyn_graph.nodes[nb]["node_penalty"]
            nc = cost + step_cost
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(queue, (nc, nb))
    return float("inf")


def _load_v1_checkpoint(agent: GNNDQN, path: pathlib.Path) -> None:
    """Load V1 checkpoint using its own key names (differ from GNNDQN.load())."""
    ck = torch.load(str(path), map_location=agent.device, weights_only=False)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None


def _build_graph(grid_seed: int) -> DynamicGraph:
    """Build and perturb a graph to approximate the trained-final state."""
    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=grid_seed,
    )
    for _ in range(PERTURBATION_STEPS):
        g.update_graph()
    return g


def _parse_grid_seed(scenario_id: str) -> int:
    """Extract grid_seed from scenario_id string like 'seed191664964_s0'."""
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _checkpoint_path(seed: int, scenario_id: str) -> pathlib.Path:
    return AGENTS_DIR / f"warm_seed{seed}_scen{scenario_id}.pt"


def _library_path(seed: int) -> pathlib.Path:
    return LIBRARIES_DIR / f"seed_{seed}.pt"


# ──────────────────────────────────────────────────────────────────────────────
# V3 rollout with adaptive lambda and per-step diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def run_v3_rollout(
    agent: GNNDQN,
    dyn_graph: DynamicGraph,
    data,
    source: str,
    destination: str,
    library: ExpertLibrary,
    lambda_max: float = LAMBDA_MAX,
) -> dict:
    """Greedy rollout with adaptive lambda.  Returns cost, reach, and telemetry."""
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0
    lambdas: list[float] = []
    max_sims: list[float] = []

    for _ in range(MAX_STEPS_PER_EVAL):
        valid = [
            nb for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
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
        if diag["max_similarity"] is not None:
            max_sims.append(diag["max_similarity"])

        edge = dyn_graph.graph[current][action]
        cost += edge["distance"] + 0.1 * edge["time"] + dyn_graph.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    return {
        "reached_goal": reached,
        "cost": cost if reached else float("inf"),
        "path_len": len(path),
        "mean_lambda": sum(lambdas) / max(len(lambdas), 1),
        "mean_max_sim": sum(max_sims) / max(len(max_sims), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_cell(cell: dict) -> dict:
    seed         = cell["seed"]
    scenario_id  = cell["scenario_id"]
    source       = cell["source"]
    destination  = cell["destination"]
    euclidean    = cell["euclidean_distance"]

    grid_seed    = _parse_grid_seed(scenario_id)
    ck_path      = _checkpoint_path(seed, scenario_id)
    lib_path     = _library_path(seed)

    if not ck_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found: {lib_path}")

    # Build graph
    g = _build_graph(grid_seed)
    data = dynamic_graph_to_pyg(g)

    # Load V1 warm agent
    agent = GNNDQN(node_in_dim=4, hidden_dim=128, seed=seed)
    _load_v1_checkpoint(agent, ck_path)

    # Load V2 library
    library = ExpertLibrary.load(str(lib_path))

    dijkstra = _dijkstra_cost(g, source, destination)

    # ── V3 rollout (adaptive lambda) ──────────────────────────────────────────
    t0 = time.perf_counter()
    v3 = run_v3_rollout(agent, g, data, source, destination, library)
    v3_infer_ms = (time.perf_counter() - t0) * 1000

    v3_reached  = v3["reached_goal"]
    v3_cost     = v3["cost"] if v3_reached else None
    v3_strict   = v3_reached
    v3_reasonable = v3_reached and v3_cost is not None and v3_cost <= K_THRESHOLD * max(dijkstra, 1e-9)
    v3_ratio    = (v3_cost / max(dijkstra, 1e-9)) if v3_cost is not None else None

    # ── V2 control rollout (fixed lambda=0.5) ─────────────────────────────────
    agent.encode(data)  # refresh cache
    v2_result = evaluate_agent(
        agent, g, source, destination, data,
        max_steps=MAX_STEPS_PER_EVAL,
        library=library,
        lambda_retr=LAMBDA_RETR_V2,
    )
    v2_reached = v2_result["reached_goal"]
    v2_cost    = v2_result["cost"] if v2_reached else None
    v2_strict  = v2_reached

    return {
        "seed":              seed,
        "scenario_id":       scenario_id,
        "src":               source,
        "dst":               destination,
        "euclidean_distance": euclidean,
        "v3_cost":           v3_cost,
        "v3_strict":         v3_strict,
        "v3_reasonable":     v3_reasonable,
        "v3_cost_ratio":     v3_ratio,
        "v3_infer_ms":       v3_infer_ms,
        "v3_mean_lambda":    v3["mean_lambda"],
        "v3_mean_max_sim":   v3["mean_max_sim"],
        "v2_repro_cost":     v2_cost,
        "v2_repro_strict":   v2_strict,
        "dijkstra_cost":     dijkstra,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate stats helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_sweep(json_path: pathlib.Path) -> dict[str, dict]:
    """Load a sweep JSON and index by (seed, scenario_id) key."""
    with open(json_path) as f:
        data = json.load(f)
    return {(c["seed"], c["scenario_id"]): c for c in data}


def _agg_from_index(index: dict[str, dict]) -> dict:
    """Compute published aggregate stats from a full sweep index."""
    data = list(index.values())
    n = len(data)
    strict = sum(1 for c in data if c.get("warm_strict"))
    costs = [c["warm_cost"] for c in data if c.get("warm_cost") is not None]
    wins = sum(
        1 for c in data
        if c.get("warm_cost") is not None and (
            c.get("cold_cost") is None or c["warm_cost"] < c["cold_cost"]
        )
    )
    mean_cost = sum(costs) / len(costs) if costs else float("nan")
    return {"n": n, "strict": strict, "wins": wins, "mean_cost": mean_cost}


def _print_table(v1_idx: dict, v2_idx: dict, v3_cells: list[dict]) -> None:
    v1 = _agg_from_index(v1_idx)
    v2 = _agg_from_index(v2_idx)
    n = len(v3_cells)
    v3_strict    = sum(1 for c in v3_cells if c["v3_strict"])
    v3_costs     = [c["v3_cost"] for c in v3_cells if c["v3_cost"] is not None]
    v3_mean_cost = sum(v3_costs) / len(v3_costs) if v3_costs else float("nan")
    # "win" = V3 reached goal and V2-control did not, or both reached and V3 cost < V2-control cost
    v3_wins = sum(
        1 for c in v3_cells
        if c["v3_cost"] is not None and (
            c["v2_repro_cost"] is None or c["v3_cost"] < c["v2_repro_cost"]
        )
    )
    v3_mean_lambda = sum(c["v3_mean_lambda"] for c in v3_cells) / max(n, 1)

    print()
    print(f"{'':20} {'V1 (all 25)':>11} {'V2 (all 25)':>11} {f'V3 ({n} cells)':>12}")
    print("-" * 56)
    print(f"{'goal-reach':20} {v1['strict']:>5}/{v1['n']:<5} {v2['strict']:>5}/{v2['n']:<5} {v3_strict:>5}/{n}")
    print(f"{'win vs ctrl':20} {v1['wins']:>5}/{v1['n']:<5} {v2['wins']:>5}/{v2['n']:<5} {v3_wins:>5}/{n}")
    mean_cost_v1 = f"{v1['mean_cost']:.0f}" if v1['mean_cost'] == v1['mean_cost'] else "n/a"
    mean_cost_v2 = f"{v2['mean_cost']:.0f}" if v2['mean_cost'] == v2['mean_cost'] else "n/a"
    mean_cost_v3 = f"{v3_mean_cost:.0f}" if v3_mean_cost == v3_mean_cost else "n/a"
    print(f"{'mean cost':20} {mean_cost_v1:>11} {mean_cost_v2:>11} {mean_cost_v3:>12}")
    print(f"{'mean lam (V3)':20} {'n/a':>11} {LAMBDA_RETR_V2:>11.2f} {v3_mean_lambda:>12.3f}")
    print()


def _check_v2_reproduction(v3_cells: list[dict], v2_idx: dict) -> None:
    """Warn if V2-control goal-reach deviates >10% from published V2 for the same cells."""
    matched_published = 0
    matched_control   = 0
    for c in v3_cells:
        key = (c["seed"], c["scenario_id"])
        pub = v2_idx.get(key)
        if pub is not None:
            matched_published += int(pub.get("warm_strict", False))
            matched_control   += int(c["v2_repro_strict"])

    n_matched = sum(1 for c in v3_cells if (c["seed"], c["scenario_id"]) in v2_idx)
    if n_matched == 0:
        print("V2 reproduction check: no matching cells found in V2 sweep JSON.")
        return

    if matched_published > 0:
        deviation = abs(matched_control - matched_published) / matched_published
        flag = "FLAG" if deviation > 0.10 else "OK"
        print(
            f"V2 reproduction check ({n_matched} matched cells): "
            f"published={matched_published}/{n_matched}  "
            f"control={matched_control}/{n_matched}  "
            f"deviation={deviation:.0%}  {flag}"
        )
    else:
        print(
            f"V2 reproduction check ({n_matched} matched cells): "
            f"published=0 strict; control={matched_control}/{n_matched}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep_v3_adaptive.json")
    ap.add_argument("--single-cell", action="store_true",
                    help="Evaluate only seed=42, scenario_id index 0 (smoke test)")
    ap.add_argument("--lambda-max", type=float, default=LAMBDA_MAX)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load V1 sweep as source of truth for cell definitions
    if not V1_SWEEP_PATH.exists():
        raise FileNotFoundError(
            f"V1 sweep JSON not found at {V1_SWEEP_PATH}.\n"
            f"Copy it from the V1 runs folder first."
        )
    with open(V1_SWEEP_PATH) as f:
        v1_cells = json.load(f)

    if args.single_cell:
        # Limit to the first cell of seed=42
        v1_cells = [c for c in v1_cells if c["seed"] == 42][:1]

    total = len(v1_cells)
    print(f"QWarm-RL V3 Evaluation — {total} cell(s), lambda_max={args.lambda_max}")
    print(f"Checkpoint dir : {AGENTS_DIR}")
    print(f"Library dir    : {LIBRARIES_DIR}")
    print(f"Output         : {out_path}")
    print()

    results: list[dict] = []
    for i, cell in enumerate(v1_cells, 1):
        seed        = cell["seed"]
        scenario_id = cell["scenario_id"]
        src         = cell["source"]
        dst         = cell["destination"]
        print(
            f"  [{i:>2}/{total}] seed={seed}  {src}->{dst}  "
            f"(scen={scenario_id})",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            row = evaluate_cell(cell)
        except Exception as exc:
            print(f"         ERROR: {exc}", flush=True)
            gc.collect()
            continue
        elapsed = time.perf_counter() - t0
        v3c = row["v3_cost"]
        print(
            f"         v3={'inf' if v3c is None else f'{v3c:.0f}'}  "
            f"strict={row['v3_strict']}  mean_lam={row['v3_mean_lambda']:.3f}  "
            f"mean_sim={row['v3_mean_max_sim']:.3f}  ({elapsed:.0f}s)",
            flush=True,
        )
        results.append(row)

        # Write incremental checkpoint so a crash doesn't lose progress
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=lambda x: None if (isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf"))) else x)

        gc.collect()

    # Final output
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda x: None if (isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf"))) else x)
    print(f"\nResults saved to {out_path}")

    if results:
        v1_idx = _load_sweep(V1_SWEEP_PATH)
        v2_idx = _load_sweep(V2_SWEEP_PATH) if V2_SWEEP_PATH.exists() else {}
        _print_table(v1_idx, v2_idx, results)
        if v2_idx:
            _check_v2_reproduction(results, v2_idx)


if __name__ == "__main__":
    main()
