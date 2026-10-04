"""Evaluate V1 / V2 / V3 modes on the long-trained agent + rebuilt library.

Mirrors the cell in runs/sweep_v3_adaptive.json for seed=42, scenario_id
seed191664964_s0 (src=Node_639, dst=Node_8583), but uses the 4x-compute
long-trained checkpoint and the library rebuilt against its encoder.

Results are appended to runs/v3_long_train_results.json.

CLI:
    uv run python scripts/run_v3_long_train_evaluation.py
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

# ──────────────────────────────────────────────────────────────────────────────
# Config — must match the trained run and the rebuild script
# ──────────────────────────────────────────────────────────────────────────────

GRID_SEED = 191664964
GRID_TEMPLATE = {
    "grid_width": 100,
    "grid_height": 100,
    "extra_edges": 4,
    "deactivate_prob": 0.30,
}
PERTURBATION_STEPS = 5
MAX_STEPS = 1000
K_THRESHOLD = 3.0
LAMBDA_MAX = 1.0
LAMBDA_RETR_V2 = 0.5
HIDDEN_DIM = 128

SOURCE = "Node_639"
DESTINATION = "Node_8583"
AGENT_PATH = pathlib.Path("runs/agents_v1_long/warm_seed42_4x_compute.pt")
LIBRARY_PATH = pathlib.Path("runs/libraries_long/seed_42_long.pt")
RESULTS_PATH = pathlib.Path("runs/v3_long_train_results.json")


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


def _load_checkpoint(agent: GNNDQN, path: pathlib.Path) -> None:
    ck = torch.load(str(path), map_location=agent.device, weights_only=False)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None


def _rollout_v1(agent, dyn_graph, data, source, destination) -> dict:
    """Greedy rollout with no library (pure network Q-values)."""
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0

    for _ in range(MAX_STEPS):
        valid = [
            nb for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break
        action = agent.choose_action(current, valid, destination, data, epsilon=0.0)
        edge = dyn_graph.graph[current][action]
        cost += edge["distance"] + 0.1 * edge["time"] + dyn_graph.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    return {"reached_goal": reached, "cost": cost if reached else float("inf"), "path_len": len(path)}


def _rollout_v2(agent, dyn_graph, data, source, destination, library) -> dict:
    """Fixed-lambda retrieval blending (V2 control, lambda=0.5)."""
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0

    for _ in range(MAX_STEPS):
        valid = [
            nb for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break
        action = agent.select_action_with_library(
            current, valid, destination, data, library,
            adaptive_lambda=False,
            lambda_retr=LAMBDA_RETR_V2,
        )
        edge = dyn_graph.graph[current][action]
        cost += edge["distance"] + 0.1 * edge["time"] + dyn_graph.nodes[action]["node_penalty"]
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached = path[-1] == destination
    return {"reached_goal": reached, "cost": cost if reached else float("inf"), "path_len": len(path)}


def _rollout_v3(agent, dyn_graph, data, source, destination, library) -> dict:
    """Adaptive-lambda retrieval blending (V3)."""
    agent.encode(data)
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0
    lambdas: list[float] = []
    max_sims: list[float] = []

    for _ in range(MAX_STEPS):
        valid = [
            nb for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
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
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not AGENT_PATH.exists():
        raise FileNotFoundError(f"Long-trained agent not found: {AGENT_PATH}")
    if not LIBRARY_PATH.exists():
        raise FileNotFoundError(
            f"Rebuilt library not found: {LIBRARY_PATH}\n"
            f"Run rebuild_library_from_agent.py first."
        )

    print("\nV3 Long-Train Evaluation")
    print(f"  Agent   : {AGENT_PATH}")
    print(f"  Library : {LIBRARY_PATH}")
    print(f"  Query   : {SOURCE} -> {DESTINATION}")
    print()

    # ── Build graph + apply perturbation steps ─────────────────────────────────
    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=GRID_SEED,
    )
    for _ in range(PERTURBATION_STEPS):
        g.update_graph()
    data = dynamic_graph_to_pyg(g)

    dijkstra = _dijkstra_cost(g, SOURCE, DESTINATION)
    print(f"  Dijkstra cost: {dijkstra:.2f}", flush=True)

    # ── Load agent and library ──────────────────────────────────────────────────
    agent = GNNDQN(node_in_dim=4, hidden_dim=HIDDEN_DIM, seed=42)
    _load_checkpoint(agent, AGENT_PATH)

    library = ExpertLibrary.load(str(LIBRARY_PATH))
    print(f"  Library size : {len(library)} entries", flush=True)

    # ── V1 rollout (no library) ────────────────────────────────────────────────
    print("Running V1 (no library)...", flush=True)
    t0 = time.perf_counter()
    v1 = _rollout_v1(agent, g, data, SOURCE, DESTINATION)
    v1_ms = (time.perf_counter() - t0) * 1000

    v1_strict = v1["reached_goal"]
    v1_cost = v1["cost"] if v1_strict else None
    v1_reasonable = v1_strict and v1_cost is not None and v1_cost <= K_THRESHOLD * max(dijkstra, 1e-9)
    v1_ratio = (v1_cost / max(dijkstra, 1e-9)) if v1_cost is not None else None
    print(f"  strict={v1_strict}  cost={v1_cost}  infer_ms={v1_ms:.1f}", flush=True)

    # ── V2 rollout (fixed lambda=0.5) ──────────────────────────────────────────
    print("Running V2 (fixed lambda=0.5)...", flush=True)
    t0 = time.perf_counter()
    v2 = _rollout_v2(agent, g, data, SOURCE, DESTINATION, library)
    v2_ms = (time.perf_counter() - t0) * 1000

    v2_strict = v2["reached_goal"]
    v2_cost = v2["cost"] if v2_strict else None
    v2_reasonable = v2_strict and v2_cost is not None and v2_cost <= K_THRESHOLD * max(dijkstra, 1e-9)
    v2_ratio = (v2_cost / max(dijkstra, 1e-9)) if v2_cost is not None else None
    print(f"  strict={v2_strict}  cost={v2_cost}  infer_ms={v2_ms:.1f}", flush=True)

    # ── V3 rollout (adaptive lambda) ───────────────────────────────────────────
    print("Running V3 (adaptive lambda)...", flush=True)
    t0 = time.perf_counter()
    v3 = _rollout_v3(agent, g, data, SOURCE, DESTINATION, library)
    v3_ms = (time.perf_counter() - t0) * 1000

    v3_strict = v3["reached_goal"]
    v3_cost = v3["cost"] if v3_strict else None
    v3_reasonable = v3_strict and v3_cost is not None and v3_cost <= K_THRESHOLD * max(dijkstra, 1e-9)
    v3_ratio = (v3_cost / max(dijkstra, 1e-9)) if v3_cost is not None else None
    print(f"  strict={v3_strict}  cost={v3_cost}  mean_lambda={v3['mean_lambda']:.3f}  infer_ms={v3_ms:.1f}", flush=True)

    # ── Append record ──────────────────────────────────────────────────────────
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "agent_path": str(AGENT_PATH),
        "library_path": str(LIBRARY_PATH),
        "grid_seed": GRID_SEED,
        "source": SOURCE,
        "destination": DESTINATION,
        "dijkstra_cost": dijkstra,
        "perturbation_steps": PERTURBATION_STEPS,
        "v1_strict": v1_strict,
        "v1_cost": v1_cost,
        "v1_reasonable": v1_reasonable,
        "v1_cost_ratio": v1_ratio,
        "v1_infer_ms": v1_ms,
        "v2_strict": v2_strict,
        "v2_cost": v2_cost,
        "v2_reasonable": v2_reasonable,
        "v2_cost_ratio": v2_ratio,
        "v2_infer_ms": v2_ms,
        "v3_strict": v3_strict,
        "v3_cost": v3_cost,
        "v3_reasonable": v3_reasonable,
        "v3_cost_ratio": v3_ratio,
        "v3_infer_ms": v3_ms,
        "v3_mean_lambda": v3["mean_lambda"],
        "v3_mean_max_sim": v3["mean_max_sim"],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as fh:
            existing = json.load(fh)
    existing.append(record)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(existing, fh, indent=2, default=lambda x: None if isinstance(x, float) and (x != x or abs(x) == float("inf")) else x)

    print(f"\nRecord appended to {RESULTS_PATH}")
    print(f"  V1: {'reached' if v1_strict else 'MISS'}  cost={v1_cost}  ratio={v1_ratio}")
    print(f"  V2: {'reached' if v2_strict else 'MISS'}  cost={v2_cost}  ratio={v2_ratio}")
    print(f"  V3: {'reached' if v3_strict else 'MISS'}  cost={v3_cost}  ratio={v3_ratio}  mean_lambda={v3['mean_lambda']:.3f}")


if __name__ == "__main__":
    main()
