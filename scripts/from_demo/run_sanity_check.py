"""Task 0 — Sanity check: current codebase reproduces V1 behavior.

Compares greedy rollout on the same cell using:
  Step 1  current project codebase
  Step 2  V1 legacy source (C:/Users/amirh/Desktop/backups/V1/src on PYTHONPATH)

Saves runs/unified_benchmark_sanity.json.
"""
from __future__ import annotations

import heapq
import json
import os
import pathlib
import subprocess
import sys
import time

import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg

# ── Config ────────────────────────────────────────────────────────────────────
CELL = dict(
    seed=42,
    scenario_id="seed191664964_s0",
    source="Node_639",
    destination="Node_8583",
)
GRID_SEED = 191664964
PERTURB_STEPS = 5
MAX_STEPS = 1000

CKPT_PATH = pathlib.Path("runs/agents_v1/warm_seed42_scenseed191664964_s0.pt")
V1_SRC    = pathlib.Path("C:/Users/amirh/Desktop/backups/V1/src")
OUT_PATH  = pathlib.Path("runs/unified_benchmark_sanity.json")
HELPER    = pathlib.Path(__file__).parent / "_sanity_v1_eval.py"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_ckpt(agent: GNNDQN, path: pathlib.Path) -> None:
    ck = torch.load(str(path), map_location=agent.device, weights_only=False)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None


def _dijkstra(graph: dict, nodes: dict, src: str, dst: str) -> float:
    q = [(0.0, src)]; best = {src: 0.0}; vis: set = set()
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


def _greedy_rollout(agent: GNNDQN, g: DynamicGraph, data, src: str, dst: str):
    agent.encode(data)
    path = [src]; vis = {src}; cur = src; cost = 0.0
    t0 = time.perf_counter()
    for _ in range(MAX_STEPS):
        valid = [nb for nb, d in g.graph[cur].items()
                 if d["active"] and g.nodes[nb]["active"] and nb not in vis]
        if not valid:
            break
        action = agent.choose_action(cur, valid, dst, data, epsilon=0.0)
        e = g.graph[cur][action]
        cost += e["distance"] + 0.1 * e["time"] + g.nodes[action]["node_penalty"]
        path.append(action); vis.add(action); cur = action
        if cur == dst:
            break
    ms = (time.perf_counter() - t0) * 1000
    reached = (path[-1] == dst)
    return {"reached": reached, "cost": cost if reached else float("inf"), "infer_ms": ms}


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("TASK 0 — Sanity check")
    print(f"  Cell  : seed={CELL['seed']}  {CELL['source']} -> {CELL['destination']}")
    print(f"  Scen  : {CELL['scenario_id']}")
    print("=" * 64)

    # Build graph (shared for both evaluations)
    g = DynamicGraph(
        grid_width=100, grid_height=100,
        extra_edges=4, deactivate_prob=0.30,
        seed=GRID_SEED,
    )
    for _ in range(PERTURB_STEPS):
        g.update_graph()
    data = dynamic_graph_to_pyg(g)
    dij = _dijkstra(g.graph, g.nodes, CELL["source"], CELL["destination"])

    # ── Step 1: current codebase ──────────────────────────────────────────────
    print("\nStep 1  current codebase …")
    agent_cur = GNNDQN(node_in_dim=4, hidden_dim=128, seed=CELL["seed"])
    _load_ckpt(agent_cur, CKPT_PATH)
    t_cur = time.perf_counter()
    cur = _greedy_rollout(agent_cur, g, data, CELL["source"], CELL["destination"])
    cur["wall_s"] = time.perf_counter() - t_cur
    cur_cost_str = "inf" if cur["cost"] == float("inf") else f"{cur['cost']:.4f}"
    print(f"  reached={cur['reached']}  cost={cur_cost_str}  infer_ms={cur['infer_ms']:.1f}")

    # ── Step 2: V1 legacy via subprocess ─────────────────────────────────────
    print("\nStep 2  V1 legacy codebase (subprocess) …")
    env_v1 = os.environ.copy()
    env_v1["PYTHONPATH"] = str(V1_SRC.resolve()) + os.pathsep + env_v1.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(HELPER),
         "--ckpt",      str(CKPT_PATH.resolve()),
         "--src",       CELL["source"],
         "--dst",       CELL["destination"],
         "--grid-seed", str(GRID_SEED),
         "--seed",      str(CELL["seed"])],
        capture_output=True, text=True, env=env_v1,
    )
    if proc.returncode != 0:
        print("  ERROR running V1 helper:")
        print(proc.stderr[-2000:])
        sys.exit(1)

    try:
        v1_raw = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        print("  ERROR parsing V1 helper output:")
        print(proc.stdout[:500])
        print(proc.stderr[:500])
        sys.exit(1)

    v1 = {
        "reached": bool(v1_raw["reached"]),
        "cost":    v1_raw["cost"] if v1_raw["cost"] is not None else float("inf"),
        "infer_ms": v1_raw["infer_ms"],
    }
    v1_cost_str = "inf" if v1["cost"] == float("inf") else f"{v1['cost']:.4f}"
    print(f"  reached={v1['reached']}  cost={v1_cost_str}  infer_ms={v1['infer_ms']:.1f}")

    # ── Step 3: compare ───────────────────────────────────────────────────────
    print("\nStep 3  comparison …")
    cur_cost = cur["cost"]
    leg_cost = v1["cost"]

    if cur_cost == float("inf") and leg_cost == float("inf"):
        delta = 0.0
        status = "SANITY OK: both codebases failed to reach goal (cost=inf). " \
                 "Behaviour is identical."
    elif cur_cost == float("inf") or leg_cost == float("inf"):
        delta = float("inf")
        status = "SANITY FAILED: one codebase reached goal but not the other."
    else:
        delta = abs(cur_cost - leg_cost) / max(abs(leg_cost), 1e-9)
        if delta <= 0.10:
            status = (
                f"SANITY OK: current codebase reproduces V1 within ±10% "
                f"(delta={delta:.2%}), the inference-rule comparison is "
                f"methodologically sound."
            )
        else:
            status = (
                f"SANITY FAILED: current codebase differs from V1 by {delta:.2%}. "
                f"Halting before main benchmark. Investigate codebase drift in "
                f"encoder/Q-head/env."
            )

    print(f"\n  current_cost  = {cur_cost if cur_cost < 1e15 else 'inf'}")
    print(f"  legacy_cost   = {leg_cost if leg_cost < 1e15 else 'inf'}")
    print(f"  dijkstra_ref  = {dij:.4f}")
    print(f"  relative_delta= {delta:.4f}")
    print(f"\n  {status}")

    result = {
        "cell": CELL,
        "grid_seed": GRID_SEED,
        "perturbation_steps": PERTURB_STEPS,
        "dijkstra_ref": dij if dij < 1e15 else None,
        "current_codebase": {
            "reached": cur["reached"],
            "cost": cur["cost"] if cur["cost"] < 1e15 else None,
            "infer_ms": cur["infer_ms"],
        },
        "legacy_v1": {
            "reached": v1["reached"],
            "cost": v1["cost"] if v1["cost"] < 1e15 else None,
            "infer_ms": v1["infer_ms"],
        },
        "relative_delta": delta if delta < 1e15 else None,
        "sanity_passed": delta <= 0.10,
        "status": status,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")

    if not result["sanity_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
