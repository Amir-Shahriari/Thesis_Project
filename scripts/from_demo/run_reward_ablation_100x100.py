"""Reward-design ablation -- warm-start V1 on the 100x100 long-train subset cells.

Grid of conditions (default): gamma in {0.95, 0.99} x reward in {legacy, fixed}.
"fixed" is one of the config-flagged invalid-action variants documented in
docs/reward_variants.md (selected via --fixed-mode, default "nonterminal").

Cells: the 5 long-train-subset cells = first failing cell per seed from the
100x100 HEADLINE sweep (for seed 42 the subset used s1 because s0 already had a
pre-existing 4x result). Read directly from the first 5 records of
runs/v1_long_train_subset_results.json so the selection is identical.

Training: warm-start V1 at 1x budget -- HEADLINE protocol:
  n_iterations=10, episodes_per_iteration=200, grad_steps_per_episode=4,
  batch_size=64, hidden_dim=128, expert_ratio=0.30,
  pre_seed_n_states=3, pre_seed_k_paths=3.
gamma is passed to BOTH GNNDQN and train_gnn_dqn (expert MC returns).

Eval: greedy rollout, dual budget (300 primary / 1000 extended), Dijkstra
reference under the env cost formula. Output records use the same schema as
runs/v1_long_train_subset_results.json plus {gamma, reward_mode,
warm_strict_1000, warm_cost_1000, warm_cost_ratio_1000}.

Writes (incrementally, crash-resumable -- existing (cell, gamma, mode) combos
are skipped on rerun):
  runs/reward_ablation/reward_ablation_results.json
  runs/reward_ablation/agents/warm_seed{seed}_scen{scenario_id}_g{gamma}_{mode}.pt

Usage:
  uv run python scripts/run_reward_ablation_100x100.py                 # full grid
  uv run python scripts/run_reward_ablation_100x100.py --fixed-mode scaled
  uv run python scripts/run_reward_ablation_100x100.py --gammas 0.95 0.99
  uv run python scripts/run_reward_ablation_100x100.py --dry-run       # list runs only
"""
from __future__ import annotations

import argparse
import datetime
import functools
import gc
import heapq
import json
import math
import pathlib
import time
from typing import Any

import numpy as np
import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.utils.seeding import set_global_seed

try:
    from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA as _FaithfulQAOA
    _HAS_FAITHFUL_QAOA = True
except ImportError:
    _HAS_FAITHFUL_QAOA = False

SUBSET_JSON = pathlib.Path("runs/v1_long_train_subset_results.json")
OUT_DIR = pathlib.Path("runs/reward_ablation")
RESULTS_JSON = OUT_DIR / "reward_ablation_results.json"
AGENTS_DIR = OUT_DIR / "agents"

N_CELLS = 5                       # first failing cell per seed (subset order)
EVAL_BUDGETS = [300, 1000]
K_THRESHOLD = 3.0

GRID_CFG: dict[str, Any] = dict(
    grid_width=100,
    grid_height=100,
    extra_edges=4,
    deactivate_prob=0.30,
)

TRAIN_CFG: dict[str, Any] = dict(
    n_iterations=10,              # 1x budget
    episodes_per_iteration=200,
    grad_steps_per_episode=4,
    batch_size=64,
    hidden_dim=128,
    expert_ratio=0.30,
    pre_seed_n_states=3,
    pre_seed_k_paths=3,
)


def _json_float(v: Any) -> "float | None":
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v)


def _parse_grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _dijkstra(graph: dict, nodes: dict, src: str, dst: str) -> float:
    """Dijkstra with env cost formula: distance + 0.1*time + node_penalty."""
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


def _build_oracles(dyn_graph: DynamicGraph, seed: int) -> list:
    oracles: list = [
        ClassicalAStar(dyn_graph.nodes, dyn_graph.graph),
        QuantumInspiredStochasticOracle(dyn_graph.nodes, dyn_graph.graph),
    ]
    if _HAS_FAITHFUL_QAOA:
        oracles.append(_FaithfulQAOA(
            dyn_graph.nodes, dyn_graph.graph,
            p_layers=2, k_candidate_paths=5, max_edges_in_subgraph=20,
            n_optimiser_restarts=2, max_optimiser_iters=40, seed=seed,
        ))
    return oracles


def _make_env_class(reward_mode: str):
    """Env constructor for the condition. 'legacy' = plain PathfindingEnv."""
    if reward_mode == "legacy":
        return PathfindingEnv
    return functools.partial(PathfindingEnv, invalid_penalty_mode=reward_mode)


def _eval_dual_budget(agent: GNNDQN, g: DynamicGraph, src: str, dst: str, data) -> dict[int, dict]:
    agent.encode(data)
    results: dict[int, dict] = {}
    for budget in EVAL_BUDGETS:
        path = [src]
        visited: set[str] = {src}
        current = src
        cost = 0.0
        for _ in range(budget):
            valid = [
                nb for nb, d in g.graph[current].items()
                if d["active"] and g.nodes[nb]["active"] and nb not in visited
            ]
            if not valid:
                break
            action = agent.choose_action(current, valid, dst, data, epsilon=0.0)
            edge = g.graph[current][action]
            cost += edge["distance"] + 0.1 * edge["time"] + g.nodes[action]["node_penalty"]
            path.append(action)
            visited.add(action)
            current = action
            if current == dst:
                break
        reached = (path[-1] == dst)
        results[budget] = {"strict": reached, "cost": cost if reached else float("inf")}
    return results


def _save_checkpoint(agent: GNNDQN, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_raw_state_dict": agent._encoder_raw.state_dict(),
        "q_head_state_dict": agent._q_head_raw.state_dict(),
        "hidden_dim": TRAIN_CFG["hidden_dim"],
        "node_in_dim": 4,
    }, str(path))


def run_condition(cell: dict, gamma: float, reward_mode: str) -> dict:
    seed = cell["seed"]
    scenario_id = cell["scenario_id"]
    src = cell["source"]
    dst = cell["destination"]
    grid_seed = _parse_grid_seed(scenario_id)

    set_global_seed(seed)
    g = DynamicGraph(**GRID_CFG, seed=grid_seed)
    oracles = _build_oracles(g, seed)
    env_class = _make_env_class(reward_mode)

    agent = GNNDQN(
        node_in_dim=4,
        hidden_dim=TRAIN_CFG["hidden_dim"],
        gamma=gamma,
        seed=seed,
    )
    buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(seed),
    )

    t0 = time.perf_counter()
    train_gnn_dqn(
        g, env_class, agent, buf, oracles, [(src, dst)],
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=seed,
        pre_seed_n_states=TRAIN_CFG["pre_seed_n_states"],
        pre_seed_k_paths=TRAIN_CFG["pre_seed_k_paths"],
        gamma=gamma,
    )
    train_s = time.perf_counter() - t0

    data = dynamic_graph_to_pyg(g, device=agent.device)
    t0 = time.perf_counter()
    res = _eval_dual_budget(agent, g, src, dst, data)
    infer_ms = (time.perf_counter() - t0) * 1000

    dij = _dijkstra(g.graph, g.nodes, src, dst)
    r300, r1000 = res[300], res[1000]

    def _ratio(cost: float) -> float:
        return cost / dij if (cost < float("inf") and 0 < dij < float("inf")) else float("inf")

    ckpt = AGENTS_DIR / f"warm_seed{seed}_scen{scenario_id}_g{gamma}_{reward_mode}.pt"
    _save_checkpoint(agent, ckpt)

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "scenario_id": scenario_id,
        "source": src,
        "destination": dst,
        "grid": dict(GRID_CFG),
        "training_config": {
            **TRAIN_CFG,
            "total_rollouts": TRAIN_CFG["n_iterations"] * TRAIN_CFG["episodes_per_iteration"],
            "gamma": gamma,
        },
        # Ablation condition
        "gamma": gamma,
        "reward_mode": reward_mode,
        # Primary budget (max_steps=300) -- same field names as the subset JSON
        "warm_cost": _json_float(r300["cost"]),
        "warm_strict": r300["strict"],
        "warm_reasonable": r300["strict"] and _ratio(r300["cost"]) <= K_THRESHOLD,
        "warm_cost_ratio": _json_float(_ratio(r300["cost"])),
        # Extended budget (max_steps=1000)
        "warm_cost_1000": _json_float(r1000["cost"]),
        "warm_strict_1000": r1000["strict"],
        "warm_cost_ratio_1000": _json_float(_ratio(r1000["cost"])),
        "dijkstra_cost": _json_float(dij),
        "warm_infer_ms": infer_ms,
        "warm_train_s": train_s,
        "k_threshold": K_THRESHOLD,
        "checkpoint": str(ckpt.resolve()),
        "baseline_1x_cost": None,
        "baseline_1x_strict": False,
        "compute_multiplier": 1,
    }
    del agent, buf, oracles, g, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.95, 0.99],
                    help="discount factors to sweep (default: 0.95 0.99)")
    ap.add_argument("--fixed-mode", choices=["nonterminal", "scaled"],
                    default="nonterminal",
                    help="which reward variant is the 'fixed' arm (default: nonterminal)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the planned runs and exit without training")
    args = ap.parse_args()

    cells = json.loads(SUBSET_JSON.read_text())[:N_CELLS]
    modes = ["legacy", args.fixed_mode]
    runs = [
        (cell, gamma, mode)
        for cell in cells
        for gamma in args.gammas
        for mode in modes
    ]

    print(f"Reward ablation -- {len(cells)} cells x {len(args.gammas)} gammas "
          f"x {len(modes)} reward modes = {len(runs)} runs")
    for cell, gamma, mode in runs:
        print(f"  seed={cell['seed']:<7} {cell['scenario_id']:<22} "
              f"gamma={gamma}  mode={mode}")
    if args.dry_run:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if RESULTS_JSON.exists():
        records = json.loads(RESULTS_JSON.read_text())
        print(f"Resuming: {len(records)} existing records in {RESULTS_JSON}")
    done_keys = {(r["seed"], r["scenario_id"], r["gamma"], r["reward_mode"])
                 for r in records}

    wall_t0 = time.perf_counter()
    for i, (cell, gamma, mode) in enumerate(runs, 1):
        key = (cell["seed"], cell["scenario_id"], gamma, mode)
        if key in done_keys:
            print(f"[{i:>2}/{len(runs)}] {key} -- already done, skipping")
            continue
        print(f"\n[{i:>2}/{len(runs)}] seed={cell['seed']} {cell['scenario_id']} "
              f"{cell['source']}->{cell['destination']}  gamma={gamma}  mode={mode}",
              flush=True)
        t_run = time.perf_counter()
        try:
            rec = run_condition(cell, gamma, mode)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}", flush=True)
            traceback.print_exc()
            gc.collect()
            continue
        rec["wall_run_s"] = time.perf_counter() - t_run
        records.append(rec)
        with open(RESULTS_JSON, "w") as f:
            json.dump(records, f, indent=2)
        dij = rec["dijkstra_cost"]
        dij_str = f"{dij:.1f}" if dij is not None else "unreachable"
        print(f"  strict={rec['warm_strict']} (1000: {rec['warm_strict_1000']})  "
              f"cost={rec['warm_cost']}  dij={dij_str}  "
              f"train={rec['warm_train_s']:.0f}s  wall={rec['wall_run_s']:.0f}s",
              flush=True)

    total = time.perf_counter() - wall_t0
    print(f"\nDone. {len(records)} records in {RESULTS_JSON}  "
          f"(wall {total/3600:.2f} h)")

    # Compact summary: reach rate per (gamma, mode)
    print("\nGoal-reach (strict@300 / strict@1000) per condition:")
    for gamma in args.gammas:
        for mode in modes:
            sel = [r for r in records
                   if r["gamma"] == gamma and r["reward_mode"] == mode]
            if not sel:
                continue
            s300 = sum(r["warm_strict"] for r in sel)
            s1000 = sum(r["warm_strict_1000"] for r in sel)
            print(f"  gamma={gamma}  mode={mode:<12} "
                  f"{s300}/{len(sel)} @300   {s1000}/{len(sel)} @1000")


if __name__ == "__main__":
    main()
