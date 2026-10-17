"""Goal-adjacent seeding ablation at 50x50: is the warm agent's reach advantage
carried by the oracle demonstrations, or by the warm-only goal-adjacent terminal
transitions ("+100 at the doorstep") that the cold agent never receives?

Pre-registered in runs/goaladj_ablation_50x50/PREREGISTRATION.md, committed
before any run. Do not change the settings below after results exist.

Arms (all trained and evaluated inside every cell, on one recorded stack):
  warm_no_goaladj  the thesis warm agent with the goal-adjacent block removed
                   (oracle library + per-iteration refresh kept, rho=0.30)
  cold_goaladj     the thesis cold agent plus the same goal-adjacent terminal
                   transitions, kept in the expert pool at rho=0.30 for the
                   whole run (no oracle demonstrations)
  persistent       the thesis warm agent (replication reference)
  cold             the thesis cold agent (replication reference)

Everything else (grid, scenarios, training budget, masked action space,
evaluation) is imported from the 50x50 sweep via run_persistence_ablation.

Environment:
  QWARM_GA_ARMS      comma list (default warm_no_goaladj,cold_goaladj,persistent,cold)
  QWARM_SEEDS        subset of seeds, for sharding across processes
  QWARM_OUT_DIR      output dir (default runs/goaladj_ablation_50x50)
  QWARM_SMOKE=1      code-path smoke test only: 1 cell, 1 iteration, 10 episodes. NOT evidence.
  QWARM_ALLOW_CPU=1  permit a non-smoke run without CUDA (default: refuse)

Usage:
  .venv/Scripts/python.exe scripts/run_goaladj_ablation.py              # run
  .venv/Scripts/python.exe scripts/run_goaladj_ablation.py --aggregate  # stats only
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_persistence_ablation as pa  # noqa: E402  (sets QWARM_MASK_VISITED=1 first)
from qwarm.env.dynamic_graph import DynamicGraph  # noqa: E402
from qwarm.env.pathfinding_env import PathfindingEnv  # noqa: E402
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg  # noqa: E402
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer, Transition  # noqa: E402
from qwarm.training.train_gnn_dqn import train_gnn_dqn  # noqa: E402
from qwarm.utils.seeding import set_global_seed  # noqa: E402

base = pa.base
TC = pa.TC
SMOKE = pa.SMOKE
ALL_ARMS = ["warm_no_goaladj", "cold_goaladj", "persistent", "cold"]
ARMS = [a.strip() for a in os.environ.get("QWARM_GA_ARMS", ",".join(ALL_ARMS)).split(",") if a.strip()]
for _a in ARMS:
    assert _a in ALL_ARMS, f"unknown arm {_a}"
OUT_DIR = pathlib.Path(os.environ.get(
    "QWARM_OUT_DIR",
    "runs/goaladj_ablation_smoke" if SMOKE else "runs/goaladj_ablation_50x50"))


def write_provenance() -> dict:
    dev, name, cap = "cpu", None, None
    if torch.cuda.is_available():
        dev = "cuda:0"
        name = torch.cuda.get_device_name(0)
        cap = list(torch.cuda.get_device_capability(0))
    try:  # hash the diff as raw bytes (the persistence run's text decode failed silently)
        diff = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        diff = None
    rec = {
        "label": "50x50 goal-adjacent seeding ablation" + (" (SMOKE)" if SMOKE else ""),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "resolved_device": dev,
        "gpu_name": name,
        "compute_capability": cap,
        "sys_executable": sys.executable,
        "python": platform.python_version(),
        "numpy_version": np.__version__,
        "git_rev": pa._git("rev-parse", "--short", "HEAD"),
        "git_dirty_files": [l.split(maxsplit=1)[1]
                            for l in (pa._git("status", "--porcelain", "--untracked-files=no") or "").splitlines()
                            if len(l.split(maxsplit=1)) == 2],
        "git_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "arms": ARMS,
        "n_iterations": pa.N_ITER,
        "episodes_per_iteration": pa.EPISODES,
        "smoke": SMOKE,
        "env": {k: os.environ.get(k) for k in (
            "QWARM_MASK_VISITED", "QWARM_SEEDS", "QWARM_GA_ARMS", "QWARM_OUT_DIR",
            "QWARM_SMOKE", "CUDA_VISIBLE_DEVICES")},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(s) for s in pa.SEEDS)
    with open(OUT_DIR / f"provenance_seeds_{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=2)
    return rec


def goal_adjacent_transitions(g: DynamicGraph, queries: list[tuple[str, str]]) -> list[Transition]:
    """The goal-adjacent block of train_gnn_dqn, verbatim."""
    out = []
    for src, dst in queries:
        for node_id, nbrs in g.graph.items():
            if dst not in nbrs:
                continue
            e = nbrs[dst]
            if not (e["active"] and g.nodes[node_id]["active"] and g.nodes[dst]["active"]):
                continue
            step_cost = e["distance"] + 0.1 * e["time"] + g.nodes[dst]["node_penalty"]
            out.append(Transition(
                state_node=node_id, action_node=dst, reward=-step_cost + 100.0,
                next_state_node=dst, done=True, valid_next_actions=[],
                is_expert=True, iteration_added=0, goal_node=dst,
            ))
    return out


def _train(g, agent, buf, oracles, queries, re_seed: bool, seed: int, goal_adj: bool = True) -> None:
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=pa.N_ITER,
        episodes_per_iteration=pa.EPISODES,
        grad_steps_per_episode=TC["grad_steps_per_episode"],
        batch_size=TC["batch_size"],
        re_seed_experts_each_iteration=re_seed,
        seed=seed,
        pre_seed_n_states=TC["pre_seed_n_states"],
        pre_seed_k_paths=TC["pre_seed_k_paths"],
        env_kwargs={"mask_visited": True},
        goal_adjacent_seeding=goal_adj,
    )


def run_arm(arm: str, seed: int, scenario) -> dict:
    set_global_seed(seed)
    src, dst = scenario.source_node, scenario.destination_node
    queries = [(src, dst)]
    g = DynamicGraph(**base.GRID_CFG, seed=scenario.grid_seed)
    agent = pa._agent(seed)
    extra: dict = {}
    t0 = time.perf_counter()

    if arm in ("persistent", "warm_no_goaladj"):
        oracles = base._build_oracles(g, seed)
        buf = ExpertReplayBuffer(expert_ratio=TC["expert_ratio"], rng=np.random.default_rng(seed))
        _train(g, agent, buf, oracles, queries, re_seed=True, seed=seed,
               goal_adj=(arm == "persistent"))
    elif arm == "cold":
        buf = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(seed))
        _train(g, agent, buf, [], queries, re_seed=False, seed=seed)
    else:  # cold_goaladj
        buf = ExpertReplayBuffer(expert_ratio=TC["expert_ratio"], rng=np.random.default_rng(seed))
        buf.expert_pool.extend(goal_adjacent_transitions(g, queries))
        extra["goal_adjacent_transitions"] = len(buf.expert_pool)
        _train(g, agent, buf, [], queries, re_seed=False, seed=seed)
        extra["expert_pool_at_end"] = len(buf.expert_pool)

    train_s = time.perf_counter() - t0
    data = dynamic_graph_to_pyg(g, device=agent.device)
    res = base._eval_dual_budget(agent, g, src, dst, data)
    dij = base._dijkstra(g.graph, g.nodes, src, dst)
    row = {"arm": arm, "train_s": train_s, "grad_steps_total": agent._step_count,
           "dijkstra_cost": base._json_float(dij), **extra}
    for b in pa.BUDGETS:
        sfx = "" if b == 300 else f"_{b}"
        row["strict" + sfx] = bool(res[b]["strict"])
        row["cost" + sfx] = base._json_float(res[b]["cost"])
    del agent, buf
    gc.collect()
    return row


PRIMARY = [("warm_no_goaladj", "cold"), ("cold_goaladj", "cold")]
SECONDARY = [("persistent", "warm_no_goaladj"), ("persistent", "cold_goaladj"), ("persistent", "cold"),
             ("warm_no_goaladj", "cold_goaladj")]


def aggregate(out_dir: pathlib.Path) -> dict:
    cells = []
    for f in sorted(out_dir.glob("cells_seed_*.json")):
        cells += json.load(open(f))
    by_arm: dict[str, list] = {}
    for c in sorted(cells, key=lambda c: (c["seed"], c["scenario_id"])):
        for arm, row in c["arms"].items():
            by_arm.setdefault(arm, []).append(row)
    n = len(cells)
    agg: dict = {"n_cells": n, "arms": {}, "tests": {}}
    for arm, rows in by_arm.items():
        agg["arms"][arm] = {"reach": sum(r["strict"] for r in rows), "n": len(rows),
                            "reach_1000": sum(r["strict_1000"] for r in rows)}
    for a1, a2 in PRIMARY + SECONDARY:
        if a1 in by_arm and a2 in by_arm and len(by_arm[a1]) == len(by_arm[a2]) == n:
            agg["tests"][f"{a1}_vs_{a2}"] = {
                "reach": pa.exact_mcnemar([r["strict"] for r in by_arm[a1]],
                                          [r["strict"] for r in by_arm[a2]]),
                "reach_1000": pa.exact_mcnemar([r["strict_1000"] for r in by_arm[a1]],
                                               [r["strict_1000"] for r in by_arm[a2]]),
                "cost": pa.wilcoxon_cost([r["cost"] for r in by_arm[a1]],
                                         [r["cost"] for r in by_arm[a2]]),
            }
    ps = sorted(((k, agg["tests"][k]["reach"]["p_exact"]) for k in
                 (f"{a}_vs_{b}" for a, b in PRIMARY) if k in agg["tests"]
                 and agg["tests"][k]["reach"]["p_exact"] is not None), key=lambda kp: kp[1])
    running = 0.0
    for i, (k, p) in enumerate(ps):  # Holm over the two pre-registered primary comparisons
        running = max(running, min(1.0, (len(ps) - i) * p))
        agg["tests"][k]["reach"]["p_holm"] = running
    agg["eval_state_consistent_cells"] = sum(
        1 for c in cells if len({json.dumps(r["dijkstra_cost"]) for r in c["arms"].values()}) == 1)
    with open(out_dir / "aggregate.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    return agg


def main() -> None:
    if "--aggregate" in sys.argv:
        print(json.dumps(aggregate(OUT_DIR), indent=2))
        return
    prov = write_provenance()
    if not SMOKE and not prov["cuda_available"] and os.environ.get("QWARM_ALLOW_CPU") != "1":
        raise SystemExit(f"Refusing a full run on {prov['torch_version']} without CUDA.")
    print(f"arms={ARMS}  iters={pa.N_ITER}  eps={pa.EPISODES}  torch={prov['torch_version']}  "
          f"device={prov['resolved_device']}  smoke={SMOKE}", flush=True)
    tag = "_".join(str(s) for s in pa.SEEDS)
    out = OUT_DIR / f"cells_seed_{tag}.json"
    done = json.load(open(out)) if out.exists() else []
    done_ids = {(c["seed"], c["scenario_id"]) for c in done}
    for seed in pa.SEEDS:
        rng = np.random.default_rng(seed)
        tmpl = {k: base.GRID_CFG[k] for k in ("grid_width", "grid_height",
                                               "extra_edges", "deactivate_prob")}
        scenarios = base.sample_scenarios(tmpl, n_scenarios=base.N_SCENARIOS, rng=rng)[:pa.N_SCEN]
        for sc in scenarios:
            if (seed, sc.scenario_id) in done_ids:
                print(f"skip (done) seed={seed} {sc.scenario_id}", flush=True)
                continue
            cell = {"seed": seed, "scenario_id": sc.scenario_id, "grid_seed": sc.grid_seed,
                    "source": sc.source_node, "destination": sc.destination_node, "arms": {}}
            for arm in ARMS:
                row = run_arm(arm, seed, sc)
                cell["arms"][arm] = row
                print(f"  seed={seed} {sc.scenario_id} {arm:16s} reach={row['strict']} "
                      f"cost={row['cost']} steps={row['grad_steps_total']} "
                      f"t={row['train_s']:.0f}s", flush=True)
            done.append(cell)
            with open(out, "w") as fh:  # incremental: a crash loses at most one cell
                json.dump(done, fh, indent=2)
    print(json.dumps(aggregate(OUT_DIR), indent=2), flush=True)


if __name__ == "__main__":
    main()
