"""V1 long-train (4x compute) entry point — RECREATED.

The original script that produced runs/v1_long_train_subset_results.json and
the checkpoints in runs/agents_v1_long/ was lost; only its consumers survived
(build_libraries_long_train_subset.py, run_v3_long_train_subset_evaluation.py,
audit_eval_reachability_longtrain.py). This recreation is built from:

  * the training_config recorded verbatim in every record of
    runs/v1_long_train_subset_results.json (n_iterations=40,
    episodes_per_iteration=200, grad_steps_per_episode=4, batch_size=64,
    hidden_dim=128, expert_ratio=0.3, pre_seed_n_states=3, pre_seed_k_paths=3),
    i.e. configs/default.yaml with n_iterations 10 -> 40;
  * the cell protocol of scripts/run_multi_seed_warm_vs_cold.py (_run_cell,
    warm arm) which produced the 1x baseline the subset was selected from;
  * the eval-state identification by audit_eval_reachability_longtrain.py:
    every cell's recorded dijkstra_cost matches the graph after
    update_graph(iteration=1..40) — exactly the state train_gnn_dqn leaves
    the graph in after 40 iterations, confirming eval-on-final-train-state;
  * the checkpoint schema observed in the existing .pt files.

The oracle pool is not recorded in the subset JSON; the sweep default
("full") is used, matching the 50x50 aggregate's recorded 3-oracle pool.

Subset selection: the 15 cells where the 1x sweep failed
(warm_strict=False in runs/sweep_v1_on_100x100.json).

Usage:
    # Full 4x training run over the failing subset (hours per cell on GPU):
    uv run python scripts/run_v1_long_train.py

    # Evaluation phase only, from an existing checkpoint (no training) --
    # validates that the recreated eval protocol reproduces the recorded
    # cost/reach for that cell:
    uv run python scripts/run_v1_long_train.py \\
        --eval-only 7/seed2029167941_s3 --device cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import numpy as np
import torch

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.oracles.pool import build_oracle_pool, pool_pre_seed_k_paths
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.utils.seeding import set_global_seed

GRID_CFG = {
    "grid_width": 100,
    "grid_height": 100,
    "extra_edges": 4,
    "deactivate_prob": 0.30,
}
TRAIN_CFG = {
    "n_iterations": 40,
    "episodes_per_iteration": 200,
    "grad_steps_per_episode": 4,
    "batch_size": 64,
    "hidden_dim": 128,
    "expert_ratio": 0.30,
    "pre_seed_n_states": 3,
    "pre_seed_k_paths": 3,
    "oracle_pool": "full",
}
K_THRESHOLD = 3.0
EVAL_MAX_STEPS = 300  # evaluate_with_reasonableness default, as in the sweep
COMPUTE_MULTIPLIER = 4

SWEEP_1X_JSON = pathlib.Path("runs/sweep_v1_on_100x100.json")
SUBSET_JSON = pathlib.Path("runs/v1_long_train_subset_results.json")
AGENTS_DIR = pathlib.Path("runs/agents_v1_long")
DEFAULT_OUT = pathlib.Path("runs/v1_long_train_results_recreated.json")


def _grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _build_eval_graph(scenario_id: str) -> DynamicGraph:
    """Rebuild the iteration-40 eval state: the graph train_gnn_dqn leaves
    behind after update_graph(iteration=1..40) (confirmed by the long-train
    reachability audit)."""
    g = DynamicGraph(**GRID_CFG, seed=_grid_seed(scenario_id))
    for i in range(1, TRAIN_CFG["n_iterations"] + 1):
        g.update_graph(iteration=i)
    return g


def _evaluate(agent: GNNDQN, g: DynamicGraph, src: str, dst: str):
    data = dynamic_graph_to_pyg(g, device=agent.device)
    t0 = time.perf_counter()
    verdict = evaluate_with_reasonableness(
        g, agent, src, dst, k_threshold=K_THRESHOLD,
        max_steps=EVAL_MAX_STEPS, data=data,
    )
    infer_ms = (time.perf_counter() - t0) * 1000
    return verdict, infer_ms


def _ckpt_path(seed: int, scenario_id: str) -> pathlib.Path:
    return AGENTS_DIR / f"warm_seed{seed}_scen{scenario_id}_4x.pt"


def _load_agent(seed: int, scenario_id: str, device: str) -> GNNDQN:
    path = _ckpt_path(seed, scenario_id)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    agent = GNNDQN(
        node_in_dim=ck.get("node_in_dim", 4),
        hidden_dim=ck.get("hidden_dim", TRAIN_CFG["hidden_dim"]),
        seed=seed, device=device,
    )
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None
    return agent


def _save_checkpoint(agent: GNNDQN, seed: int, scenario_id: str) -> pathlib.Path:
    path = _ckpt_path(seed, scenario_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_raw_state_dict": agent._encoder_raw.state_dict(),
        "q_head_state_dict": agent._q_head_raw.state_dict(),
        "hidden_dim": TRAIN_CFG["hidden_dim"],
        "node_in_dim": 4,
        "seed": seed,
        "scenario_id": scenario_id,
        "grid": dict(GRID_CFG),
        "train": {k: v for k, v in TRAIN_CFG.items() if k != "oracle_pool"},
        "kind": "warm",
        "build_version": "v1",
        "training_config": _training_config_record(),
    }, str(path))
    return path


def _training_config_record() -> dict:
    return {
        "n_iterations": TRAIN_CFG["n_iterations"],
        "episodes_per_iteration": TRAIN_CFG["episodes_per_iteration"],
        "grad_steps_per_episode": TRAIN_CFG["grad_steps_per_episode"],
        "batch_size": TRAIN_CFG["batch_size"],
        "hidden_dim": TRAIN_CFG["hidden_dim"],
        "expert_ratio": TRAIN_CFG["expert_ratio"],
        "pre_seed_n_states": TRAIN_CFG["pre_seed_n_states"],
        "pre_seed_k_paths": TRAIN_CFG["pre_seed_k_paths"],
        "total_rollouts": TRAIN_CFG["n_iterations"]
        * TRAIN_CFG["episodes_per_iteration"],
    }


def _failing_subset() -> list[dict]:
    """The cells where the 1x sweep failed (warm_strict=False)."""
    sweep = json.loads(SWEEP_1X_JSON.read_text())
    return [r for r in sweep if not r["warm_strict"]]


# ── Training (full run) ───────────────────────────────────────────────────────

def train_cell(cell: dict, device: str) -> dict:
    from datetime import datetime

    seed, scenario_id = cell["seed"], cell["scenario_id"]
    src, dst = cell["source"], cell["destination"]
    set_global_seed(seed)

    g = DynamicGraph(**GRID_CFG, seed=_grid_seed(scenario_id))
    oracles = build_oracle_pool(
        g.nodes, g.graph, pool=TRAIN_CFG["oracle_pool"], seed=seed,
    )
    agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"],
                   seed=seed, device=device)
    buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(seed),
    )
    print(f"  [warm 4x: training {TRAIN_CFG['n_iterations']} iterations...]",
          flush=True)
    t0 = time.perf_counter()
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, [(src, dst)],
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=seed,
        pre_seed_n_states=TRAIN_CFG["pre_seed_n_states"],
        pre_seed_k_paths=pool_pre_seed_k_paths(
            TRAIN_CFG["oracle_pool"], TRAIN_CFG["pre_seed_k_paths"]
        ),
    )
    train_s = time.perf_counter() - t0
    ckpt = _save_checkpoint(agent, seed, scenario_id)

    verdict, infer_ms = _evaluate(agent, g, src, dst)
    return {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "scenario_id": scenario_id,
        "source": src,
        "destination": dst,
        "grid": dict(GRID_CFG),
        "training_config": _training_config_record(),
        "warm_cost": verdict.cost,
        "warm_strict": verdict.reached_goal_strict,
        "warm_reasonable": verdict.reached_goal_reasonable,
        "warm_cost_ratio": verdict.cost_ratio,
        "dijkstra_cost": verdict.dijkstra_reference_cost,
        "warm_infer_ms": infer_ms,
        "warm_train_s": train_s,
        "k_threshold": K_THRESHOLD,
        "checkpoint": str(ckpt.resolve()),
        "baseline_1x_cost": cell.get("warm_cost"),
        "baseline_1x_strict": cell.get("warm_strict", False),
        "compute_multiplier": COMPUTE_MULTIPLIER,
    }


# ── Eval-only (validation from an existing checkpoint, no training) ──────────

def eval_only(cell_key: str, device: str) -> bool:
    seed_s, scenario_id = cell_key.split("/", 1)
    seed = int(seed_s)

    recorded = None
    if SUBSET_JSON.exists():
        recorded = next((r for r in json.loads(SUBSET_JSON.read_text())
                         if r["seed"] == seed
                         and r["scenario_id"] == scenario_id), None)

    set_global_seed(seed)
    agent = _load_agent(seed, scenario_id, device)
    g = _build_eval_graph(scenario_id)
    if recorded is not None:
        src, dst = recorded["source"], recorded["destination"]
    else:
        sweep = {(r["seed"], r["scenario_id"]): r
                 for r in json.loads(SWEEP_1X_JSON.read_text())}
        cell = sweep[(seed, scenario_id)]
        src, dst = cell["source"], cell["destination"]

    verdict, infer_ms = _evaluate(agent, g, src, dst)
    print(f"\nEval-only  seed={seed}  scenario={scenario_id}  device={device}")
    print(f"  reach={verdict.reached_goal_strict}  cost={verdict.cost!r}  "
          f"dijkstra={verdict.dijkstra_reference_cost!r}  "
          f"ratio={verdict.cost_ratio!r}  infer={infer_ms:.0f}ms")

    if recorded is None:
        print("  (no recorded values to compare against)")
        return True
    rc, rr = recorded["warm_cost"], recorded["warm_strict"]
    rd = recorded["dijkstra_cost"]
    tol = lambda a, b: (a == b) or abs(a - b) <= 1e-6 * max(1.0, abs(b))
    ok_cost = tol(verdict.cost, rc)
    ok_reach = verdict.reached_goal_strict == rr
    ok_dij = tol(verdict.dijkstra_reference_cost, rd)
    print(f"  recorded: reach={rr}  cost={rc!r}  dijkstra={rd!r}")
    print(f"  match: cost={'OK' if ok_cost else 'MISMATCH'}  "
          f"reach={'OK' if ok_reach else 'MISMATCH'}  "
          f"dijkstra={'OK' if ok_dij else 'MISMATCH'}")
    return ok_cost and ok_reach and ok_dij


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-only", metavar="SEED/SCENARIO_ID", default=None,
                    help="Skip training: load the existing checkpoint for this "
                         "cell (e.g. 7/seed2029167941_s3), rebuild the "
                         "iteration-40 eval state, evaluate, and compare to "
                         "the recorded result.")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Results JSON for the training run (the original "
                         "artifact runs/v1_long_train_subset_results.json is "
                         "never overwritten).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.eval_only:
        ok = eval_only(args.eval_only, args.device)
        raise SystemExit(0 if ok else 1)

    cells = _failing_subset()
    print(f"V1 long-train (4x): {len(cells)} cells failing at 1x "
          f"from {SWEEP_1X_JSON}")
    out_path = pathlib.Path(args.out)
    results: list[dict] = []
    for i, cell in enumerate(cells):
        print(f"[{i + 1}/{len(cells)}] seed={cell['seed']} "
              f"scenario={cell['scenario_id']}", flush=True)
        row = train_cell(cell, args.device)
        results.append(row)
        print(f"  warm_strict={row['warm_strict']}  cost={row['warm_cost']}  "
              f"train={row['warm_train_s']:.0f}s", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
        gc.collect()
    print(f"\nSaved {len(results)} cells to {out_path}")


if __name__ == "__main__":
    main()
