"""Fleet-mode (multi-query) training: ONE agent serving MANY queries.

WHY THIS EXPERIMENT EXISTS
--------------------------
Every agent in the thesis so far is trained on a single (source, destination)
pair and evaluated on that same pair. That is upstream of three weak results:

  * the transfer benchmark reaches 1-3/10 and its OOD hypothesis is null --
    §4.8 already attributes this to the per-cell single-query regime;
  * the V3 similarity gate saturates, which a single-goal representation makes
    far more likely;
  * the fleet benchmark evaluates one agent on 999 queries it was never
    trained for, so its throughput number is measured on an agent that cannot
    actually solve them.

It also undercuts the thesis's own framing: a goal-conditioned, inductive,
permutation-equivariant encoder is motivated by generalisation, then trained
on one goal.

This script trains warm and cold agents on N_train queries and evaluates BOTH
on the training queries and on a disjoint held-out set, so "does the warm-start
advantage survive generalisation across goals?" becomes a measurable question.

WHAT IT PRODUCES
----------------
  agents/  fleet_{arm}_{scale}_seed{seed}.pt        final checkpoints
           snapshots/fleet_{arm}_..._it{k}.pt       per-iteration encoders
  manifest_fleet_seed{seed}_{scale}.json            readable by the geometry
                                                    scripts (--manifest)
  fleet_results_seed{seed}_{scale}.json             per-query eval + train logs

The snapshots are the point of the callback: they let the geometry panel be
plotted against goal-reach on one time axis, which is the only way to show that
representation quality degrades *while the reward curve is still rising*.

WHAT IT WILL NOT DO
-------------------
It will not beat Dijkstra or A* on path quality. Under full observability with
known edge weights on a graph that is static within a trajectory, an exact
shortest-path algorithm is optimal by construction. The claim this experiment
supports is about sample efficiency and cross-goal generalisation, not about
displacing exact search.

RUN
---
    python scripts/run_fleet_mode.py --seed 42 --scale 25x25

See --help for the full grid. Start at 25x25; it is the only scale where the
whole warm/cold x seed grid is comfortably affordable on one GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.oracles.pool import build_oracle_pool, pool_pre_seed_k_paths
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.utils.seeding import set_global_seed

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Fleet presets. Episode counts are deliberately LOWER than the single-query
# configs in demo_agents/manifest.json because every episode now rolls out
# |Q| queries instead of one: total rollouts per iteration scale with the
# number of training queries, so holding episodes_per_iteration fixed would
# multiply wall-clock by |Q|.
SCALES: dict[str, dict] = {
    "25x25": {
        "grid": {"grid_width": 25, "grid_height": 25, "extra_edges": 2,
                 "deactivate_prob": 0.15, "node_deactivate_prob": 0.05},
        "train": {"n_iterations": 8, "episodes_per_iteration": 25,
                  "grad_steps_per_episode": 4, "batch_size": 64,
                  "hidden_dim": 64, "expert_ratio": 0.40,
                  "pre_seed_n_states": 3, "pre_seed_k_paths": 5},
    },
    "50x50": {
        "grid": {"grid_width": 50, "grid_height": 50, "extra_edges": 3,
                 "deactivate_prob": 0.22, "node_deactivate_prob": 0.05},
        "train": {"n_iterations": 10, "episodes_per_iteration": 20,
                  "grad_steps_per_episode": 4, "batch_size": 64,
                  "hidden_dim": 128, "expert_ratio": 0.30,
                  "pre_seed_n_states": 2, "pre_seed_k_paths": 3},
    },
    "100x100": {
        "grid": {"grid_width": 100, "grid_height": 100, "extra_edges": 4,
                 "deactivate_prob": 0.30, "node_deactivate_prob": 0.05},
        "train": {"n_iterations": 10, "episodes_per_iteration": 20,
                  "grad_steps_per_episode": 4, "batch_size": 64,
                  "hidden_dim": 128, "expert_ratio": 0.30,
                  "pre_seed_n_states": 2, "pre_seed_k_paths": 3},
    },
}


def save_ckpt(agent: GNNDQN, path: pathlib.Path, cfg: dict, seed: int,
              scenario_id: str, kind: str) -> None:
    """Same schema as demo_agents/*.pt so the geometry scripts load new and old
    checkpoints through one code path. `goal_relative` is stamped because it
    changes the Q-head's input width: loading a goal-relative checkpoint into a
    default head is a shape error, and the reverse would silently mis-slice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_raw_state_dict": agent._encoder_raw.state_dict(),
        "q_head_state_dict": agent._q_head_raw.state_dict(),
        "hidden_dim": cfg["train"]["hidden_dim"],
        "node_in_dim": 4,
        "goal_relative": bool(agent._q_head_raw.goal_relative),
        "seed": seed,
        "scenario_id": scenario_id,
        "grid": cfg["grid"],
        "train": cfg["train"],
        "kind": kind,
        "build_version": "fleet-mode",
    }, path)


def train_arm(arm: str, seed: int, cfg: dict, grid_seed: int,
              train_queries: list[tuple[str, str]], oracle_pool: str,
              device: str, snap_dir: pathlib.Path | None,
              snapshot_every: int, tag: str, her_k: int = 0,
              her_strategy: str = "future",
              goal_relative: bool = False,
              mask_visited: bool = False) -> tuple[GNNDQN, DynamicGraph, dict]:
    """Train one arm on ALL training queries. arm='cold' -> no oracles, ratio 0."""
    set_global_seed(seed)
    tr = cfg["train"]
    g = DynamicGraph(seed=grid_seed, **cfg["grid"])

    warm = arm == "warm"
    oracles = build_oracle_pool(
        g.nodes, g.graph, pool=oracle_pool, seed=seed
    ) if warm else []

    agent = GNNDQN(node_in_dim=4, hidden_dim=tr["hidden_dim"],
                   device=device, seed=seed, goal_relative=goal_relative)
    buf = ExpertReplayBuffer(
        expert_ratio=tr["expert_ratio"] if warm else 0.0,
        rng=np.random.default_rng(seed),
    )

    cb = None
    if snap_dir is not None and snapshot_every > 0:
        def cb(iteration, ag, dg, logs, _c=cfg, _s=seed, _t=tag, _a=arm):
            if (iteration + 1) % snapshot_every and iteration != 0:
                return
            save_ckpt(ag, snap_dir / f"{_t}_{_a}_it{iteration:03d}.pt",
                      _c, _s, f"{_t}_it{iteration:03d}", f"{_a}_snapshot")

    kwargs: dict = dict(
        env_kwargs={"mask_visited": mask_visited},
        n_iterations=tr["n_iterations"],
        episodes_per_iteration=tr["episodes_per_iteration"],
        grad_steps_per_episode=tr["grad_steps_per_episode"],
        batch_size=tr["batch_size"],
        re_seed_experts_each_iteration=warm,
        seed=seed,
        iteration_callback=cb,
        her_k=her_k,
        her_strategy=her_strategy,
    )
    if warm:
        kwargs["pre_seed_n_states"] = tr["pre_seed_n_states"]
        kwargs["pre_seed_k_paths"] = pool_pre_seed_k_paths(
            oracle_pool, tr["pre_seed_k_paths"]
        )

    logs = train_gnn_dqn(g, PathfindingEnv, agent, buf, oracles,
                         train_queries, **kwargs)
    return agent, g, logs


def evaluate(agent: GNNDQN, g: DynamicGraph, queries: list[tuple[str, str]],
             k_threshold: float) -> list[dict]:
    data = dynamic_graph_to_pyg(g, device=agent.device)
    rows = []
    for src, dst in queries:
        t0 = time.perf_counter()
        v = evaluate_with_reasonableness(
            g, agent, src, dst, k_threshold=k_threshold, data=data
        )
        rows.append({
            "source": src, "destination": dst,
            "cost": None if v.cost == float("inf") else float(v.cost),
            "reached": bool(v.reached_goal_strict),
            "reasonable": bool(v.reached_goal_reasonable),
            "cost_ratio": v.cost_ratio,
            "dijkstra_cost": v.dijkstra_reference_cost,
            "eval_ms": (time.perf_counter() - t0) * 1000,
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    reached = [r for r in rows if r["reached"]]
    ratios = [r["cost_ratio"] for r in reached
              if r["cost_ratio"] is not None and np.isfinite(r["cost_ratio"])]
    return {
        "n": n,
        "reach": len(reached),
        "reach_rate": len(reached) / n if n else 0.0,
        "reasonable_rate": sum(r["reasonable"] for r in rows) / n if n else 0.0,
        "mean_cost_ratio": float(np.mean(ratios)) if ratios else None,
        "median_cost_ratio": float(np.median(ratios)) if ratios else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fleet-mode (multi-query) warm-vs-cold training.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scale", choices=sorted(SCALES), default="25x25")
    ap.add_argument("--n-train-queries", type=int, default=20)
    ap.add_argument("--n-eval-queries", type=int, default=20,
                    help="held-out queries, disjoint from the training set")
    ap.add_argument("--oracle-pool", default="classical_only",
                    choices=["full", "classical_only", "quantum_only"],
                    help="classical_only skips the CPU-bound QAOA solver; the "
                         "demonstration-source ablation found goal-reach "
                         "invariant to provenance, so it is the default here")
    ap.add_argument("--arms", nargs="+", default=["warm", "cold"],
                    choices=["warm", "cold"])
    ap.add_argument("--k-threshold", type=float, default=3.0)
    ap.add_argument("--snapshot-every", type=int, default=1,
                    help="save an encoder snapshot every N iterations; 0 disables")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override n_iterations. Use this for the training-budget "
                         "control: 'does not generalise across goals' and "
                         "'undertrained' predict the same held-out result, and "
                         "only a budget sweep separates them")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override episodes_per_iteration")
    ap.add_argument("--her-k", type=int, default=0,
                    help="Hindsight Experience Replay goals per transition. "
                         "0 = off (baseline). 4 is the usual choice. Relabels "
                         "trajectories against goals the agent actually reached, "
                         "so training is no longer confined to the assigned "
                         "destinations")
    ap.add_argument("--her-strategy", default="future", choices=["future", "final"])
    ap.add_argument("--goal-relative", action="store_true",
                    help="add (h_act - h_goal) and (h_cur - h_goal) to the "
                         "Q-head input, giving it the goal RELATION rather than "
                         "requiring it to be inferred from concatenation. Stays "
                         "in the head, so the encoder is still computed once "
                         "per graph")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the run tag, so variants do not "
                         "overwrite each other's checkpoints or manifests")
    ap.add_argument("--mask-visited", action="store_true",
                    help="Exclude visited nodes from the TRAINING action "
                         "set (the MDP of Chapter 3). Off by default so "
                         "existing fleet artefacts stay reproducible.")
    ap.add_argument("--out-dir", default="runs/fleet")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(SCALES[args.scale]))  # deep copy before override
    if args.iterations is not None:
        cfg["train"]["n_iterations"] = args.iterations
    if args.episodes is not None:
        cfg["train"]["episodes_per_iteration"] = args.episodes
    out_dir = pathlib.Path(args.out_dir)
    agents_dir = out_dir / "agents"
    snap_dir = agents_dir / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample train + held-out queries from ONE topology, so generalisation is
    # measured across goals rather than across graphs.
    rng = np.random.default_rng(args.seed)
    total = args.n_train_queries + args.n_eval_queries
    scenarios = sample_scenarios(cfg["grid"], n_scenarios=total, rng=rng,
                                 min_euclidean_fraction=0.6)
    grid_seed = scenarios[0].grid_seed
    train_sc = scenarios[:args.n_train_queries]
    eval_sc = scenarios[args.n_train_queries:]
    train_q = [(s.source_node, s.destination_node) for s in train_sc]
    eval_q = [(s.source_node, s.destination_node) for s in eval_sc]
    tag = f"fleet_{args.scale}_seed{args.seed}{args.tag_suffix}"

    print(f"=== fleet-mode {args.scale} seed={args.seed} "
          f"grid_seed={grid_seed} ===")
    print(f"    train queries: {len(train_q)}   held-out: {len(eval_q)}")
    print(f"    oracle pool  : {args.oracle_pool}")
    print(f"    iterations   : {cfg['train']['n_iterations']} x "
          f"{cfg['train']['episodes_per_iteration']} episodes x "
          f"{len(train_q)} queries")
    print(f"    HER          : k={args.her_k} ({args.her_strategy})"
          f"{'  [OFF]' if args.her_k == 0 else ''}")
    print(f"    goal-relative: {args.goal_relative}")
    print(f"    snapshots    : every {args.snapshot_every} iter -> {snap_dir}")

    results: dict = {
        "scale": args.scale, "seed": args.seed, "grid_seed": grid_seed,
        "oracle_pool": args.oracle_pool,
        "n_train_queries": len(train_q), "n_eval_queries": len(eval_q),
        "ablation": {
            "her_k": args.her_k,
            "her_strategy": args.her_strategy,
            "goal_relative": args.goal_relative,
            "iterations": cfg["train"]["n_iterations"],
            "episodes_per_iteration": cfg["train"]["episodes_per_iteration"],
        },
        "config": cfg, "arms": {},
    }
    manifest_scenarios: list[dict] = []

    for arm in args.arms:
        print(f"\n--- {arm} ---", flush=True)
        t0 = time.perf_counter()
        agent, g, logs = train_arm(
            arm, args.seed, cfg, grid_seed, train_q, args.oracle_pool,
            args.device, snap_dir if args.snapshot_every else None,
            args.snapshot_every, tag,
            her_k=args.her_k, her_strategy=args.her_strategy,
            goal_relative=args.goal_relative,
            mask_visited=args.mask_visited,
        )
        train_s = time.perf_counter() - t0
        print(f"    trained in {train_s/60:.1f} min", flush=True)

        ckpt = agents_dir / f"{tag}_{arm}.pt"
        save_ckpt(agent, ckpt, cfg, args.seed, tag, arm)

        train_rows = evaluate(agent, g, train_q, args.k_threshold)
        held_rows = evaluate(agent, g, eval_q, args.k_threshold)
        s_tr, s_he = summarise(train_rows), summarise(held_rows)

        print(f"    train  queries: reach {s_tr['reach']}/{s_tr['n']} "
              f"({s_tr['reach_rate']:.0%})  "
              f"median ratio {s_tr['median_cost_ratio']}")
        print(f"    HELD-OUT      : reach {s_he['reach']}/{s_he['n']} "
              f"({s_he['reach_rate']:.0%})  "
              f"median ratio {s_he['median_cost_ratio']}")

        results["arms"][arm] = {
            "checkpoint": ckpt.name, "train_seconds": train_s,
            "training_log": logs,
            "train_queries": {"summary": s_tr, "rows": train_rows},
            "held_out": {"summary": s_he, "rows": held_rows},
        }

    # Manifest for the geometry scripts. One "scenario" per arm-pair; the
    # geometry audit rebuilds the graph from grid/train + grid_seed, and the
    # fleet n_iterations here is what determines the evaluation state.
    manifest_scenarios.append({
        "scenario_id": tag,
        "source": train_q[0][0], "destination": train_q[0][1],
        "grid_seed": grid_seed,
        **{a: f"{tag}_{a}.pt" for a in args.arms},
    })
    manifest = {
        f"{args.scale}@fleet_seed{args.seed}": {
            "grid": cfg["grid"], "train": cfg["train"],
            "hidden_dim": cfg["train"]["hidden_dim"], "node_in_dim": 4,
            "grid_seed": grid_seed, "sweep_seed": args.seed,
            "oracle_pool": args.oracle_pool, "mode": "fleet",
            "scenarios": manifest_scenarios,
        }
    }
    (agents_dir / f"manifest_{tag}.json").parent.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))

    res_path = out_dir / f"fleet_results_seed{args.seed}_{args.scale}{args.tag_suffix}.json"
    res_path.write_text(json.dumps(results, indent=2))

    print(f"\nWrote {res_path}")
    print(f"Wrote {agents_dir / f'manifest_{tag}.json'}")
    if "warm" in results["arms"] and "cold" in results["arms"]:
        w = results["arms"]["warm"]["held_out"]["summary"]
        c = results["arms"]["cold"]["held_out"]["summary"]
        print(f"\nHELD-OUT warm vs cold: {w['reach']}/{w['n']} vs "
              f"{c['reach']}/{c['n']}  "
              f"({w['reach_rate']:.0%} vs {c['reach_rate']:.0%})")


if __name__ == "__main__":
    main()
