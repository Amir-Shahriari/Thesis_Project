"""Train warm + cold agents for ONE seed at ONE scale and save checkpoints.

This is the unit of work for the multi-seed geometry study: each PBS array task
runs one (seed, scale) pair, so N seeds x M scales fan out across the cluster
with no shared state. Each task writes its own manifest fragment; merge them
with --merge once the array completes.

The released demo_agents/ checkpoints cover a single grid topology
(grid_seed=191664964). The geometry results are therefore n=1 in the dimension
a reviewer cares about most. This script exists to fix that.

Examples
--------
    # one cell (what an array task runs)
    python scripts/train_checkpoints.py --seed 42 --scale 25x25 \
        --agents-dir runs/agents --manifest runs/agents/manifest_seed42_25x25.json

    # classical-only oracle: skips the CPU-bound QAOA solver entirely
    python scripts/train_checkpoints.py --seed 42 --scale 25x25 --oracle-pool classical_only

    # merge fragments into one manifest the geometry scripts can read
    python scripts/train_checkpoints.py --merge runs/agents/manifest_*.json \
        --manifest runs/agents/manifest.json
"""
from __future__ import annotations

import argparse
import glob
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
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.oracles.pool import build_oracle_pool, pool_pre_seed_k_paths
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.utils.seeding import set_global_seed

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Scale presets mirror demo_agents/manifest.json so newly trained checkpoints
# are directly comparable with the released ones.
SCALES: dict[str, dict] = {
    "25x25": {
        "grid": {"grid_width": 25, "grid_height": 25, "extra_edges": 2,
                 "deactivate_prob": 0.15, "node_deactivate_prob": 0.05},
        "train": {"n_iterations": 5, "episodes_per_iteration": 100,
                  "grad_steps_per_episode": 4, "batch_size": 64,
                  "hidden_dim": 64, "expert_ratio": 0.40,
                  "pre_seed_n_states": 5, "pre_seed_k_paths": 10},
    },
    "50x50": {
        "grid": {"grid_width": 50, "grid_height": 50, "extra_edges": 3,
                 "deactivate_prob": 0.22, "node_deactivate_prob": 0.05},
        "train": {"n_iterations": 10, "episodes_per_iteration": 200,
                  "grad_steps_per_episode": 4, "batch_size": 64,
                  "hidden_dim": 128, "expert_ratio": 0.30,
                  "pre_seed_n_states": 3, "pre_seed_k_paths": 3},
    },
    "100x100": {
        "grid": {"grid_width": 100, "grid_height": 100, "extra_edges": 4,
                 "deactivate_prob": 0.30, "node_deactivate_prob": 0.05},
        "train": {"n_iterations": 10, "episodes_per_iteration": 200,
                  "grad_steps_per_episode": 4, "batch_size": 64,
                  "hidden_dim": 128, "expert_ratio": 0.30,
                  "pre_seed_n_states": 3, "pre_seed_k_paths": 3},
    },
}


def _train_one(arm: str, seed: int, scenario, cfg: dict, oracle_pool: str,
               device: str) -> GNNDQN:
    """Train a single agent. arm='cold' means an empty oracle set, ratio 0."""
    set_global_seed(seed)
    grid, tr = cfg["grid"], cfg["train"]
    g = DynamicGraph(seed=scenario.grid_seed, **grid)

    warm = arm == "warm"
    oracles = build_oracle_pool(
        g.nodes, g.graph, pool=oracle_pool, seed=seed
    ) if warm else []

    agent = GNNDQN(node_in_dim=4, hidden_dim=tr["hidden_dim"],
                   device=device, seed=seed)
    buf = ExpertReplayBuffer(
        expert_ratio=tr["expert_ratio"] if warm else 0.0,
        rng=np.random.default_rng(seed),
    )
    kwargs = dict(
        n_iterations=tr["n_iterations"],
        episodes_per_iteration=tr["episodes_per_iteration"],
        grad_steps_per_episode=tr["grad_steps_per_episode"],
        batch_size=tr["batch_size"],
        re_seed_experts_each_iteration=warm,
        seed=seed,
    )
    if warm:
        kwargs["pre_seed_n_states"] = tr["pre_seed_n_states"]
        kwargs["pre_seed_k_paths"] = pool_pre_seed_k_paths(
            oracle_pool, tr["pre_seed_k_paths"]
        )
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles,
        [(scenario.source_node, scenario.destination_node)], **kwargs
    )
    return agent


def save_ckpt(agent: GNNDQN, path: pathlib.Path, cfg: dict, seed: int,
              scenario_id: str, kind: str) -> None:
    """Write in the same schema the released demo_agents/*.pt use, so the
    geometry scripts load new and old checkpoints through one code path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_raw_state_dict": agent._encoder_raw.state_dict(),
        "q_head_state_dict": agent._q_head_raw.state_dict(),
        "hidden_dim": cfg["train"]["hidden_dim"],
        "node_in_dim": 4,
        "seed": seed,
        "scenario_id": scenario_id,
        "grid": cfg["grid"],
        "train": cfg["train"],
        "kind": kind,
        "build_version": "multiseed-geometry-study",
    }, path)


def run(seed: int, scale: str, n_scenarios: int, oracle_pool: str,
        agents_dir: pathlib.Path, manifest_path: pathlib.Path,
        device: str, skip_existing: bool) -> None:
    cfg = SCALES[scale]
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(cfg["grid"], n_scenarios=n_scenarios,
                                 rng=rng, min_euclidean_fraction=0.6)
    grid_seed = scenarios[0].grid_seed
    entry = {
        "grid": cfg["grid"],
        "train": cfg["train"],
        "hidden_dim": cfg["train"]["hidden_dim"],
        "node_in_dim": 4,
        "grid_seed": grid_seed,
        "sweep_seed": seed,
        "oracle_pool": oracle_pool,
        "scenarios": [],
    }

    for i, sc in enumerate(scenarios):
        row = {
            "scenario_id": sc.scenario_id,
            "source": sc.source_node,
            "destination": sc.destination_node,
            "euclidean_distance": sc.euclidean_distance,
            "grid_seed": sc.grid_seed,
        }
        for arm in ("warm", "cold"):
            name = f"{arm}_{scale}_seed{seed}_s{i}.pt"
            out = agents_dir / name
            row[arm] = name
            if skip_existing and out.exists():
                print(f"  [{scale} seed={seed} s{i}/{arm}] exists, skipped",
                      flush=True)
                continue
            t0 = time.perf_counter()
            print(f"  [{scale} seed={seed} s{i}/{arm}] training...", flush=True)
            agent = _train_one(arm, seed, sc, cfg, oracle_pool, device)
            save_ckpt(agent, out, cfg, seed, sc.scenario_id, arm)
            row[f"{arm}_train_s"] = round(time.perf_counter() - t0, 1)
            print(f"      saved {name} ({row[f'{arm}_train_s']:.0f}s)", flush=True)
        entry["scenarios"].append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({scale: entry}, indent=2))
    print(f"\nWrote manifest fragment {manifest_path}")


def merge(patterns: list[str], out_path: pathlib.Path) -> None:
    """Merge per-task fragments. Scales with several seeds are keyed
    '<scale>@seed<N>' so every topology stays a distinct manifest entry."""
    merged: dict = {}
    files: list[str] = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    for f in files:
        frag = json.loads(pathlib.Path(f).read_text())
        for scale, entry in frag.items():
            key = scale
            if key in merged:
                key = f"{scale}@seed{entry.get('sweep_seed', 'x')}"
            suffix = 2
            while key in merged:
                key = f"{scale}@seed{entry.get('sweep_seed', 'x')}_{suffix}"
                suffix += 1
            merged[key] = entry
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(files)} fragments -> {out_path} ({len(merged)} entries)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scale", choices=sorted(SCALES), default="25x25")
    ap.add_argument("--n-scenarios", type=int, default=5)
    ap.add_argument("--oracle-pool", default="full",
                    choices=["full", "classical_only", "quantum_only"])
    ap.add_argument("--agents-dir", default="runs/agents")
    ap.add_argument("--manifest", default=None,
                    help="output manifest path (default derived from seed/scale)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: leave already-saved checkpoints alone")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge these manifest fragments into --manifest and exit")
    args = ap.parse_args()

    agents_dir = pathlib.Path(args.agents_dir)
    if args.merge:
        out = pathlib.Path(args.manifest or (agents_dir / "manifest.json"))
        merge(args.merge, out)
        return

    if args.seed is None:
        ap.error("--seed is required unless --merge is given")
    manifest = pathlib.Path(
        args.manifest
        or (agents_dir / f"manifest_seed{args.seed}_{args.scale}.json")
    )
    run(args.seed, args.scale, args.n_scenarios, args.oracle_pool,
        agents_dir, manifest, args.device, args.skip_existing)


if __name__ == "__main__":
    main()
