"""Pilot: retrain warm+cold for the demo's hero cell (50x50_s1) with the
graph density randomized during training, to see whether that fixes the
warm agent's poor performance at the OOD perturbation levels the demo's
Explore-tab selector exposes (0.5x, 2.0x of the training deactivate_prob).

Background: demo_app/server.py's /reach OOD branch evaluates the existing
checkpoints on a graph rebuilt with deactivate_prob and node_deactivate_prob
scaled by a multiplier the graph was never trained across. Measured in
runs/ood_warm_vs_cold.json, 50x50_s1 (the demo's default cell) totally fails
to reach at 0.5x (500/500 steps) and reaches at 2.0x only at a 52.5x
optimality ratio (vs 1.54x at the training level). This is an ordinary
out-of-distribution generalization gap, not a bug in the graph-rebuild code.

This script trains ONE new warm+cold pair on the identical scenario
(grid_seed, source, destination, all other hyperparameters) from
demo_agents/manifest.json's "50x50" entry, scenario index 1 -- the ONLY
change is deactivate_mult_range=(0.5, 2.0) passed to train_gnn_dqn, so both
arms see graph density resampled every iteration from that range instead of
a single fixed value. Both arms are retrained (not just warm) so the
warm/cold contrast still isolates demonstration availability as the only
variable -- see the density-robustness design note.

Output checkpoints go to demo_agents_ood/, in the same schema
demo_app/server.py::_load_checkpoint expects. demo_agents/ and
demo_agents/manifest.json are never touched, so the paper-reported numbers
and verify_demo_claims.py stay exactly as they are.

Usage:
    uv run python scripts/train_ood_robust_pilot.py
"""
from __future__ import annotations

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
from qwarm.eval.scenario_sampler import Scenario
from qwarm.oracles.pool import build_oracle_pool, pool_pre_seed_k_paths
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.utils.device import resolve_device
from qwarm.utils.seeding import set_global_seed

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "demo_agents" / "manifest.json"
OUT_DIR = ROOT / "demo_agents_ood"

# The demo's UI-exposed OOD levels. Training sees density resampled every
# iteration from base * Uniform(0.5, 2.0), matching exactly what a reviewer
# can select in the Perturbation dropdown.
DEACTIVATE_MULT_RANGE = (0.5, 2.0)

# v1 (manifest's stock 10 iterations x 200 episodes/iter, density resampled
# EVERY iteration) collapsed at 1.0x/1.5x/2.0x -- see
# runs/ood_robust_pilot_v1_failed/. v2 tripled the budget (20 x 300) on a
# budget-starvation hypothesis -- it got WORSE across the board, including
# wiping out v1's one improvement (0.5x) -- see
# runs/ood_robust_pilot_v2_failed/. That result is inconsistent with a
# sample-count shortfall and consistent with per-iteration density switching
# being a moving target the replay buffer/target network can't track (more
# switches in v2 -> more forgetting).
#
# v3 tests that directly: same budget as v1 (no more GPU cost), but density is
# held fixed for a BLOCK of consecutive iterations before switching, instead
# of every iteration. block_size=5 over 10 iterations gives exactly 2 blocks
# (1000 episodes of sustained exposure per density draw), maximizing
# per-density dwell time within the unchanged budget.
BUDGET_OVERRIDE: dict = {}
DENSITY_BLOCK_SIZE = 5

SCENARIO_INDEX = 1  # 50x50_s1 -- the demo's default/hero cell
SEED = 191664964  # scenario's grid_seed, reused as the training seed too,
                   # matching how the original 50x50.pt / 50x50_s1.pt were produced


def _train_one(arm: str, cfg: dict, scenario: Scenario, oracle_pool: str,
                device: str) -> GNNDQN:
    """Train a single agent. arm='cold' means an empty oracle set, ratio 0."""
    set_global_seed(SEED)
    grid, tr = cfg["grid"], cfg["train"]
    g = DynamicGraph(seed=scenario.grid_seed, **grid)

    warm = arm == "warm"
    oracles = build_oracle_pool(
        g.nodes, g.graph, pool=oracle_pool, seed=SEED
    ) if warm else []

    agent = GNNDQN(node_in_dim=4, hidden_dim=tr["hidden_dim"],
                   device=device, seed=SEED)
    buf = ExpertReplayBuffer(
        expert_ratio=tr["expert_ratio"] if warm else 0.0,
        rng=np.random.default_rng(SEED),
    )
    kwargs = dict(
        n_iterations=tr["n_iterations"],
        episodes_per_iteration=tr["episodes_per_iteration"],
        grad_steps_per_episode=tr["grad_steps_per_episode"],
        batch_size=tr["batch_size"],
        re_seed_experts_each_iteration=warm,
        seed=SEED,
        deactivate_mult_range=DEACTIVATE_MULT_RANGE,
        density_block_size=DENSITY_BLOCK_SIZE,
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


def save_ckpt(agent: GNNDQN, path: pathlib.Path, cfg: dict,
              scenario_id: str, kind: str) -> None:
    """Same schema demo_agents/*.pt use, loadable through _load_checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_raw_state_dict": agent._encoder_raw.state_dict(),
        "q_head_state_dict": agent._q_head_raw.state_dict(),
        "hidden_dim": cfg["train"]["hidden_dim"],
        "node_in_dim": 4,
        "seed": SEED,
        "scenario_id": scenario_id,
        "grid": cfg["grid"],
        "train": cfg["train"],
        "kind": kind,
        "deactivate_mult_range": list(DEACTIVATE_MULT_RANGE),
        "density_block_size": DENSITY_BLOCK_SIZE,
        "build_version": "ood-robust-pilot-50x50_s1-v3-blocked-density",
    }, path)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    entry = manifest["50x50"]
    scenario_row = entry["scenarios"][SCENARIO_INDEX]
    scenario = Scenario(
        grid_seed=scenario_row["grid_seed"],
        source_node=scenario_row["source"],
        destination_node=scenario_row["destination"],
        euclidean_distance=scenario_row["euclidean_distance"],
        scenario_id=scenario_row["scenario_id"],
    )
    cfg = {"grid": entry["grid"], "train": {**entry["train"], **BUDGET_OVERRIDE}}
    device = resolve_device("auto")

    print(f"Pilot: {scenario.scenario_id} ({scenario.source_node} -> "
          f"{scenario.destination_node}), device={device}, "
          f"deactivate_mult_range={DEACTIVATE_MULT_RANGE}, "
          f"density_block_size={DENSITY_BLOCK_SIZE}, "
          f"budget_override={BUDGET_OVERRIDE}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ("warm", "cold"):
        t0 = time.perf_counter()
        print(f"[{arm}] training...", flush=True)
        agent = _train_one(arm, cfg, scenario, oracle_pool="full", device=device)
        elapsed = time.perf_counter() - t0
        out = OUT_DIR / f"{arm}_50x50_s1.pt"
        save_ckpt(agent, out, cfg, scenario.scenario_id, arm)
        print(f"[{arm}] done in {elapsed:.0f}s, saved {out}", flush=True)

    print(f"\nPilot checkpoints in {OUT_DIR}/")


if __name__ == "__main__":
    main()
