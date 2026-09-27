"""Persistence ablation at 50x50: does keeping demonstrations in the replay
stream matter, or would supplying them once, at the start, do as well?

Pre-registered in the thesis repository:
    thesis-sections/PREREG-persistence-ablation.md
Design, tests and the reading of every outcome are fixed there. Do not change
the settings below after results exist.

Arms (all trained and evaluated inside every cell, on one recorded stack):
  persistent       demos pre-seeded, kept at rho=0.30, refreshed each iteration
                   (identical call to the thesis warm agent)
  dqfd_pretrain    same pre-seeded pool; N_PRE expert-only steps under the
                   agent's own TD + DQfD-margin loss; pool REMOVED; RL at rho=0
  bc_pretrain      same pre-seeded pool; N_PRE supervised steps (cross-entropy on
                   the expert action over active neighbours); pool REMOVED; RL
  cold             no demonstrations (identical call to the thesis cold agent)
  keep_no_refresh  optional: pre-seeded pool kept at rho=0.30, never refreshed

This script wraps train_gnn_dqn and imports the 50x50 sweep's grid, training
config, oracles, scenario sampling and evaluation, so it adds arms without
touching the training loop.

Environment:
  QWARM_ARMS         comma list (default persistent,dqfd_pretrain,bc_pretrain,cold)
  QWARM_N_PRETRAIN   pretraining gradient steps (default 1000, pre-registered)
  QWARM_SEEDS        subset of seeds, for sharding across processes
  QWARM_OUT_DIR      output dir (default runs/persistence_ablation_50x50)
  QWARM_SMOKE=1      code-path smoke test only: 1 cell, 1 iteration, 10
                     episodes, 20 pretraining steps. NOT evidence.
  QWARM_ALLOW_CPU=1  permit a non-smoke run without CUDA (default: refuse)

Usage:
  uv run python scripts/run_persistence_ablation.py            # full run
  uv run python scripts/run_persistence_ablation.py --aggregate  # stats only
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time

# The masked action space is part of the pre-registered design; the base
# module reads this flag at import time, so it must be set first.
os.environ["QWARM_MASK_VISITED"] = "1"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

import run_sweep_50x50 as base  # noqa: E402
from qwarm.agents.gnn_dqn import GNNDQN  # noqa: E402
from qwarm.env.dynamic_graph import DynamicGraph  # noqa: E402
from qwarm.env.pathfinding_env import PathfindingEnv  # noqa: E402
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg  # noqa: E402
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer, Transition  # noqa: E402
from qwarm.training.expert_seeding import (  # noqa: E402
    discover_all_paths,
    seed_buffer_from_path_library,
)
from qwarm.training.train_gnn_dqn import train_gnn_dqn  # noqa: E402
from qwarm.utils.seeding import set_global_seed  # noqa: E402

assert base.MASK_VISITED, "masked action space is required by the pre-registration"

SMOKE = os.environ.get("QWARM_SMOKE", "") == "1"
ALL_ARMS = ["persistent", "dqfd_pretrain", "bc_pretrain", "cold", "keep_no_refresh"]
ARMS = [a.strip() for a in os.environ.get(
    "QWARM_ARMS", "persistent,dqfd_pretrain,bc_pretrain,cold").split(",") if a.strip()]
for _a in ARMS:
    assert _a in ALL_ARMS, f"unknown arm {_a}"

N_PRE = int(os.environ.get("QWARM_N_PRETRAIN", "20" if SMOKE else "1000"))
N_ITER = 1 if SMOKE else base.TRAIN_CFG["n_iterations_1x"]
EPISODES = 10 if SMOKE else base.TRAIN_CFG["episodes_per_iteration"]
SEEDS = base.SEEDS[:1] if SMOKE else base.SEEDS
N_SCEN = 1 if SMOKE else base.N_SCENARIOS
OUT_DIR = pathlib.Path(os.environ.get(
    "QWARM_OUT_DIR",
    "runs/persistence_ablation_smoke" if SMOKE else "runs/persistence_ablation_50x50"))
BUDGETS = base.EVAL_BUDGETS
TC = base.TRAIN_CFG


# ── Provenance ────────────────────────────────────────────────────────────────

def _git(*args: str) -> "str | None":
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_provenance() -> dict:
    """Stack record, as scripts/analysis/record_provenance.py, plus the dirty
    state of the working tree, which that script does not capture."""
    dev, name, cap = "cpu", None, None
    if torch.cuda.is_available():
        dev = "cuda:0"
        name = torch.cuda.get_device_name(0)
        cap = list(torch.cuda.get_device_capability(0))
    diff = _git("diff", "HEAD") or ""
    rec = {
        "label": "50x50 persistence ablation" + (" (SMOKE)" if SMOKE else ""),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "cuda_available": torch.cuda.is_available(),
        "resolved_device": dev,
        "gpu_name": name,
        "compute_capability": cap,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python": platform.python_version(),
        "git_rev": _git("rev-parse", "--short", "HEAD"),
        "git_dirty_files": [l.split(maxsplit=1)[1]
                            for l in (_git("status", "--porcelain", "--untracked-files=no") or "").splitlines()
                            if len(l.split(maxsplit=1)) == 2],
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "arms": ARMS,
        "n_pretrain": N_PRE,
        "n_iterations": N_ITER,
        "episodes_per_iteration": EPISODES,
        "smoke": SMOKE,
        "env": {k: os.environ.get(k) for k in (
            "QWARM_MASK_VISITED", "QWARM_SEEDS", "QWARM_ARMS", "QWARM_N_PRETRAIN",
            "QWARM_OUT_DIR", "QWARM_SMOKE", "CUDA_VISIBLE_DEVICES")},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(s) for s in SEEDS)
    with open(OUT_DIR / f"provenance_seeds_{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=2)
    return rec


# ── Shared construction ───────────────────────────────────────────────────────

def _agent(seed: int) -> GNNDQN:
    return GNNDQN(node_in_dim=4, hidden_dim=TC["hidden_dim"], gamma=TC["gamma"], seed=seed)


def _preseed(g: DynamicGraph, buf: ExpertReplayBuffer, oracles: list,
             queries: list[tuple[str, str]]) -> int:
    """Exactly the pre-population block of train_gnn_dqn (library + goal-adjacent)."""
    lib = discover_all_paths(
        g, oracles, queries,
        n_perturbation_states=TC["pre_seed_n_states"],
        n_shortest_paths=TC["pre_seed_k_paths"],
        base_iteration=0, gamma=TC["gamma"],
    )
    seed_buffer_from_path_library(g, buf, lib, iteration=0)
    for src, dst in queries:
        for node_id, nbrs in g.graph.items():
            if dst not in nbrs:
                continue
            e = nbrs[dst]
            if not (e["active"] and g.nodes[node_id]["active"] and g.nodes[dst]["active"]):
                continue
            step_cost = e["distance"] + 0.1 * e["time"] + g.nodes[dst]["node_penalty"]
            buf.expert_pool.append(Transition(
                state_node=node_id, action_node=dst, reward=-step_cost + 100.0,
                next_state_node=dst, done=True, valid_next_actions=[],
                is_expert=True, iteration_added=0, goal_node=dst,
            ))
    return len(buf.expert_pool)


def _expert_batch(buf: ExpertReplayBuffer, rng: np.random.Generator, n: int) -> list:
    pool = list(buf.expert_pool)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def pretrain_dqfd(agent: GNNDQN, g: DynamicGraph, buf: ExpertReplayBuffer,
                  n_steps: int, seed: int) -> dict:
    """DQfD pretraining phase: expert-only batches under the agent's own loss."""
    rng = np.random.default_rng(seed + 4211)
    data = dynamic_graph_to_pyg(g, device=agent.device)
    agent.encode(data)
    losses = []
    for _ in range(n_steps):
        batch = _expert_batch(buf, rng, TC["batch_size"])
        losses.append(agent.learn_from_batch(batch, data, goal_node=batch[0].goal_node))
    agent.update_target()
    return {"pretrain_steps": n_steps, "pretrain_final_loss": float(np.mean(losses[-50:]))}


def pretrain_bc(agent: GNNDQN, g: DynamicGraph, buf: ExpertReplayBuffer,
                n_steps: int, seed: int) -> dict:
    """Behaviour cloning: cross-entropy of the expert action over the active
    neighbours, the same candidate set the DQfD margin term uses."""
    rng = np.random.default_rng(seed + 4211)
    data = dynamic_graph_to_pyg(g, device=agent.device)
    x = data.x.to(agent.device)
    ei = data.edge_index.to(agent.device)
    idx = data.node_id_to_idx
    params = list(agent._encoder_raw.parameters()) + list(agent._q_head_raw.parameters())
    losses, used, skipped = [], 0, 0
    for _ in range(n_steps):
        agent._encoder_raw.train()
        agent._q_head_raw.train()
        batch = _expert_batch(buf, rng, TC["batch_size"])
        emb = agent._encoder_raw(x, ei)
        terms = []
        for t in batch:
            g_node = t.goal_node
            if (t.state_node not in idx or t.action_node not in idx
                    or g_node is None or g_node not in idx):
                skipped += 1
                continue
            s_i, a_i, g_i = idx[t.state_node], idx[t.action_node], idx[g_node]
            nbr = ei[1, ei[0] == s_i]
            pos = (nbr == a_i).nonzero(as_tuple=True)[0]
            if nbr.numel() < 2 or pos.numel() == 0:
                skipped += 1
                continue
            logits = agent._q_head_raw(emb[s_i], emb[nbr], emb[g_i]).reshape(1, -1)
            terms.append(F.cross_entropy(logits, pos[:1]))
            used += 1
        if not terms:
            continue
        loss = torch.stack(terms).mean()
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, agent.grad_clip_norm)
        agent.optimizer.step()
        agent._step_count += 1
        if agent._step_count % agent.target_update_interval == 0:
            agent.update_target()
        losses.append(float(loss.item()))
    agent.update_target()
    agent.encode(data)
    return {"pretrain_steps": n_steps,
            "pretrain_final_loss": float(np.mean(losses[-50:])) if losses else None,
            "bc_examples_used": used, "bc_examples_skipped": skipped}


def _train(g, agent, buf, oracles, queries, re_seed: bool, seed: int) -> None:
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=N_ITER,
        episodes_per_iteration=EPISODES,
        grad_steps_per_episode=TC["grad_steps_per_episode"],
        batch_size=TC["batch_size"],
        re_seed_experts_each_iteration=re_seed,
        seed=seed,
        pre_seed_n_states=TC["pre_seed_n_states"],
        pre_seed_k_paths=TC["pre_seed_k_paths"],
        env_kwargs={"mask_visited": True},
    )


def run_arm(arm: str, seed: int, scenario) -> dict:
    set_global_seed(seed)
    src, dst = scenario.source_node, scenario.destination_node
    queries = [(src, dst)]
    g = DynamicGraph(**base.GRID_CFG, seed=scenario.grid_seed)
    agent = _agent(seed)
    extra: dict = {}
    t0 = time.perf_counter()

    if arm == "persistent":
        oracles = base._build_oracles(g, seed)
        buf = ExpertReplayBuffer(expert_ratio=TC["expert_ratio"], rng=np.random.default_rng(seed))
        _train(g, agent, buf, oracles, queries, re_seed=True, seed=seed)
    elif arm == "cold":
        buf = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(seed))
        _train(g, agent, buf, [], queries, re_seed=False, seed=seed)
    else:
        oracles = base._build_oracles(g, seed)
        buf = ExpertReplayBuffer(expert_ratio=TC["expert_ratio"], rng=np.random.default_rng(seed))
        extra["expert_pool_size"] = _preseed(g, buf, oracles, queries)
        if arm == "keep_no_refresh":
            pass  # pool stays at rho=0.30 through RL; oracles are not re-consulted
        else:
            fn = pretrain_dqfd if arm == "dqfd_pretrain" else pretrain_bc
            extra.update(fn(agent, g, buf, N_PRE, seed))
            # Remove demonstrations entirely: sample() back-fills from the
            # expert pool when the online pool is short, so a ratio of 0 alone
            # would let them leak back into early batches.
            buf.expert_pool.clear()
            buf.expert_ratio = 0.0
        extra["rl_steps_start"] = agent._step_count
        _train(g, agent, buf, [], queries, re_seed=False, seed=seed)

    train_s = time.perf_counter() - t0
    data = dynamic_graph_to_pyg(g, device=agent.device)
    res = base._eval_dual_budget(agent, g, src, dst, data)
    dij = base._dijkstra(g.graph, g.nodes, src, dst)
    row = {"arm": arm, "train_s": train_s, "grad_steps_total": agent._step_count,
           "dijkstra_cost": base._json_float(dij), **extra}
    for b in BUDGETS:
        sfx = "" if b == 300 else f"_{b}"
        row["strict" + sfx] = bool(res[b]["strict"])
        row["cost" + sfx] = base._json_float(res[b]["cost"])
    del agent, buf
    gc.collect()
    return row


# ── Statistics ────────────────────────────────────────────────────────────────

def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    p = None
    if n10 + n01 > 0:
        p = float(scipy_stats.binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"b_first_only": n10, "c_second_only": n01, "p_exact": p}


def wilcoxon_cost(ca: list, cb: list) -> dict:
    pairs = [(x, y) for x, y in zip(ca, cb) if x is not None and y is not None]
    out = {"n_both_reached": len(pairs), "p": None, "first_cheaper": None}
    if len(pairs) >= 6:
        d = [x - y for x, y in pairs if x != y]
        out["first_cheaper"] = sum(1 for x, y in pairs if x < y)
        if d:
            out["p"] = float(scipy_stats.wilcoxon(d).pvalue)
    return out


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

    def cmp(a1: str, a2: str) -> None:
        if a1 in by_arm and a2 in by_arm and len(by_arm[a1]) == len(by_arm[a2]) == n:
            ra = [r["strict"] for r in by_arm[a1]]
            rb = [r["strict"] for r in by_arm[a2]]
            agg["tests"][f"{a1}_vs_{a2}"] = {
                "reach": exact_mcnemar(ra, rb),
                "cost": wilcoxon_cost([r["cost"] for r in by_arm[a1]],
                                      [r["cost"] for r in by_arm[a2]]),
            }

    primary = [("persistent", "dqfd_pretrain"), ("persistent", "bc_pretrain")]
    for a1, a2 in primary + [("persistent", "cold"), ("dqfd_pretrain", "cold"),
                             ("bc_pretrain", "cold"), ("persistent", "keep_no_refresh")]:
        cmp(a1, a2)
    # Holm over the two pre-registered primary comparisons
    ps = [(k, agg["tests"][f"{a}_vs_{b}"]["reach"]["p_exact"])
          for a, b in primary if f"{a}_vs_{b}" in agg["tests"]
          for k in [f"{a}_vs_{b}"]]
    ps = [(k, p) for k, p in ps if p is not None]
    ps.sort(key=lambda kp: kp[1])
    m = len(ps)
    running = 0.0
    for i, (k, p) in enumerate(ps):
        running = max(running, min(1.0, (m - i) * p))
        agg["tests"][k]["reach"]["p_holm"] = running
    # Every arm must have been evaluated on the same graph state in a cell
    agg["eval_state_consistent_cells"] = sum(
        1 for c in cells if len({json.dumps(r["dijkstra_cost"]) for r in c["arms"].values()}) == 1)
    with open(out_dir / "aggregate.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    return agg


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    if "--aggregate" in sys.argv:
        print(json.dumps(aggregate(OUT_DIR), indent=2))
        return
    prov = write_provenance()
    if not SMOKE and not prov["cuda_available"] and os.environ.get("QWARM_ALLOW_CPU") != "1":
        raise SystemExit(
            f"Refusing a full run on {prov['torch_version']} without CUDA. The published "
            "50x50 arms ran on torch 2.11.0+cu128; reinstall the CUDA build, or set "
            "QWARM_ALLOW_CPU=1 to run every arm on CPU deliberately.")
    print(f"arms={ARMS}  n_pre={N_PRE}  iters={N_ITER}  eps={EPISODES}  "
          f"torch={prov['torch_version']}  device={prov['resolved_device']}  smoke={SMOKE}",
          flush=True)

    tag = "_".join(str(s) for s in SEEDS)
    out = OUT_DIR / f"cells_seed_{tag}.json"
    done = json.load(open(out)) if out.exists() else []
    done_ids = {(c["seed"], c["scenario_id"]) for c in done}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tmpl = {k: base.GRID_CFG[k] for k in ("grid_width", "grid_height",
                                               "extra_edges", "deactivate_prob")}
        scenarios = base.sample_scenarios(tmpl, n_scenarios=base.N_SCENARIOS, rng=rng)[:N_SCEN]
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
