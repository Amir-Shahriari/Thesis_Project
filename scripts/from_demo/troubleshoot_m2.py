"""Targeted diagnostic for the 4 cells that fail M2.

For each failing cell, this script:
1. Trains warm agent (matching sweep config)
2. Inspects the expert pool composition after training
3. Does a step-by-step greedy rollout showing Q-values at each step
4. Identifies the first "wrong turn" decision
5. Checks whether the correct action is in the expert pool
"""
from __future__ import annotations

import json
import time
import numpy as np
import torch

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.metrics import _dijkstra_cost
from qwarm.eval.path_evaluator import evaluate_agent
from qwarm.utils.seeding import set_global_seed

# ── Same config as the sweep ──────────────────────────────────────────────────
GRID_TEMPLATE = {"grid_width": 25, "grid_height": 25, "extra_edges": 2, "deactivate_prob": 0.15}
TRAIN_CFG = {
    "n_iterations": 5,
    "episodes_per_iteration": 100,
    "grad_steps_per_episode": 4,   # Updated config
    "batch_size": 64,
    "hidden_dim": 64,
    "expert_ratio": 0.40,
}

# ── 4 failing cells from sweep_phase3_separategraph.json ─────────────────────
# (seed, grid_seed, src, dst, warm_cost_seen, dijkstra_seen)
FAILING_CELLS = [
    (2024, 518677876, "Node_50",  "Node_104", 3046, 49.08),   # ratio 62x - WORST
    (1337, None,      "Node_576", "Node_245", 1267, 45.38),   # ratio 28x
]

def get_grid_seed(outer_seed: int, src: str, dst: str) -> int:
    """Recover the grid_seed used by sample_scenarios."""
    import numpy as np
    from qwarm.eval.scenario_sampler import sample_scenarios
    rng = np.random.default_rng(outer_seed)
    scenarios = sample_scenarios(GRID_TEMPLATE, n_scenarios=5, rng=rng, min_euclidean_fraction=0.6)
    for s in scenarios:
        if s.source_node == src and s.destination_node == dst:
            return s.grid_seed
    raise ValueError(f"Could not find scenario {src}->{dst} for seed={outer_seed}")


def get_optimal_path(g: DynamicGraph, src: str, dst: str) -> list[str]:
    """Run Dijkstra to get the actual optimal path."""
    import heapq
    queue = [(0.0, src, [src])]
    best = {src: 0.0}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == dst:
            return path
        if cost > best.get(node, float("inf")) + 1e-9:
            continue
        for nb, edge in g.graph[node].items():
            if not edge["active"] or not g.nodes[nb]["active"]:
                continue
            sc = edge["distance"] + 0.1 * edge["time"] + g.nodes[nb]["node_penalty"]
            nc = cost + sc
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(queue, (nc, nb, path + [nb]))
    return []


def diagnose_cell(outer_seed: int, grid_seed_override: int | None,
                  src: str, dst: str,
                  prev_warm_cost: float, prev_dijk: float):
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: seed={outer_seed}  {src} -> {dst}")
    print(f"  Previous result: warm={prev_warm_cost:.0f}  dijkstra={prev_dijk:.1f}  ratio={prev_warm_cost/prev_dijk:.1f}x")
    print(f"{'='*70}")

    set_global_seed(outer_seed)
    grid_seed = grid_seed_override if grid_seed_override is not None else get_grid_seed(outer_seed, src, dst)
    print(f"  grid_seed={grid_seed}")

    # ── 1. Build fresh graph and get state-5 dijkstra ─────────────────────────
    g = DynamicGraph(**GRID_TEMPLATE, seed=grid_seed)
    dijk_state0 = _dijkstra_cost(g, src, dst)
    opt_path_state0 = get_optimal_path(g, src, dst)

    g_eval = DynamicGraph(**GRID_TEMPLATE, seed=grid_seed)
    for i in range(1, TRAIN_CFG["n_iterations"] + 1):
        g_eval.update_graph(iteration=i)
    dijk_state5 = _dijkstra_cost(g_eval, src, dst)
    opt_path_state5 = get_optimal_path(g_eval, src, dst)

    print(f"\n  Dijkstra at state 0: {dijk_state0:.2f}  path ({len(opt_path_state0)} hops): {opt_path_state0}")
    print(f"  Dijkstra at state 5: {dijk_state5:.2f}  path ({len(opt_path_state5)} hops): {opt_path_state5}")

    # ── 2. Train warm agent ───────────────────────────────────────────────────
    g_train = DynamicGraph(**GRID_TEMPLATE, seed=grid_seed)
    oracles = [ClassicalAStar(g_train.nodes, g_train.graph),
               QuantumInspiredStochasticOracle(g_train.nodes, g_train.graph)]
    queries = [(src, dst)]

    agent = GNNDQN(node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=outer_seed)
    buf = ExpertReplayBuffer(expert_ratio=TRAIN_CFG["expert_ratio"],
                              rng=np.random.default_rng(outer_seed))

    print(f"\n  Training warm agent...")
    t0 = time.perf_counter()
    logs = train_gnn_dqn(
        g_train, PathfindingEnv, agent, buf, oracles, queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=outer_seed,
        pre_seed_n_states=TRAIN_CFG["n_iterations"],
        pre_seed_k_paths=10,
    )
    train_s = time.perf_counter() - t0
    print(f"  Training done in {train_s:.1f}s")
    print(f"  Expert pool: {len(buf.expert_pool)}  Online pool: {len(buf.online_pool)}")
    print(f"  Final iter goal_reach_rate: {logs['goal_reach_rate'][-1]:.2%}")
    print(f"  Final iter mean_return: {logs['mean_return'][-1]:.1f}")
    print(f"  Final iter mean_loss: {logs['mean_loss'][-1]:.4f}")

    # ── 3. Expert pool analysis ───────────────────────────────────────────────
    expert_states = set()
    expert_actions_from = {}
    for t in buf.expert_pool:
        expert_states.add(t.state_node)
        if t.state_node not in expert_actions_from:
            expert_actions_from[t.state_node] = []
        expert_actions_from[t.state_node].append(t.action_node)

    print(f"\n  Expert pool analysis:")
    print(f"    Unique source states: {len(expert_states)}")
    print(f"    Optimal path nodes in expert pool (state 5): {sum(1 for n in opt_path_state5 if n in expert_states)}/{len(opt_path_state5)}")

    # Check if each hop of optimal path at state 5 is in expert pool
    for i in range(len(opt_path_state5) - 1):
        s, a = opt_path_state5[i], opt_path_state5[i+1]
        in_pool = any(t.state_node == s and t.action_node == a for t in buf.expert_pool)
        print(f"    Hop {i}: {s} -> {a}  in_expert_pool={in_pool}")

    # ── 4. Evaluate at graph state = end-of-training ──────────────────────────
    data = dynamic_graph_to_pyg(g_train, device=agent.device)
    result = evaluate_agent(agent, g_train, src, dst, data)
    agent_path = result["path"]
    agent_cost = result["cost"]
    agent_reached = result["reached_goal"]
    dijk_final = _dijkstra_cost(g_train, src, dst)

    print(f"\n  Evaluation (graph at final training state):")
    print(f"    Dijkstra: {dijk_final:.2f}  ({len(get_optimal_path(g_train, src, dst))} hops)")
    print(f"    Agent reached goal: {agent_reached}  cost: {agent_cost:.1f}  hops: {len(agent_path)-1}")
    if agent_reached:
        print(f"    Ratio: {agent_cost/max(dijk_final,1e-9):.2f}x")

    # ── 5. Step-by-step Q-value trace from source ─────────────────────────────
    print(f"\n  Step-by-step Q-value trace (first 10 steps from {src}):")
    agent.encode(data)
    current = src
    visited = {src}
    step_costs = []

    for step in range(10):
        valid = [nb for nb, d in g_train.graph[current].items()
                 if d["active"] and g_train.nodes[nb]["active"] and nb not in visited]
        if not valid:
            print(f"    Step {step}: DEAD END at {current}")
            break

        # Get Q values for all valid actions
        with torch.no_grad():
            q_vals = {}
            for nb in valid:
                q_vec = agent.q_values(current, [nb], dst, data)
                q_vals[nb] = float(q_vec[0])

        chosen = max(q_vals, key=q_vals.get)
        edge = g_train.graph[current][chosen]
        sc = edge["distance"] + 0.1 * edge["time"] + g_train.nodes[chosen]["node_penalty"]

        # Is this the optimal next hop?
        opt_next = opt_path_state5[opt_path_state5.index(current) + 1] if current in opt_path_state5 and opt_path_state5.index(current) < len(opt_path_state5)-1 else None
        on_opt = "(ON OPT PATH)" if chosen == opt_next else f"(WRONG - opt={opt_next})"
        in_expert = any(t.state_node == current and t.action_node == chosen for t in buf.expert_pool)
        opt_in_expert = opt_next and any(t.state_node == current and t.action_node == opt_next for t in buf.expert_pool)

        top3 = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  ".join(f"{k}={v:.3f}" for k, v in top3)
        print(f"    Step {step}: {current} -> {chosen} (cost={sc:.1f}) {on_opt}")
        print(f"      top Q: {top3_str}")
        print(f"      chosen_in_expert={in_expert}  opt_in_expert={opt_in_expert}")

        if chosen == dst:
            print(f"    ** REACHED GOAL **")
            break
        visited.add(chosen)
        step_costs.append(sc)
        current = chosen

    # ── 6. Pinpoint: at source, what Q does agent assign to optimal next hop? ──
    src_node = src
    if src_node in [t.state_node for t in buf.expert_pool]:
        valid_at_src = [nb for nb, d in g_train.graph[src_node].items()
                        if d["active"] and g_train.nodes[nb]["active"]]
        print(f"\n  Q-values at source {src_node} ({len(valid_at_src)} valid actions):")
        agent.encode(data)
        q_at_src = {nb: float(agent.q_values(src_node, [nb], dst, data)[0]) for nb in valid_at_src}
        for nb, q in sorted(q_at_src.items(), key=lambda x: x[1], reverse=True)[:5]:
            is_opt = nb in (opt_path_state5[1:2] if len(opt_path_state5) > 1 else [])
            in_exp = any(t.state_node == src_node and t.action_node == nb for t in buf.expert_pool)
            print(f"    {nb}: Q={q:.4f}  is_opt_next={is_opt}  in_expert={in_exp}")


def main():
    # Get grid_seeds for seed=1337 cells
    import numpy as np
    from qwarm.eval.scenario_sampler import sample_scenarios
    rng = np.random.default_rng(1337)
    scenarios = sample_scenarios(GRID_TEMPLATE, n_scenarios=5, rng=rng, min_euclidean_fraction=0.6)
    gs_1337 = {(s.source_node, s.destination_node): s.grid_seed for s in scenarios}
    print("grid_seeds for seed=1337:", gs_1337)

    cells = [
        # (outer_seed, grid_seed, src, dst, prev_warm, prev_dijk)
        (2024, 518677876, "Node_50",  "Node_104", 3046, 49.08),
    ]

    for cell in cells:
        diagnose_cell(*cell)
        print()


if __name__ == "__main__":
    main()
