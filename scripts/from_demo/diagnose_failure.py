"""§5 diagnostic: reproduce worst cell and capture 10-step rollout with Q-values.

Worst cell: seed=2024, scenario_id=seed518677876_s3
  src=Node_174  dst=Node_451  dijkstra=27.66  warm=2367.0  ratio=85.56×
"""
from __future__ import annotations

import heapq
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
from qwarm.eval.scenario_sampler import sample_scenarios
from qwarm.utils.seeding import set_global_seed

GRID_TEMPLATE = {
    "grid_width": 25,
    "grid_height": 25,
    "extra_edges": 2,
    "deactivate_prob": 0.15,
}
TRAIN_CFG = {
    "n_iterations": 5,
    "episodes_per_iteration": 100,
    "grad_steps_per_episode": 2,
    "batch_size": 64,
    "hidden_dim": 64,
    "expert_ratio": 0.40,
}
TARGET_SEED = 2024
TARGET_SCENARIO_IDX = 3  # s3


def _dijkstra_next_hop(graph, nodes, source, destination):
    """Return the optimal next hop from source toward destination."""
    queue: list[tuple[float, str, str | None]] = [(0.0, source, None)]
    best: dict[str, float] = {source: 0.0}
    parent: dict[str, str | None] = {source: None}
    visited: set[str] = set()
    while queue:
        cost, node, _ = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            break
        for nb, edata in graph[node].items():
            if edata["active"] and nodes[nb]["active"]:
                step_cost = edata["distance"] + 0.1 * edata["time"] + nodes[nb]["node_penalty"]
                nc = cost + step_cost
                if nc < best.get(nb, float("inf")):
                    best[nb] = nc
                    parent[nb] = node
                    heapq.heappush(queue, (nc, nb, node))
    # trace back first hop
    if destination not in parent:
        return None
    node = destination
    while parent[node] != source and parent[node] is not None:
        node = parent[node]
    return node if parent[node] == source else None


def _rollout_with_qvals(agent, dyn_graph, source, destination, data, n_steps=10):
    """Greedy rollout capturing Q-values, expert recommendation, and step cost."""
    agent.encode(data)
    current = source
    visited: set[str] = {source}
    rows = []

    for step in range(n_steps):
        valid = [
            nb for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            rows.append({
                "step": step, "state": current, "valid_actions": [],
                "expert_rec": None, "agent_choice": None,
                "q_values": {}, "step_cost": None, "note": "NO_VALID_ACTIONS"
            })
            break

        # Q-values
        q_tensor = agent.q_values(current, valid, destination, data)
        q_dict = {a: round(float(q_tensor[i].item()), 4) for i, a in enumerate(valid)}
        agent_choice = valid[int(q_tensor.argmax().item())]

        # Expert recommendation: optimal next hop toward dst ignoring visited
        # (dijkstra from current over the current active graph)
        expert_rec = _dijkstra_next_hop(dyn_graph.graph, dyn_graph.nodes, current, destination)

        # Step cost
        edge = dyn_graph.graph[current][agent_choice]
        step_cost = (
            edge["distance"] + 0.1 * edge["time"]
            + dyn_graph.nodes[agent_choice]["node_penalty"]
        )

        rows.append({
            "step": step,
            "state": current,
            "valid_actions": valid[:5],  # cap at 5 for readability
            "n_valid": len(valid),
            "expert_rec": expert_rec,
            "agent_choice": agent_choice,
            "q_agent_choice": round(float(q_tensor[valid.index(agent_choice)].item()), 4),
            "q_expert": round(float(q_tensor[valid.index(expert_rec)].item()), 4) if expert_rec in valid else None,
            "q_max": round(float(q_tensor.max().item()), 4),
            "q_min": round(float(q_tensor.min().item()), 4),
            "step_cost": round(step_cost, 4),
            "wrong_choice": agent_choice != expert_rec,
        })

        visited.add(agent_choice)
        current = agent_choice
        if current == destination:
            rows[-1]["note"] = "REACHED_GOAL"
            break

    return rows


def main():
    print(f"\n{'='*65}")
    print("  §5 DIAGNOSTIC: seed=2024 scenario_id=seed518677876_s3")
    print(f"{'='*65}")

    set_global_seed(TARGET_SEED)
    rng = np.random.default_rng(TARGET_SEED)
    scenarios = sample_scenarios(
        GRID_TEMPLATE, n_scenarios=5, rng=rng, min_euclidean_fraction=0.6
    )
    scenario = scenarios[TARGET_SCENARIO_IDX]
    print(f"  Scenario: {scenario.scenario_id}  grid_seed={scenario.grid_seed}")
    print(f"  Source: {scenario.source_node}  ->  Dest: {scenario.destination_node}")
    print(f"  Euclidean dist: {scenario.euclidean_distance:.1f}")

    g = DynamicGraph(
        grid_width=GRID_TEMPLATE["grid_width"],
        grid_height=GRID_TEMPLATE["grid_height"],
        extra_edges=GRID_TEMPLATE["extra_edges"],
        deactivate_prob=GRID_TEMPLATE["deactivate_prob"],
        seed=scenario.grid_seed,
    )
    src = scenario.source_node
    dst = scenario.destination_node
    queries = [(src, dst)]
    oracles = [
        ClassicalAStar(g.nodes, g.graph),
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
    ]

    # ---------- Train warm agent ----------
    print("\n  Training warm agent...")
    warm_agent = GNNDQN(
        node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=TARGET_SEED
    )
    warm_agent.open_trace("runs/diag_trace.jsonl")
    warm_buf = ExpertReplayBuffer(
        expert_ratio=TRAIN_CFG["expert_ratio"],
        rng=np.random.default_rng(TARGET_SEED),
    )
    t0 = time.perf_counter()
    train_gnn_dqn(
        g, PathfindingEnv, warm_agent, warm_buf, oracles, queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=True,
        seed=TARGET_SEED,
    )
    warm_agent.close_trace()
    print(f"  Warm training done in {time.perf_counter()-t0:.1f}s")
    print(f"  Expert pool: {len(warm_buf.expert_pool)}  Online pool: {len(warm_buf.online_pool)}")

    # ---------- Train cold agent ----------
    print("\n  Training cold agent...")
    cold_agent = GNNDQN(
        node_in_dim=4, hidden_dim=TRAIN_CFG["hidden_dim"], seed=TARGET_SEED
    )
    cold_buf = ExpertReplayBuffer(expert_ratio=0.0, rng=np.random.default_rng(TARGET_SEED))
    train_gnn_dqn(
        g, PathfindingEnv, cold_agent, cold_buf, [], queries,
        n_iterations=TRAIN_CFG["n_iterations"],
        episodes_per_iteration=TRAIN_CFG["episodes_per_iteration"],
        grad_steps_per_episode=TRAIN_CFG["grad_steps_per_episode"],
        batch_size=TRAIN_CFG["batch_size"],
        re_seed_experts_each_iteration=False,
        seed=TARGET_SEED,
    )
    print("  Cold training done.")

    # ---------- Evaluate ----------
    data = dynamic_graph_to_pyg(g, device=warm_agent.device)

    def _greedy_cost(agent, name):
        agent.encode(data)
        current = src
        visited: set[str] = {src}
        cost = 0.0
        for _ in range(500):
            valid = [
                nb for nb, d in g.graph[current].items()
                if d["active"] and g.nodes[nb]["active"] and nb not in visited
            ]
            if not valid:
                break
            action = agent.choose_action(current, valid, dst, data, epsilon=0.0)
            edge = g.graph[current][action]
            cost += edge["distance"] + 0.1 * edge["time"] + g.nodes[action]["node_penalty"]
            visited.add(action)
            current = action
            if current == dst:
                print(f"  {name} reached goal  cost={cost:.2f}")
                return cost
        print(f"  {name} did NOT reach goal")
        return float("inf")

    from qwarm.eval.metrics import _dijkstra_cost
    dijkstra_ref = _dijkstra_cost(g, src, dst)
    print(f"\n  Dijkstra reference cost: {dijkstra_ref:.2f}")
    warm_cost = _greedy_cost(warm_agent, "Warm")
    cold_cost = _greedy_cost(cold_agent, "Cold")
    if warm_cost < float("inf"):
        print(f"  Warm ratio: {warm_cost/max(dijkstra_ref,1e-9):.2f}×")
    if cold_cost < float("inf"):
        print(f"  Cold ratio: {cold_cost/max(dijkstra_ref,1e-9):.2f}×")

    # ---------- 10-step rollout ----------
    print(f"\n{'='*65}")
    print("  10-STEP ROLLOUT (warm agent, greedy, epsilon=0)")
    print(f"{'='*65}")
    print(f"  src={src}  dst={dst}")
    print(f"  {'Step':<5} {'State':<12} {'ExpertRec':<12} {'AgentChoice':<12} {'Q_choice':>9} {'Q_expert':>9} {'Q_max':>9} {'Cost':>8} {'Match'}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*5}")

    warm_agent.encode(data)
    rollout = _rollout_with_qvals(warm_agent, g, src, dst, data, n_steps=10)
    for r in rollout:
        match_str = "OK" if not r.get("wrong_choice") else "WRONG"
        note = r.get("note", "")
        print(
            f"  {r['step']:<5} {r['state']:<12} {str(r['expert_rec']):<12} "
            f"{str(r['agent_choice']):<12} {str(r.get('q_agent_choice','—')):>9} "
            f"{str(r.get('q_expert','—')):>9} {str(r.get('q_max','—')):>9} "
            f"{str(r.get('step_cost','—')):>8} {match_str}  {note}"
        )

    # ---------- Training trace summary ----------
    print(f"\n{'='*65}")
    print("  TRAINING TRACE SUMMARY (last 10 gradient steps)")
    print(f"{'='*65}")
    try:
        with open("runs/diag_trace.jsonl") as f:
            lines = f.readlines()
        print(f"  Total gradient steps logged: {len(lines)}")
        print(f"  {'iter':>5} {'td_loss':>10} {'margin_loss':>12} {'expert%':>8} {'online%':>8}")
        for line in lines[-10:]:
            rec = json.loads(line)
            print(
                f"  {rec.get('iteration', '?'):>5} "
                f"{rec.get('td_loss', float('nan')):>10.4f} "
                f"{rec.get('margin_loss', float('nan')):>12.4f} "
                f"{rec.get('expert_frac', float('nan')):>8.2%} "
                f"{rec.get('online_frac', float('nan')):>8.2%}"
            )
    except FileNotFoundError:
        print("  (trace file not found — trace may not be emitted by this build)")

    # ---------- Failed-cell table ----------
    import pathlib
    sweep = json.loads(pathlib.Path("runs/sweep_phase3_dqfd.json").read_text())
    failed = [c for c in sweep if c.get("warm_cost_ratio") is not None and c["warm_cost_ratio"] > 8.0]
    failed.sort(key=lambda c: -c["warm_cost_ratio"])
    print(f"\n{'='*65}")
    print(f"  FAILED M2 CELLS (ratio > 8.0) — {len(failed)} of 25")
    print(f"{'='*65}")
    print(f"  {'seed':>7} {'scenario_id':<25} {'warm_cost':>10} {'cold_cost':>10} {'dijkstra':>10} {'ratio':>7}")
    for c in failed:
        cd = c["cold_cost"] if c["cold_cost"] is not None else float("inf")
        print(
            f"  {c['seed']:>7} {c['scenario_id']:<25} "
            f"{c['warm_cost']:>10.1f} {cd:>10.1f} "
            f"{c['dijkstra_cost']:>10.2f} {c['warm_cost_ratio']:>7.2f}×"
        )


if __name__ == "__main__":
    main()
