"""V1-legacy evaluation helper.  Run with V1 backup on PYTHONPATH.

Prints a single JSON line to stdout.
"""
from __future__ import annotations
import argparse, heapq, json, sys, time
import torch

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg


def _load_ckpt(agent: GNNDQN, path: str) -> None:
    ck = torch.load(path, map_location=agent.device, weights_only=False)
    agent._encoder_raw.load_state_dict(ck["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ck["q_head_state_dict"])
    agent.update_target()
    agent._cached_embeddings = None
    agent._cached_data_id = None


def _rollout(agent: GNNDQN, g: DynamicGraph, data, src: str, dst: str, max_steps: int = 1000):
    agent.encode(data)
    path = [src]; vis = {src}; cur = src; cost = 0.0
    t0 = time.perf_counter()
    for _ in range(max_steps):
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
    return {"reached": reached, "cost": cost if reached else None, "infer_ms": ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--grid-seed", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g = DynamicGraph(grid_width=100, grid_height=100, extra_edges=4,
                     deactivate_prob=0.30, seed=args.grid_seed)
    for _ in range(5):
        g.update_graph()
    data = dynamic_graph_to_pyg(g)

    agent = GNNDQN(node_in_dim=4, hidden_dim=128, seed=args.seed)
    _load_ckpt(agent, args.ckpt)
    result = _rollout(agent, g, data, args.src, args.dst)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
