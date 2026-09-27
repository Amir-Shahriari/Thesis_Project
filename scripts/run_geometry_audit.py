"""Experiment A -- embedding-geometry audit of trained GNN-RL encoders.

Loads every released agent checkpoint, reconstructs the exact post-training
graph state deterministically from the manifest (grid seed + n_iterations),
re-encodes it, and reports the full geometry panel plus a graph-locality probe.

The locality probe is the load-bearing measurement: it asks whether embedding
cosine distance still tracks graph hop distance. If it does not, a
similarity-gated retrieval mechanism is selecting neighbours that are not
neighbours in the environment -- the "silent failure" the survey predicts.

Usage:
    python scripts/run_geometry_audit.py --out runs/geometry_audit.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.geometry import all_metrics, locality_spearman

ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_NODE_DEACT = 0.05


def rebuild_graph(scale_entry: dict, grid_seed: int) -> DynamicGraph:
    """Same recipe the demo server uses, so the graph matches the checkpoint."""
    grid = scale_entry["grid"]
    n_iters = scale_entry["train"]["n_iterations"]
    g = DynamicGraph(
        grid_width=grid["grid_width"],
        grid_height=grid["grid_height"],
        extra_edges=grid.get("extra_edges", 2),
        deactivate_prob=grid.get("deactivate_prob", 0.15),
        node_deactivate_prob=grid.get("node_deactivate_prob", _DEFAULT_NODE_DEACT),
        seed=grid_seed,
    )
    for i in range(1, n_iters + 1):
        g.update_graph(iteration=i)
    return g


def bfs_hops(g: DynamicGraph, source: str, max_hops: int = 64) -> dict[str, int]:
    """Hop distance from source over the ACTIVE subgraph."""
    dist = {source: 0}
    frontier = collections.deque([source])
    while frontier:
        node = frontier.popleft()
        d = dist[node]
        if d >= max_hops:
            continue
        for nb, edge in g.graph.get(node, {}).items():
            if nb in dist:
                continue
            if edge["active"] and g.nodes[nb]["active"] and g.nodes[node]["active"]:
                dist[nb] = d + 1
                frontier.append(nb)
    return dist


def locality_pairs(
    g: DynamicGraph,
    data,
    n_anchors: int = 40,
    per_anchor: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Sample (i, j) node-index pairs with a finite graph hop distance."""
    rng = rng or np.random.default_rng(0)
    idx = data.node_id_to_idx
    active = [n for n in g.nodes if g.nodes[n]["active"] and n in idx]
    if len(active) < 2:
        return (np.array([], int), np.array([], int)), np.array([])

    anchors = rng.choice(active, size=min(n_anchors, len(active)), replace=False)
    I, J, D = [], [], []
    for a in anchors:
        hops = bfs_hops(g, str(a))
        reach = [n for n in hops if n != a and n in idx]
        if not reach:
            continue
        pick = rng.choice(reach, size=min(per_anchor, len(reach)), replace=False)
        for p in pick:
            I.append(idx[str(a)])
            J.append(idx[str(p)])
            D.append(hops[str(p)])
    return (np.asarray(I, int), np.asarray(J, int)), np.asarray(D, float)


def load_agent(ckpt_path: pathlib.Path, hidden_dim: int) -> GNNDQN:
    """Released demo checkpoints use *_state_dict keys rather than GNNDQN.save()'s
    schema, and carry their own hidden_dim -- prefer the checkpoint's own value
    so a manifest/checkpoint mismatch cannot silently load the wrong width."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    dim = int(ckpt.get("hidden_dim", hidden_dim))
    # goal_relative widens the Q-head input; absent on pre-ablation checkpoints,
    # where False is the correct default.
    agent = GNNDQN(node_in_dim=int(ckpt.get("node_in_dim", 4)),
                   hidden_dim=dim, device="cpu", seed=0,
                   goal_relative=bool(ckpt.get("goal_relative", False)))
    agent._encoder_raw.load_state_dict(ckpt["encoder_raw_state_dict"])
    agent._q_head_raw.load_state_dict(ckpt["q_head_state_dict"])
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    return agent


def audit_checkpoint(ckpt_path: pathlib.Path, scale_entry: dict,
                     grid_seed: int, hidden_dim: int, rng) -> dict:
    g = rebuild_graph(scale_entry, grid_seed)
    data = dynamic_graph_to_pyg(g)

    agent = load_agent(ckpt_path, hidden_dim)
    with torch.no_grad():
        H = agent._encoder_raw(data.x, data.edge_index)

    panel = all_metrics(H, rng=rng)
    pairs, hops = locality_pairs(g, data, rng=rng)
    panel["locality"] = locality_spearman(H, hops, pairs)

    # Same probe on the RAW input features, as a floor: the 4-dim node feature
    # vector contains normalised coordinates, so its locality correlation is
    # what the encoder had available before it learned anything.
    panel["locality_input_features"] = locality_spearman(data.x, hops, pairs)
    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/geometry_audit.json")
    ap.add_argument("--manifest", default="demo_agents/manifest.json")
    ap.add_argument("--agents-dir", default=None,
                    help="directory holding the .pt files "
                         "(default: the manifest's own directory)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    agents_dir = (pathlib.Path(args.agents_dir) if args.agents_dir
                  else manifest_path.parent)
    if not agents_dir.is_absolute():
        agents_dir = ROOT / agents_dir

    manifest = json.loads(manifest_path.read_text())
    rng = np.random.default_rng(args.seed)
    rows = []

    for scale, entry in manifest.items():
        grid_seed = entry["grid_seed"]
        hidden_dim = entry["hidden_dim"]
        for sc in entry.get("scenarios", []):
            for arm in ("warm", "cold"):
                fname = sc.get(arm)
                if not fname:
                    continue
                path = agents_dir / fname
                if not path.exists():
                    print(f"  MISSING {path}", flush=True)
                    continue
                print(f"  [{scale}/{arm}] {sc['scenario_id']} <- {fname}", flush=True)
                panel = audit_checkpoint(path, entry, grid_seed, hidden_dim, rng)
                panel.update({
                    "scale": scale,
                    "arm": arm,
                    "scenario_id": sc["scenario_id"],
                    "checkpoint": fname,
                    "source": sc.get("source"),
                    "destination": sc.get("destination"),
                    "reached": sc.get(f"{arm}_reached"),
                    "cost_ratio": sc.get(f"{arm}_ratio"),
                })
                rows.append(panel)
                print(
                    f"      iso={panel['isoscore']:.4f} "
                    f"erank={panel['effective_rank']:.1f}/{panel['dim']} "
                    f"cos={panel['mean_pair_cosine']:.3f} "
                    f"cos_c={panel['mean_pair_cosine_centred']:.3f} "
                    f"locality_rho={panel['locality']['rho']:.3f}",
                    flush=True,
                )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
