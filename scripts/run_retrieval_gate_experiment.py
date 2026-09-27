"""Experiment B -- does the similarity gate carry signal, and can it be repaired?

Builds an expert library under MATCHED-encoder, MATCHED-graph-state conditions
(the regime in which the thesis found the V3 gate saturating), then measures
three things under four embedding treatments, none of which require retraining.

MEASUREMENTS
------------
1. Gate saturation
   The distribution of sim_max(s) = max cosine to any stored state embedding,
   over all active nodes NOT already in the library (self-matches excluded --
   a node compared against its own stored entry scores 1.0 by construction and
   makes every downstream statistic degenerate).

2. Retrieval quality  [the decisive measurement]
   For each query node the library returns a nearest stored state s*. We ask how
   far s* actually is from the query IN THE GRAPH, and compare against two
   reference points on the same query:
       best   = hop distance to the CLOSEST stored state (perfect retrieval)
       random = expected hop distance to a uniformly random stored state
   quality = (random - retrieved) / (random - best)
       1.0 -> retrieval finds the genuinely nearest stored experience
       0.0 -> retrieval is no better than picking a stored entry at random
   This operationalises "the retrieval mechanism continues to return neighbours
   ... but the neighbours are no longer informative".

3. Gate discrimination AUC
   Can sim_max separate NEAR states (1-2 hops from a stored state, excluding
   the stored states themselves) from FAR states (upper quartile of
   hop-to-library distance)? 0.5 = the gate is blind to novelty.

TREATMENTS (all post-hoc, no retraining)
    raw        encoder output as-is (what V2/V3 use today)
    centred    subtract the corpus mean before the cosine
    whitened   ZCA whitening (Mu & Viswanath all-but-the-top, full covariance)
    raw+pctl   raw embeddings, gate value = empirical CDF of sim_max
               (rank-preserving: restores gate SPREAD, cannot change ranking)

Usage:
    python scripts/run_retrieval_gate_experiment.py --out runs/retrieval_gate.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.geometry import whiten_apply, whiten_fit
from qwarm.training.expert_seeding import _build_nx_graph, _compute_transitions

from run_geometry_audit import bfs_hops, load_agent, rebuild_graph  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_library_paths(g, src: str, dst: str, k_paths: int) -> list[list[str]]:
    G = _build_nx_graph(g)
    if not (G.has_node(src) and G.has_node(dst)):
        return []
    try:
        if not nx.has_path(G, src, dst):
            return []
    except nx.NodeNotFound:
        return []
    out: list[list[str]] = []
    try:
        for path in nx.shortest_simple_paths(G, src, dst, weight="weight"):
            out.append(path)
            if len(out) >= k_paths:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return out


def library_entries(g, paths: list[list[str]], gamma: float = 0.95) -> list[dict]:
    entries: list[dict] = []
    for p in paths:
        for t in _compute_transitions(g, p, iteration=0, goal_node=p[-1], gamma=gamma):
            entries.append({"state": t.state_node, "action": t.action_node,
                            "q": float(t.reward)})
    return entries


def treatments(H: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "raw": H,
        "centred": H - H.mean(dim=0, keepdim=True),
        "whitened": whiten_apply(H, whiten_fit(H)),
    }


def cos_matrix(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return F.normalize(Q, dim=-1, eps=1e-12) @ F.normalize(K, dim=-1, eps=1e-12).T


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score(pos) > score(neg)), ties 0.5."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(allv.size, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def run_cell(ckpt_path, scale_entry, grid_seed, hidden_dim, src, dst,
             k_lib: int, max_queries: int, rng) -> dict | None:
    g = rebuild_graph(scale_entry, grid_seed)
    data = dynamic_graph_to_pyg(g)
    idx = data.node_id_to_idx

    paths = build_library_paths(g, src, dst, k_lib)
    if len(paths) < 4:
        return None
    entries = library_entries(g, paths)
    entries = [e for e in entries if e["state"] in idx and e["action"] in idx]
    if len(entries) < 10:
        return None

    # Deduplicate stored states -- hop geometry is a property of the state set.
    lib_states = sorted({e["state"] for e in entries})
    lib_state_idx = np.array([idx[s] for s in lib_states])

    # Hop distance from every node to the nearest stored state.
    hop_to_lib: dict[str, int] = {}
    per_lib_hops: dict[str, dict[str, int]] = {}
    for s in lib_states:
        h = bfs_hops(g, s, max_hops=12)
        per_lib_hops[s] = h
        for node, d in h.items():
            if d < hop_to_lib.get(node, 10 ** 9):
                hop_to_lib[node] = d

    active = [n for n in g.nodes if g.nodes[n]["active"] and n in idx]
    lib_set = set(lib_states)
    # Query set: active nodes NOT in the library (no self-matches) with a known
    # finite hop distance to at least one stored state.
    cand = [n for n in active if n not in lib_set and n in hop_to_lib]
    if len(cand) < 20:
        return None
    if len(cand) > max_queries:
        cand = list(rng.choice(cand, size=max_queries, replace=False))
    q_idx = np.array([idx[n] for n in cand])

    # Hop distance from each query node to EVERY stored state -> best / random.
    Hops = np.full((len(cand), len(lib_states)), np.nan)
    for j, s in enumerate(lib_states):
        h = per_lib_hops[s]
        for i, n in enumerate(cand):
            if n in h:
                Hops[i, j] = h[n]
    row_ok = np.isfinite(Hops).any(axis=1)
    if row_ok.sum() < 20:
        return None
    Hops = Hops[row_ok]
    q_idx = q_idx[row_ok]
    cand = [c for c, k in zip(cand, row_ok) if k]

    best_hop = np.nanmin(Hops, axis=1)
    rand_hop = np.nanmean(Hops, axis=1)

    # NEAR / FAR sets for the discrimination AUC (self-matches excluded).
    h_arr = np.array([hop_to_lib[n] for n in cand], dtype=float)
    far_thresh = np.percentile(h_arr, 75)
    near_mask = h_arr <= 2
    far_mask = h_arr > max(far_thresh, 2)
    if near_mask.sum() < 5 or far_mask.sum() < 5:
        far_mask = h_arr >= np.percentile(h_arr, 90)
        if near_mask.sum() < 5 or far_mask.sum() < 5:
            return None

    res: dict = {
        "n_library_entries": len(entries),
        "n_library_states": len(lib_states),
        "n_library_paths": len(paths),
        "n_queries": len(cand),
        "n_near": int(near_mask.sum()),
        "n_far": int(far_mask.sum()),
        "library_coverage_fraction": float(len(lib_states) / max(len(active), 1)),
        "mean_best_hop": float(best_hop.mean()),
        "mean_random_hop": float(rand_hop.mean()),
        "treatments": {},
    }

    agent = load_agent(ckpt_path, hidden_dim)
    with torch.no_grad():
        H = agent._encoder_raw(data.x, data.edge_index)

    for name, E in treatments(H).items():
        S = cos_matrix(E[q_idx], E[lib_state_idx])          # [Q, L]
        sim_max = S.max(dim=1).values.numpy()
        top1 = S.argmax(dim=1).numpy()
        retrieved_hop = Hops[np.arange(Hops.shape[0]), top1]
        ok = np.isfinite(retrieved_hop)

        denom = rand_hop[ok] - best_hop[ok]
        good = denom > 1e-9
        quality = float(
            np.mean((rand_hop[ok][good] - retrieved_hop[ok][good]) / denom[good])
        ) if good.any() else float("nan")

        res["treatments"][name] = {
            "sim_max_mean": float(sim_max.mean()),
            "sim_max_std": float(sim_max.std()),
            "sim_max_p05": float(np.percentile(sim_max, 5)),
            "sim_max_frac_above_0.95": float(np.mean(sim_max > 0.95)),
            "mean_retrieved_hop": float(np.nanmean(retrieved_hop)),
            "retrieval_quality": quality,
            "gate_auc_near_vs_far": auc(sim_max[near_mask], sim_max[far_mask]),
        }

    # Percentile calibration: rank-preserving, so retrieval quality and AUC are
    # inherited from raw by construction; only the gate's SPREAD changes.
    raw = res["treatments"]["raw"]
    S = cos_matrix(H[q_idx], H[lib_state_idx])
    sim_max = S.max(dim=1).values.numpy()
    ranks = sim_max.argsort().argsort() / max(sim_max.size - 1, 1)
    res["treatments"]["raw+pctl"] = {
        **raw,
        "sim_max_mean": float(ranks.mean()),
        "sim_max_std": float(ranks.std()),
        "sim_max_p05": float(np.percentile(ranks, 5)),
        "sim_max_frac_above_0.95": float(np.mean(ranks > 0.95)),
        "note": "rank-preserving recalibration; retrieval_quality and AUC "
                "inherited from raw by construction",
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/retrieval_gate.json")
    ap.add_argument("--manifest", default="demo_agents/manifest.json")
    ap.add_argument("--agents-dir", default=None,
                    help="directory holding the .pt files "
                         "(default: the manifest's own directory)")
    ap.add_argument("--k-lib", type=int, default=12)
    ap.add_argument("--max-queries", type=int, default=400)
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
        for sc in entry.get("scenarios", []):
            for arm in ("warm", "cold"):
                fname = sc.get(arm)
                if not fname or not (agents_dir / fname).exists():
                    continue
                print(f"  [{scale}/{arm}] {sc['scenario_id']}", flush=True)
                r = run_cell(
                    agents_dir / fname, entry, entry["grid_seed"],
                    entry["hidden_dim"], sc["source"], sc["destination"],
                    args.k_lib, args.max_queries, rng,
                )
                if r is None:
                    print("      skipped (insufficient paths/queries)", flush=True)
                    continue
                r.update({"scale": scale, "arm": arm,
                          "scenario_id": sc["scenario_id"], "checkpoint": fname})
                rows.append(r)
                print(f"      best_hop={r['mean_best_hop']:.2f} "
                      f"random_hop={r['mean_random_hop']:.2f}", flush=True)
                for t in ("raw", "centred", "whitened"):
                    d = r["treatments"][t]
                    print(f"      {t:9s} simmax={d['sim_max_mean']:.3f}"
                          f"+-{d['sim_max_std']:.3f} sat={d['sim_max_frac_above_0.95']:.2f} "
                          f"hop={d['mean_retrieved_hop']:.2f} "
                          f"quality={d['retrieval_quality']:+.3f} "
                          f"AUC={d['gate_auc_near_vs_far']:.3f}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
