"""Experiment C -- is graph locality ABSENT from the embedding, or merely
invisible to cosine?

Experiment B showed that whitening restores the similarity gate's dynamic range
but does not improve retrieval quality. That leaves two very different
diagnoses, with opposite engineering consequences:

  (i)  the encoder has destroyed graph-locality information
       -> no inference-time fix can work; the representation must change.
  (ii) the information survives but is not aligned with the cosine direction
       -> a learned retrieval metric recovers it, with the encoder untouched.

This script decides between them by training a probe to predict graph hop
distance from FROZEN pair embeddings, and comparing its held-out rank
correlation against the cosine baseline on the same pairs.

Probes (all on frozen embeddings, fit on a disjoint train split):
    cosine        no learning -- the incumbent retrieval metric
    ridge         linear regression on [|h_i - h_j|, h_i * h_j]
    mlp           1-hidden-layer MLP on the same features
    input_ridge   ridge on the RAW 4-dim node features (floor: what was
                  available before the encoder ran)

Usage:
    python scripts/run_metric_probe_experiment.py --out runs/metric_probe.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qwarm.env.pyg_adapter import dynamic_graph_to_pyg

from run_geometry_audit import bfs_hops, load_agent, rebuild_graph  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sample_pairs(g, data, n_anchors: int, per_anchor: int, rng):
    idx = data.node_id_to_idx
    active = [n for n in g.nodes if g.nodes[n]["active"] and n in idx]
    anchors = rng.choice(active, size=min(n_anchors, len(active)), replace=False)
    I, J, D = [], [], []
    for a in anchors:
        hops = bfs_hops(g, str(a), max_hops=12)
        reach = [n for n in hops if n != a and n in idx]
        if not reach:
            continue
        pick = rng.choice(reach, size=min(per_anchor, len(reach)), replace=False)
        for p in pick:
            I.append(idx[str(a)])
            J.append(idx[str(p)])
            D.append(hops[str(p)])
    return np.array(I), np.array(J), np.array(D, dtype=float)


def pair_features(E: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    a, b = E[I], E[J]
    return np.concatenate([np.abs(a - b), a * b], axis=1)


def cosine_scores(E: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - np.sum(En[I] * En[J], axis=1)      # cosine DISTANCE


def ridge_fit_predict(Xtr, ytr, Xte, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge with intercept (no sklearn dependency)."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    xm, ym = Xtr.mean(0), float(np.mean(ytr))
    Xc = Xtr - xm
    d = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ (ytr - ym))
    return (Xte - xm) @ w + ym


def mlp_fit_predict(Xtr, ytr, Xte, hidden: int = 128, epochs: int = 300,
                    seed: int = 0) -> np.ndarray:
    """Small MLP regressor in torch, with a held-out-free early stop on train
    loss plateau. Deterministic given `seed`."""
    torch.manual_seed(seed)
    xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32).unsqueeze(1)
    xe = torch.as_tensor(Xte, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bad = float("inf"), 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
        v = float(loss.item())
        if v < best - 1e-4:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= 30:
                break
    with torch.no_grad():
        return net(xe).squeeze(1).numpy()


def run_cell(ckpt_path, scale_entry, grid_seed, hidden_dim, rng,
             n_anchors: int, per_anchor: int) -> dict:
    g = rebuild_graph(scale_entry, grid_seed)
    data = dynamic_graph_to_pyg(g)
    agent = load_agent(ckpt_path, hidden_dim)
    with torch.no_grad():
        H = agent._encoder_raw(data.x, data.edge_index).numpy().astype(np.float64)
    X_in = data.x.numpy().astype(np.float64)

    I, J, D = sample_pairs(g, data, n_anchors, per_anchor, rng)
    if I.size < 400:
        return {}

    # Split by ANCHOR so train and test never share an anchor node -- otherwise
    # the probe can memorise per-anchor offsets and the correlation is inflated.
    uniq_anchors = np.unique(I)
    rng.shuffle(uniq_anchors)
    cut = int(0.6 * uniq_anchors.size)
    train_anchors = set(uniq_anchors[:cut].tolist())
    tr = np.array([i in train_anchors for i in I])
    te = ~tr
    if tr.sum() < 200 or te.sum() < 200:
        return {}

    out: dict = {"n_pairs": int(I.size), "n_train": int(tr.sum()), "n_test": int(te.sum())}

    # --- incumbent: cosine, no learning -----------------------------------
    cos = cosine_scores(H, I, J)
    out["cosine"] = float(spearmanr(cos[te], D[te]).statistic)

    # --- learned metrics on frozen embeddings ------------------------------
    F_emb = pair_features(H, I, J)
    mu, sd = F_emb[tr].mean(0), F_emb[tr].std(0) + 1e-8
    Fz = (F_emb - mu) / sd

    out["ridge"] = float(
        spearmanr(ridge_fit_predict(Fz[tr], D[tr], Fz[te]), D[te]).statistic
    )
    out["mlp"] = float(
        spearmanr(mlp_fit_predict(Fz[tr], D[tr], Fz[te]), D[te]).statistic
    )

    # --- floor: raw node features ------------------------------------------
    F_in = pair_features(X_in, I, J)
    mu2, sd2 = F_in[tr].mean(0), F_in[tr].std(0) + 1e-8
    Fz2 = (F_in - mu2) / sd2
    out["input_ridge"] = float(
        spearmanr(ridge_fit_predict(Fz2[tr], D[tr], Fz2[te]), D[te]).statistic
    )
    out["input_cosine"] = float(spearmanr(cosine_scores(X_in, I, J)[te], D[te]).statistic)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/metric_probe.json")
    ap.add_argument("--manifest", default="demo_agents/manifest.json")
    ap.add_argument("--agents-dir", default=None,
                    help="directory holding the .pt files "
                         "(default: the manifest's own directory)")
    ap.add_argument("--n-anchors", type=int, default=60)
    ap.add_argument("--per-anchor", type=int, default=150)
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
                r = run_cell(agents_dir / fname, entry,
                             entry["grid_seed"], entry["hidden_dim"], rng,
                             args.n_anchors, args.per_anchor)
                if not r:
                    print("      skipped", flush=True)
                    continue
                r.update({"scale": scale, "arm": arm,
                          "scenario_id": sc["scenario_id"], "checkpoint": fname})
                rows.append(r)
                print(f"      cosine={r['cosine']:+.3f}  ridge={r['ridge']:+.3f}  "
                      f"mlp={r['mlp']:+.3f}   (input floor: cos={r['input_cosine']:+.3f} "
                      f"ridge={r['input_ridge']:+.3f})", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
