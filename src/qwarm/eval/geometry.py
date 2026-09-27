"""Embedding-geometry diagnostics for GNN-RL encoders.

Implements the isotropy / effective-rank instruments that the accompanying
survey argues every retrieval-augmented GNN-RL paper should report alongside
return, plus two retrieval-quality probes that test whether the embedding
still carries the information a similarity gate depends on.

Metrics
-------
isoscore(H)
    IsoScore (Rudman et al., 2022): 0 = all variance on one axis, 1 = perfectly
    isotropic. Computed from the variance profile of the PCA-rotated cloud, so
    it is invariant to rotation and immune to the pathologies of the older
    average-cosine and PCA-explained-variance heuristics.

effective_rank(H)
    exp(Shannon entropy of the normalised singular-value spectrum)
    (Roy & Vetterli, 2007). The continuous analogue of matrix rank; a collapse
    from d toward 1 is the dimensional-collapse signature.

stable_rank(H)
    ||H||_F^2 / ||H||_2^2. Cheap lower bound on rank, reported as a cross-check.

mean_pair_cosine(H)
    Mean cosine similarity between random node pairs. High values indicate the
    narrow-cone anisotropy that makes cosine retrieval uninformative.

top1_variance_ratio(H)
    Fraction of total variance on the leading principal component.

locality_spearman(H, hops)
    Spearman correlation between embedding cosine DISTANCE and graph hop
    distance over sampled node pairs. This is the load-bearing retrieval probe:
    if it is ~0 the embedding no longer encodes graph locality, so a
    nearest-neighbour lookup in embedding space returns entries that are not
    neighbours in the environment -- retrieval still "works" mechanically while
    carrying no signal.

All functions take H as a [N, D] float tensor or array and return plain floats,
so results are JSON-serialisable without conversion.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def _as_numpy(H) -> np.ndarray:
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()
    return np.asarray(H, dtype=np.float64)


def isoscore(H) -> float:
    """IsoScore in [0, 1]. See Rudman, Gillman, Rayne & Eickhoff (2022)."""
    X = _as_numpy(H)
    n, d = X.shape
    if n < 2 or d < 2:
        return float("nan")

    Xc = X - X.mean(axis=0, keepdims=True)
    # PCA rotation: variance along each principal axis.
    # SVD of the centred matrix gives the same axes as eigendecomposition of cov.
    s = np.linalg.svd(Xc, compute_uv=False)
    var_pca = (s ** 2) / max(n - 1, 1)          # length min(n, d)
    if var_pca.shape[0] < d:                     # pad when n < d
        var_pca = np.concatenate([var_pca, np.zeros(d - var_pca.shape[0])])

    norm = np.linalg.norm(var_pca)
    if norm <= 0:
        return 0.0
    sigma_hat = np.sqrt(d) * var_pca / norm      # ||sigma_hat|| = sqrt(d)

    denom = np.sqrt(2.0 * (d - np.sqrt(d)))
    delta = np.linalg.norm(sigma_hat - np.ones(d)) / denom
    k = (d - (delta ** 2) * (d - np.sqrt(d))) ** 2 / (d ** 2)
    iso = (d * k - 1.0) / (d - 1.0)
    return float(np.clip(iso, 0.0, 1.0))


def effective_rank(H) -> float:
    """exp(entropy of normalised singular values) -- Roy & Vetterli (2007)."""
    X = _as_numpy(H)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def stable_rank(H) -> float:
    X = _as_numpy(H)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    if s.size == 0 or s[0] <= 0:
        return 0.0
    return float((s ** 2).sum() / (s[0] ** 2))


def top1_variance_ratio(H) -> float:
    X = _as_numpy(H)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    tot = (s ** 2).sum()
    if tot <= 0:
        return float("nan")
    return float((s[0] ** 2) / tot)


def mean_pair_cosine(H, n_pairs: int = 20000, rng: np.random.Generator | None = None) -> dict:
    """Mean/std cosine similarity over random node pairs (excludes self-pairs)."""
    X = _as_numpy(H)
    n = X.shape[0]
    if n < 2:
        return {"mean": float("nan"), "std": float("nan"), "p95": float("nan")}
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(Xn[i] * Xn[j], axis=1)
    return {
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "p95": float(np.percentile(cos, 95)),
    }


def locality_spearman(
    H,
    hop_dist: np.ndarray,
    pair_idx: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Spearman rho between embedding cosine distance and graph hop distance.

    rho near +1  -> embedding distance tracks graph distance (retrieval is
                    returning genuinely nearby states).
    rho near  0  -> embedding carries no locality information; a
                    nearest-neighbour lookup is effectively arbitrary.
    """
    X = _as_numpy(H)
    i, j = pair_idx
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(Xn[i] * Xn[j], axis=1)
    cos_dist = 1.0 - cos
    finite = np.isfinite(hop_dist) & np.isfinite(cos_dist)
    if finite.sum() < 10:
        return {"rho": float("nan"), "pvalue": float("nan"), "n_pairs": int(finite.sum())}
    rho, p = spearmanr(cos_dist[finite], hop_dist[finite])
    return {"rho": float(rho), "pvalue": float(p), "n_pairs": int(finite.sum())}


def whiten_fit(H, eps: float = 1e-5) -> dict:
    """Fit a ZCA whitening transform on an embedding matrix.

    Returns the mean and whitening matrix; apply with whiten_apply(). This is
    the post-hoc isotropy correction of Mu & Viswanath (2018) in its full
    covariance form -- no retraining, applied identically to stored library
    embeddings and to query embeddings.
    """
    X = _as_numpy(H)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return {"mean": mu, "W": W}


def whiten_apply(H, transform: dict):
    """Apply a fitted ZCA transform. Accepts and returns torch or numpy."""
    was_torch = isinstance(H, torch.Tensor)
    dev = H.device if was_torch else None
    X = _as_numpy(H)
    Z = (X - transform["mean"]) @ transform["W"]
    if was_torch:
        return torch.as_tensor(Z, dtype=torch.float32, device=dev)
    return Z


def all_metrics(H, rng: np.random.Generator | None = None) -> dict:
    """Full geometry panel for one embedding matrix.

    Reports raw AND mean-centred pair cosine because they measure different
    pathologies and only one of them breaks a cosine gate:

      * raw cosine high + IsoScore high  -> common-mean offset ("narrow cone").
        The variance profile is healthy; every vector simply points the same
        way. A cosine similarity gate saturates, but the information is still
        present in the residual.
      * centred cosine high + IsoScore low -> genuine dimensional collapse.
        The information is gone, not merely offset.

    Conflating the two is the reason a gate can look broken while the encoder
    is fine, and vice versa.
    """
    cos = mean_pair_cosine(H, rng=rng)
    X = _as_numpy(H)
    Xc = X - X.mean(axis=0, keepdims=True)
    cos_c = mean_pair_cosine(Xc, rng=rng)
    return {
        "n_nodes": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "isoscore": isoscore(H),
        "effective_rank": effective_rank(H),
        "stable_rank": stable_rank(H),
        "top1_variance_ratio": top1_variance_ratio(H),
        "mean_pair_cosine": cos["mean"],
        "std_pair_cosine": cos["std"],
        "p95_pair_cosine": cos["p95"],
        "mean_pair_cosine_centred": cos_c["mean"],
        "std_pair_cosine_centred": cos_c["std"],
        # ||mean|| / mean||.||  -- 1.0 means every vector is the same direction.
        "mean_norm_ratio": float(
            np.linalg.norm(X.mean(axis=0))
            / max(np.linalg.norm(X, axis=1).mean(), 1e-12)
        ),
    }
