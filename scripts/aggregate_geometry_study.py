"""Consolidate Experiments A/B/C into one citable aggregate artefact.

Reads:
    runs/geometry_audit.json    (A -- encoder geometry panel)
    runs/retrieval_gate.json    (B -- gate saturation, retrieval quality, fixes)
    runs/metric_probe.json      (C -- is locality absent or merely misaligned?)

Writes:
    runs/geometry_study_aggregate.json

Usage:
    python scripts/aggregate_geometry_study.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SCALES = ("25x25", "50x50")


def _load(name: str) -> list[dict]:
    p = RUNS / name
    if not p.exists():
        raise SystemExit(f"missing {p} -- run the corresponding experiment first")
    return json.loads(p.read_text())


def _paired(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def main() -> None:
    geo = _load("geometry_audit.json")
    gate = _load("retrieval_gate.json")
    probe = _load("metric_probe.json")
    out: dict = {"scales": {}}

    for scale in SCALES:
        G = [r for r in geo if r["scale"] == scale]
        B = [r for r in gate if r["scale"] == scale]
        P = [r for r in probe if r["scale"] == scale]
        g = lambda R, k: np.array([r[k] for r in R])  # noqa: E731

        # --- A: geometry, warm vs cold ---------------------------------------
        W = {r["scenario_id"]: r for r in G if r["arm"] == "warm"}
        C = {r["scenario_id"]: r for r in G if r["arm"] == "cold"}
        ids = sorted(set(W) & set(C))
        geo_block = {
            "n_checkpoints": len(G),
            "dim": G[0]["dim"],
            "isoscore_mean": float(g(G, "isoscore").mean()),
            "effective_rank_mean": float(g(G, "effective_rank").mean()),
            "mean_pair_cosine": float(g(G, "mean_pair_cosine").mean()),
            "mean_pair_cosine_centred": float(g(G, "mean_pair_cosine_centred").mean()),
            "mean_norm_ratio": float(g(G, "mean_norm_ratio").mean()),
            "locality_rho_encoder": float(
                np.mean([r["locality"]["rho"] for r in G])),
            "locality_rho_input_features": float(
                np.mean([r["locality_input_features"]["rho"] for r in G])),
            "warm_vs_cold": {},
        }
        for k in ("isoscore", "effective_rank", "mean_pair_cosine"):
            w = np.array([W[i][k] for i in ids])
            c = np.array([C[i][k] for i in ids])
            geo_block["warm_vs_cold"][k] = {
                "warm": float(w.mean()), "cold": float(c.mean()),
                "diff": float(w.mean() - c.mean()),
                "n_pairs": len(ids),
                "n_pairs_favouring_warm": int(
                    np.sum(w > c) if k != "mean_pair_cosine" else np.sum(w < c)),
                "wilcoxon_p": _paired(w, c),
            }

        # --- B: gate saturation, retrieval quality, post-hoc fixes ------------
        tq = lambda t, k: np.array([r["treatments"][t][k] for r in B])  # noqa: E731
        raw_q = tq("raw", "retrieval_quality")
        raw_auc = tq("raw", "gate_auc_near_vs_far")
        gate_block = {
            "n_checkpoints": len(B),
            "library_coverage_fraction": float(g(B, "library_coverage_fraction").mean()),
            "mean_best_hop": float(g(B, "mean_best_hop").mean()),
            "mean_random_hop": float(g(B, "mean_random_hop").mean()),
            "treatments": {},
            "raw_quality_vs_random": {
                "mean": float(raw_q.mean()),
                "t": float(ttest_1samp(raw_q, 0.0).statistic),
                "p": float(ttest_1samp(raw_q, 0.0).pvalue),
            },
            "raw_auc_vs_chance": {
                "mean": float(raw_auc.mean()),
                "t": float(ttest_1samp(raw_auc, 0.5).statistic),
                "p": float(ttest_1samp(raw_auc, 0.5).pvalue),
            },
        }
        for t in ("raw", "centred", "whitened", "raw+pctl"):
            gate_block["treatments"][t] = {
                "sim_max_mean": float(tq(t, "sim_max_mean").mean()),
                "sim_max_std": float(tq(t, "sim_max_std").mean()),
                "saturation_frac_above_0.95": float(
                    tq(t, "sim_max_frac_above_0.95").mean()),
                "mean_retrieved_hop": float(tq(t, "mean_retrieved_hop").mean()),
                "retrieval_quality": float(tq(t, "retrieval_quality").mean()),
                "gate_auc": float(tq(t, "gate_auc_near_vs_far").mean()),
            }
            if t != "raw":
                gate_block["treatments"][t]["saturation_vs_raw_p"] = _paired(
                    tq(t, "sim_max_frac_above_0.95"),
                    tq("raw", "sim_max_frac_above_0.95"))
                gate_block["treatments"][t]["quality_vs_raw_p"] = _paired(
                    tq(t, "retrieval_quality"), raw_q)

        # --- C: learned metric probe -----------------------------------------
        probe_block = {"n_checkpoints": len(P)}
        for k in ("cosine", "ridge", "mlp", "input_cosine", "input_ridge"):
            probe_block[k] = float(g(P, k).mean())
        probe_block["ridge_vs_cosine"] = {
            "diff": float((g(P, "ridge") - g(P, "cosine")).mean()),
            "ratio": float(g(P, "ridge").mean() / g(P, "cosine").mean()),
            "wilcoxon_p": _paired(g(P, "ridge"), g(P, "cosine")),
        }
        probe_block["best_learned_rho"] = float(
            max(g(P, "ridge").max(), g(P, "mlp").max()))

        out["scales"][scale] = {
            "A_geometry": geo_block,
            "B_retrieval_gate": gate_block,
            "C_metric_probe": probe_block,
        }

    out["headline"] = {
        "claim_1_geometry_collapsed": (
            "IsoScore {:.3f} (25x25) / {:.3f} (50x50) on a 0-1 scale; effective "
            "rank {:.0f}/{} and {:.0f}/{}.".format(
                out["scales"]["25x25"]["A_geometry"]["isoscore_mean"],
                out["scales"]["50x50"]["A_geometry"]["isoscore_mean"],
                out["scales"]["25x25"]["A_geometry"]["effective_rank_mean"],
                out["scales"]["25x25"]["A_geometry"]["dim"],
                out["scales"]["50x50"]["A_geometry"]["effective_rank_mean"],
                out["scales"]["50x50"]["A_geometry"]["dim"])),
        "claim_2_gate_saturates": (
            "{:.0%} / {:.0%} of query states score sim_max > 0.95.".format(
                out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["raw"]["saturation_frac_above_0.95"],
                out["scales"]["50x50"]["B_retrieval_gate"]["treatments"]["raw"]["saturation_frac_above_0.95"])),
        "claim_3_retrieval_near_random": (
            "retrieval quality {:.3f} / {:.3f} where 1.0 = nearest stored state "
            "and 0.0 = a random stored state.".format(
                out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["raw"]["retrieval_quality"],
                out["scales"]["50x50"]["B_retrieval_gate"]["treatments"]["raw"]["retrieval_quality"])),
        "claim_4_posthoc_fixes_saturation_not_quality": (
            "whitening drives saturation {:.0%} -> {:.0%} (25x25) but changes "
            "retrieval quality by {:+.4f} (p={:.2f}).".format(
                out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["raw"]["saturation_frac_above_0.95"],
                out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["whitened"]["saturation_frac_above_0.95"],
                out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["whitened"]["retrieval_quality"]
                - out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["raw"]["retrieval_quality"],
                out["scales"]["25x25"]["B_retrieval_gate"]["treatments"]["whitened"]["quality_vs_raw_p"])),
        "claim_5_information_absent_not_misaligned": (
            "a learned linear metric on frozen embeddings lifts locality rho "
            "{:.3f}->{:.3f} (25x25) and {:.3f}->{:.3f} (50x50): better than "
            "cosine, but far short of usable.".format(
                out["scales"]["25x25"]["C_metric_probe"]["cosine"],
                out["scales"]["25x25"]["C_metric_probe"]["ridge"],
                out["scales"]["50x50"]["C_metric_probe"]["cosine"],
                out["scales"]["50x50"]["C_metric_probe"]["ridge"])),
    }

    p = RUNS / "geometry_study_aggregate.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["headline"], indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
