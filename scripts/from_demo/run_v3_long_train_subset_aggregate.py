"""Aggregate V3 long-train subset results and run statistical tests.

Reads runs/v3_long_train_subset_results.json (output of run_v3_long_train_subset_evaluation.py)
and produces:
  - Aggregate table (V1 / V2 / V3) across the 10 successfully-trained cells
  - McNemar's test: V1 vs V3 reach, V1 vs V2 reach (paired binary, n=10)
  - Paired t-test: V1 vs V3 cost (cells where both reached)
  - One-sample t-test: mean lam_eff at 4x compute vs prior 1x value (0.365)

Output:
  runs/v3_long_train_subset_aggregate.json

CLI:
    uv run python scripts/run_v3_long_train_subset_aggregate.py
"""
from __future__ import annotations

import json
import math
import pathlib

import scipy.stats as stats
from scipy.stats import binomtest

RESULTS_PATH    = pathlib.Path("runs/v3_long_train_subset_results.json")
AGGREGATE_PATH  = pathlib.Path("runs/v3_long_train_subset_aggregate.json")

# Published V3 lam_eff at 1x compute from the single-cell sweep (thesis baseline)
LAMBDA_EFF_1X = 0.365


def _mcnemar_p(b: int, c: int) -> float:
    """Exact McNemar's p-value using binomial test on discordant pairs."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    # Two-sided exact test: how often would max(B,C) >= max(b,c) under H0 (p=0.5)?
    k = max(b, c)
    result = binomtest(k, n_disc, 0.5, alternative="greater")
    return min(1.0, 2 * result.pvalue)  # two-sided


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}\n"
                                f"Run run_v3_long_train_subset_evaluation.py first.")

    with open(RESULTS_PATH) as fh:
        records = json.load(fh)

    # Separate skipped from evaluated cells
    ok_records = [r for r in records if r.get("status") == "ok"]
    skipped    = [r for r in records if r.get("status") == "skipped_4x_failed"]

    # Index by (cell_idx, mode)
    by_cell: dict[int, dict[str, dict]] = {}
    for r in ok_records:
        idx  = r["cell_idx"]
        mode = r["mode"]
        by_cell.setdefault(idx, {})[mode] = r

    # Only cells that have all three modes evaluated
    complete_cells = [idx for idx, modes in by_cell.items() if all(m in modes for m in ("V1", "V2", "V3"))]
    complete_cells.sort()
    n = len(complete_cells)

    print(f"\nV3 Long-Train Subset Aggregate  ({n} complete cells)")
    print(f"Skipped (4x failed): {len(skipped)} cells")
    print()

    # ── Per-mode aggregates ─────────────────────────────────────────────────────

    def _mode_stats(mode: str) -> dict:
        rows = [by_cell[idx][mode] for idx in complete_cells]
        reach       = sum(1 for r in rows if r.get("reached_goal"))
        costs       = [r["total_cost"] for r in rows if r.get("reached_goal") and r.get("total_cost") is not None]
        ratios      = [r["cost_ratio_vs_dijkstra"] for r in rows if r.get("reached_goal") and r.get("cost_ratio_vs_dijkstra") is not None]
        mean_cost   = sum(costs) / len(costs) if costs else None
        mean_ratio  = sum(ratios) / len(ratios) if ratios else None
        result = {
            "n_cells": n,
            "n_reached": reach,
            "reach_rate": reach / n,
            "mean_cost_reached": mean_cost,
            "mean_cost_ratio_reached": mean_ratio,
        }
        if mode in ("V2", "V3"):
            sims = [r.get("mean_max_similarity") for r in rows if r.get("mean_max_similarity") is not None]
            result["mean_max_similarity"] = sum(sims) / len(sims) if sims else None
        if mode == "V3":
            lambdas = [r.get("mean_lambda_effective") for r in rows if r.get("mean_lambda_effective") is not None]
            result["mean_lambda_effective"] = sum(lambdas) / len(lambdas) if lambdas else None
        return result

    v1_stats = _mode_stats("V1")
    v2_stats = _mode_stats("V2")
    v3_stats = _mode_stats("V3")

    # ── Print table ─────────────────────────────────────────────────────────────

    def _fmt(v, fmt=".1f"):
        return "n/a" if v is None else format(v, fmt)

    lam_v3 = v3_stats.get("mean_lambda_effective")

    print(f"{'Mode':<8} {'reach':>8} {'mean cost':>12} {'mean ratio':>12} {'mean lam_eff':>14} {'mean max-sim':>14}")
    print("-" * 68)
    print(f"{'V1':<8} {v1_stats['n_reached']:>3}/{n:<4} {_fmt(v1_stats['mean_cost_reached']):>12} "
          f"{_fmt(v1_stats['mean_cost_ratio_reached']):>12} {'n/a':>12} {'n/a':>14}")
    print(f"{'V2':<8} {v2_stats['n_reached']:>3}/{n:<4} {_fmt(v2_stats['mean_cost_reached']):>12} "
          f"{_fmt(v2_stats['mean_cost_ratio_reached']):>12} {'0.500':>12} {_fmt(v2_stats.get('mean_max_similarity')):>14}")
    print(f"{'V3':<8} {v3_stats['n_reached']:>3}/{n:<4} {_fmt(v3_stats['mean_cost_reached']):>12} "
          f"{_fmt(v3_stats['mean_cost_ratio_reached']):>12} {_fmt(lam_v3, '.3f'):>12} {_fmt(v3_stats.get('mean_max_similarity')):>14}")
    print()

    # ── Statistical tests ───────────────────────────────────────────────────────

    def _reach_vec(mode: str) -> list[bool]:
        return [bool(by_cell[idx][mode].get("reached_goal")) for idx in complete_cells]

    v1_reach = _reach_vec("V1")
    v2_reach = _reach_vec("V2")
    v3_reach = _reach_vec("V3")

    # McNemar's V1 vs V3
    b_v1v3 = sum(1 for v1r, v3r in zip(v1_reach, v3_reach) if not v1r and v3r)
    c_v1v3 = sum(1 for v1r, v3r in zip(v1_reach, v3_reach) if v1r and not v3r)
    p_mcnemar_v1v3 = _mcnemar_p(b_v1v3, c_v1v3)

    # McNemar's V1 vs V2
    b_v1v2 = sum(1 for v1r, v2r in zip(v1_reach, v2_reach) if not v1r and v2r)
    c_v1v2 = sum(1 for v1r, v2r in zip(v1_reach, v2_reach) if v1r and not v2r)
    p_mcnemar_v1v2 = _mcnemar_p(b_v1v2, c_v1v2)

    # Paired t-test V1 vs V3 cost (cells where both reached)
    v1_costs_paired, v3_costs_paired = [], []
    for idx in complete_cells:
        r1 = by_cell[idx]["V1"]
        r3 = by_cell[idx]["V3"]
        if r1.get("reached_goal") and r3.get("reached_goal"):
            if r1.get("total_cost") is not None and r3.get("total_cost") is not None:
                v1_costs_paired.append(r1["total_cost"])
                v3_costs_paired.append(r3["total_cost"])

    p_paired_ttest = None
    tstat_paired   = None
    n_paired       = len(v1_costs_paired)
    if n_paired >= 2:
        tstat_paired, p_paired_ttest = stats.ttest_rel(v1_costs_paired, v3_costs_paired)

    # One-sample t-test: lam_eff < 0.365
    lambda_vals = [by_cell[idx]["V3"].get("mean_lambda_effective") for idx in complete_cells
                   if by_cell[idx]["V3"].get("mean_lambda_effective") is not None]
    p_onesample_lambda = None
    tstat_lambda       = None
    mean_lambda_obs    = sum(lambda_vals) / len(lambda_vals) if lambda_vals else None
    if len(lambda_vals) >= 2:
        tstat_lambda, p_onesample_lambda = stats.ttest_1samp(lambda_vals, LAMBDA_EFF_1X,
                                                              alternative="less")

    print("Statistical tests:")
    print(f"  McNemar V1 vs V3 reach: b={b_v1v3} c={c_v1v3}  p={p_mcnemar_v1v3:.4f}")
    print(f"  McNemar V1 vs V2 reach: b={b_v1v2} c={c_v1v2}  p={p_mcnemar_v1v2:.4f}")
    if p_paired_ttest is not None:
        print(f"  Paired t-test V1 vs V3 cost (n={n_paired}): t={tstat_paired:.3f}  p={p_paired_ttest:.4f}")
    else:
        print(f"  Paired t-test V1 vs V3 cost: insufficient paired data (n={n_paired})")
    if p_onesample_lambda is not None:
        print(f"  One-sample t-test lam_eff < {LAMBDA_EFF_1X}: t={tstat_lambda:.3f}  p={p_onesample_lambda:.4f}")
        print(f"    (observed mean lam_eff={mean_lambda_obs:.3f}, H0=0.365, alt='less')")
    else:
        print(f"  One-sample t-test lam_eff: insufficient data (n={len(lambda_vals)})")
    print()

    # ── Interpretation hints ────────────────────────────────────────────────────

    hints_fired: list[str] = []

    if lam_v3 is not None and p_onesample_lambda is not None:
        pct_reduction = (LAMBDA_EFF_1X - lam_v3) / LAMBDA_EFF_1X * 100 if lam_v3 < LAMBDA_EFF_1X else 0.0

        if lam_v3 <= 0.15 and p_onesample_lambda < 0.01:
            h = (f"V3 self-attenuation confirmed at n={n}. Mean adaptive lam falls from "
                 f"{LAMBDA_EFF_1X} (1x compute) to {lam_v3:.3f} (4x compute), a "
                 f"{pct_reduction:.0f}% reduction, statistically significant (p={p_onesample_lambda:.4f}). "
                 f"The regime-adaptive property holds across the long-train subset, not just the single-cell case.")
            hints_fired.append(h)
        elif 0.15 < lam_v3 < 0.30:
            h = (f"Partial V3 self-attenuation at n={n}: mean lam_eff falls from {LAMBDA_EFF_1X} to "
                 f"{lam_v3:.3f}. Direction correct but magnitude smaller than the single-cell case "
                 f"(0.31 -> 0.10) suggested. Further per-cell analysis recommended.")
            hints_fired.append(h)
        elif lam_v3 >= 0.30:
            h = (f"V3 self-attenuation NOT confirmed at n={n}: mean lam_eff remains at {lam_v3:.3f}. "
                 f"The single-cell finding (0.31 -> 0.10) does not generalise. The mechanism's "
                 f"regime-adaptive property is weaker than initially suggested.")
            hints_fired.append(h)

    # V2 catastrophic interference
    v2_worse_cells = sum(1 for v1r, v2r in zip(v1_reach, v2_reach) if v1r and not v2r)
    if v2_worse_cells >= 3:
        h = (f"V2 fixed-lam retrieval catastrophically interferes with the long-trained policy on "
             f"{v2_worse_cells}/{n} cells. The adaptive mechanism is necessary, not just nice-to-have.")
        hints_fired.append(h)

    # V3 safety property
    if all(v3r >= v1r for v1r, v3r in zip(v1_reach, v3_reach)):
        h = (f"Safety property V3 >= V1 confirmed at n={n} across the long-train subset. "
             f"V3 never reduces goal-reach when the policy is competent.")
        hints_fired.append(h)

    if hints_fired:
        print("Interpretation:")
        for h in hints_fired:
            print(f"  * {h}")
        print()

    # ── Save aggregate ──────────────────────────────────────────────────────────

    aggregate = {
        "n_cells_evaluated": n,
        "n_cells_skipped_4x_failed": len(skipped),
        "V1": v1_stats,
        "V2": v2_stats,
        "V3": v3_stats,
        "statistics": {
            "mcnemar_v1_vs_v3": {
                "b_v1_failed_v3_reached": b_v1v3,
                "c_v1_reached_v3_failed": c_v1v3,
                "p_value": p_mcnemar_v1v3,
            },
            "mcnemar_v1_vs_v2": {
                "b_v1_failed_v2_reached": b_v1v2,
                "c_v1_reached_v2_failed": c_v1v2,
                "p_value": p_mcnemar_v1v2,
            },
            "paired_ttest_v1_vs_v3_cost": {
                "n_paired": n_paired,
                "t_statistic": tstat_paired,
                "p_value": p_paired_ttest,
            },
            "onesample_ttest_lambda_vs_1x_baseline": {
                "h0_lambda": LAMBDA_EFF_1X,
                "observed_mean_lambda": mean_lambda_obs,
                "n_lambda_vals": len(lambda_vals),
                "t_statistic": tstat_lambda,
                "p_value": p_onesample_lambda,
                "alternative": "less",
            },
        },
        "interpretation_hints": hints_fired,
        "per_cell_detail": [
            {
                "cell_idx": idx,
                "seed": by_cell[idx]["V1"]["seed"],
                "scenario_id": by_cell[idx]["V1"]["scenario_id"],
                "V1_reached": by_cell[idx]["V1"].get("reached_goal"),
                "V2_reached": by_cell[idx]["V2"].get("reached_goal"),
                "V3_reached": by_cell[idx]["V3"].get("reached_goal"),
                "V1_cost": by_cell[idx]["V1"].get("total_cost"),
                "V2_cost": by_cell[idx]["V2"].get("total_cost"),
                "V3_cost": by_cell[idx]["V3"].get("total_cost"),
                "V3_mean_lambda": by_cell[idx]["V3"].get("mean_lambda_effective"),
                "V3_mean_max_sim": by_cell[idx]["V3"].get("mean_max_similarity"),
            }
            for idx in complete_cells
        ],
    }

    AGGREGATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGGREGATE_PATH, "w") as fh:
        json.dump(aggregate, fh, indent=2,
                  default=lambda x: None if (isinstance(x, float) and (x != x or abs(x) == float("inf"))) else x)
    print(f"Aggregate saved to {AGGREGATE_PATH}")


if __name__ == "__main__":
    main()
