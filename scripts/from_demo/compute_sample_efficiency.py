"""Sample-efficiency statistics from per-episode training traces (measured data only).

Reads the same <trace-root>/<cell>/<arm>/episodes.jsonl traces as
plot_learning_curves.py and computes:

  1. EPISODES-TO-THRESHOLD per arm: for each cell, the first episode (1-based,
     end of the first full rolling window) at which the rolling goal-reach
     rate exceeds each threshold. Cells that never cross within the budget are
     counted explicitly as "never"; the across-cell median/IQR treats never as
     +inf, so the median is reported only when more than half the cells cross.
  2. EARLY-PHASE ADVANTAGE: per-cell mean return over the first
     --early-episodes episodes, warm vs cold, paired Wilcoxon signed-rank.
  3. EARLY-PHASE MECHANISM check: episodes.jsonl records neither episode
     length nor termination reason, so this block uses the closest honest
     proxies and says so in its output: early goal-reach rate (exact, from
     reached_goal); the fraction of episodes with return == -5.0 (exact:
     in legacy invalid-penalty mode that return arises only from an invalid
     action before any valid step); and non-goal return magnitude as a
     cost-weighted length proxy (lambda_shape=0 in the traced sweep, so
     return of a non-goal episode is -accumulated step cost, minus 5 if
     invalid-terminated). Termination-type fractions (invalid vs step-cap
     vs dead-end) are NOT computable from the current traces.

Both blocks are merged into --out-json (runs/learning_curves_25x25.json by
default) under the keys "sample_efficiency" and "early_phase_advantage",
preserving whatever plot_learning_curves.py already wrote there.

The known-unreachable 25x25 cell (seed=2024, scenario seed518677876_s1, per
runs/eval_reachability_audit.json) is excluded by default; rates are
solvable-only. Incomplete cells (fewer episodes than the longest trace) are
dropped, mirroring plot_learning_curves.py.

Usage:
    uv run python scripts/compute_sample_efficiency.py            # 25x25 defaults
    uv run python scripts/compute_sample_efficiency.py --trace-root runs/traces_50x50 \\
        --out-json runs/learning_curves_50x50.json --exclude-cells
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_learning_curves import load_arm, rolling_mean  # noqa: E402

KNOWN_UNREACHABLE_25X25 = "seed2024_seed518677876_s1"

TERMINATION_TAGS = ("goal", "invalid", "step_cap", "dead_end", "other")


def load_arm_records(trace_root: pathlib.Path, arm: str,
                     exclude: list[str]) -> dict[str, list[dict]]:
    """Raw episode records per cell, complete cells only (same completeness
    rule as load_arm: a cell must have the maximum observed episode count)."""
    cells: dict[str, list[dict]] = {}
    for ep_file in sorted(trace_root.glob(f"*/{arm}/episodes.jsonl")):
        cell = ep_file.parent.parent.name
        if cell in exclude:
            continue
        recs = [json.loads(line) for line in open(ep_file)]
        if recs:
            cells[cell] = recs
    if not cells:
        return {}
    t_full = max(len(r) for r in cells.values())
    return {c: r for c, r in cells.items() if len(r) == t_full}


def episodes_to_threshold(goals: np.ndarray, window: int, threshold: float) -> list[int | None]:
    """First 1-based episode whose full-window rolling goal-reach rate exceeds
    threshold, per cell; None if never within the budget."""
    rolled = rolling_mean(goals, window)
    out: list[int | None] = []
    for row in rolled:
        idx = np.flatnonzero(row > threshold)
        # rolled[i] covers episodes i+1 .. i+window (1-based); crossing episode
        # is the end of that window.
        out.append(int(idx[0]) + window if idx.size else None)
    return out


def median_iqr_with_never(values: list[int | None]) -> dict:
    """Across-cell median/IQR treating never as +inf (quantiles reported only
    when defined, i.e. enough cells crossed), plus reached-only stats."""
    n = len(values)
    arr = np.array([np.inf if v is None else float(v) for v in values])
    n_never = int(np.isinf(arr).sum())
    reached = arr[np.isfinite(arr)]

    def finite_quantile(q: float) -> float | None:
        # Interpolating between two +inf order statistics raises an invalid-op
        # warning and yields nan; either way the quantile is undefined.
        with np.errstate(invalid="ignore"):
            v = np.quantile(arr, q)
        return float(v) if np.isfinite(v) else None

    return {
        "n_cells": n,
        "n_reached": n - n_never,
        "n_never": n_never,
        "median": finite_quantile(0.5),
        "iqr_low": finite_quantile(0.25),
        "iqr_high": finite_quantile(0.75),
        "reached_only_median": float(np.median(reached)) if reached.size else None,
        "reached_only_iqr": [float(np.quantile(reached, 0.25)),
                             float(np.quantile(reached, 0.75))] if reached.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=str, default="runs/traces_25x25")
    parser.add_argument("--out-json", type=str, default="runs/learning_curves_25x25.json")
    parser.add_argument("--arms", nargs="+", default=["warm", "cold"])
    parser.add_argument("--window", type=int, default=25,
                        help="Rolling window (episodes) for goal-reach rate.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--early-episodes", type=int, default=50,
                        help="Early-phase span (episodes 1..N) for the paired return comparison.")
    parser.add_argument("--exclude-cells", nargs="*", default=[KNOWN_UNREACHABLE_25X25],
                        help="Cells excluded from all statistics (default: the "
                             "known-unreachable 25x25 cell from runs/eval_reachability_audit.json).")
    args = parser.parse_args()

    trace_root = pathlib.Path(args.trace_root)
    out_json = pathlib.Path(args.out_json)

    arm_data = {}
    for arm in args.arms:
        returns, goals, cells, _ = load_arm(trace_root, arm)
        excluded = [c for c in cells if c in args.exclude_cells]
        keep = [i for i, c in enumerate(cells) if c not in args.exclude_cells]
        if excluded:
            print(f"  [{arm}] excluded {len(excluded)} cell(s): {', '.join(excluded)}")
        if not keep:
            print(f"  [{arm}] no usable traces under {trace_root} — skipped")
            continue
        arm_data[arm] = (returns[keep], goals[keep], [cells[i] for i in keep])
        print(f"  [{arm}] {len(keep)} cells x {returns.shape[1]} episodes")
    if not arm_data:
        raise SystemExit(f"No episode traces found under {trace_root}")

    budget = max(d[0].shape[1] for d in arm_data.values())

    # --- 1. episodes-to-threshold -------------------------------------------
    sample_efficiency = {
        "definition": (
            f"Per cell, first 1-based episode at which the rolling "
            f"(window={args.window}, full windows only) goal-reach rate "
            f"exceeds the threshold; 'never' = no crossing within the "
            f"{budget}-episode budget. Across-cell median/IQR treat never as "
            f"+inf and are null when fewer than the required share of cells "
            f"crossed; reached_only_* restrict to crossing cells."
        ),
        "window": args.window,
        "episode_budget": int(budget),
        "excluded_cells": args.exclude_cells,
        "arms": {},
    }
    for arm, (returns, goals, cells) in arm_data.items():
        per_threshold = {}
        for thr in args.thresholds:
            firsts = episodes_to_threshold(goals, args.window, thr)
            stats = median_iqr_with_never(firsts)
            stats["per_cell"] = dict(zip(cells, firsts))
            per_threshold[f"{thr:g}"] = stats
        sample_efficiency["arms"][arm] = per_threshold

    # --- 2. early-phase advantage -------------------------------------------
    early = {
        "definition": (
            f"Per-cell mean episode return over episodes 1-{args.early_episodes}, "
            f"paired across arms on common cells; two-sided Wilcoxon signed-rank."
        ),
        "episodes": [1, args.early_episodes],
        "excluded_cells": args.exclude_cells,
        "arm_means": {},
    }
    early_means = {
        arm: dict(zip(cells, returns[:, :args.early_episodes].mean(axis=1)))
        for arm, (returns, goals, cells) in arm_data.items()
    }
    for arm, by_cell in early_means.items():
        early["arm_means"][arm] = {
            "n_cells": len(by_cell),
            "mean": float(np.mean(list(by_cell.values()))),
            "per_cell": {c: float(v) for c, v in by_cell.items()},
        }
    if "warm" in early_means and "cold" in early_means:
        common = sorted(set(early_means["warm"]) & set(early_means["cold"]))
        diffs = np.array([early_means["warm"][c] - early_means["cold"][c] for c in common])
        pair = {
            "n_pairs": len(common),
            "cells": common,
            "mean_diff_warm_minus_cold": float(diffs.mean()),
            "median_diff_warm_minus_cold": float(np.median(diffs)),
        }
        if len(common) >= 2 and np.any(diffs != 0):
            w_stat, p = sps.wilcoxon(diffs)
            pair["wilcoxon_W"] = float(w_stat)
            pair["wilcoxon_p_two_sided"] = float(p)
        else:
            pair["wilcoxon_W"] = pair["wilcoxon_p_two_sided"] = None
        early["paired_warm_vs_cold"] = pair

    # --- 2.5 early-phase mechanism check (proxy-based) -----------------------
    mechanism = {
        "definition": (
            f"Episodes 1-{args.early_episodes}. Traces record no episode "
            f"length or termination reason; proxies used instead: "
            f"goal_rate is exact (reached_goal field); minus5_fraction is "
            f"exact for zero-valid-step invalid terminations (legacy mode: "
            f"return == -5.0 only there); nongoal return magnitude is a "
            f"cost-weighted length proxy (lambda_shape=0, each valid step "
            f"adds a continuous positive cost). Invalid-after-k-steps, "
            f"step-cap and dead-end terminations are NOT distinguishable."
        ),
        "episodes": [1, args.early_episodes],
        "excluded_cells": args.exclude_cells,
        "arms": {},
    }
    mech_cells: dict[str, dict[str, dict[str, float]]] = {}
    for arm, (returns, goals, cells) in arm_data.items():
        r = returns[:, :args.early_episodes]
        g = goals[:, :args.early_episodes].astype(bool)
        per_cell = {}
        for i, cell in enumerate(cells):
            nongoal = r[i][~g[i]]
            per_cell[cell] = {
                "goal_rate": float(g[i].mean()),
                "minus5_fraction": float((r[i] == -5.0).mean()),
                "nongoal_return_mean": float(nongoal.mean()) if nongoal.size else None,
            }
        mech_cells[arm] = per_cell
        pooled_nongoal = r[~g]
        mechanism["arms"][arm] = {
            "n_cells": len(cells),
            "goal_rate_mean": float(g.mean()),
            "minus5_fraction_mean": float((r == -5.0).mean()),
            "nongoal_return_quartiles": [float(q) for q in
                                         np.quantile(pooled_nongoal, [0.25, 0.5, 0.75])]
                                        if pooled_nongoal.size else None,
            "per_cell": per_cell,
        }

    # Exact stats for cells whose traces record steps/termination (written
    # after the schema extension). Older cells are covered by the proxies
    # above; both groups are labeled in the output.
    exact_cells: dict[str, dict[str, dict]] = {}
    for arm in arm_data:
        recs_by_cell = load_arm_records(trace_root, arm, args.exclude_cells)
        per = {}
        for cell, recs in recs_by_cell.items():
            early_recs = recs[:args.early_episodes]
            if not early_recs or "termination" not in early_recs[0]:
                continue
            n = len(early_recs)
            per[cell] = {
                "mean_steps": float(np.mean([rec["steps"] for rec in early_recs])),
                "termination_fractions": {
                    tag: sum(rec["termination"] == tag for rec in early_recs) / n
                    for tag in TERMINATION_TAGS
                },
                "goal_rate": sum(bool(rec["reached_goal"]) for rec in early_recs) / n,
            }
        if per:
            exact_cells[arm] = per
            mechanism["arms"][arm]["exact"] = {
                "n_cells_with_fields": len(per),
                "mean_steps": float(np.mean([v["mean_steps"] for v in per.values()])),
                "termination_fractions_mean": {
                    tag: float(np.mean([v["termination_fractions"][tag]
                                        for v in per.values()]))
                    for tag in TERMINATION_TAGS
                },
                "per_cell": per,
            }
    if "warm" in mech_cells and "cold" in mech_cells:
        common = sorted(set(mech_cells["warm"]) & set(mech_cells["cold"]))
        paired_ok = [c for c in common
                     if mech_cells["warm"][c]["nongoal_return_mean"] is not None
                     and mech_cells["cold"][c]["nongoal_return_mean"] is not None]
        cost_diffs = np.array([  # positive = warm accumulates more step cost
            -mech_cells["warm"][c]["nongoal_return_mean"]
            - (-mech_cells["cold"][c]["nongoal_return_mean"]) for c in paired_ok])
        goal_diffs = np.array([mech_cells["warm"][c]["goal_rate"]
                               - mech_cells["cold"][c]["goal_rate"] for c in common])
        paired = {
            "n_pairs_cost": len(paired_ok),
            "mean_nongoal_cost_diff_warm_minus_cold": float(cost_diffs.mean()),
            "n_pairs_goal": len(common),
            "mean_goal_rate_diff_warm_minus_cold": float(goal_diffs.mean()),
        }
        if len(cost_diffs) >= 2 and np.any(cost_diffs != 0):
            w, p = sps.wilcoxon(cost_diffs)
            paired["cost_wilcoxon_W"], paired["cost_wilcoxon_p_two_sided"] = float(w), float(p)
        else:
            paired["cost_wilcoxon_W"] = paired["cost_wilcoxon_p_two_sided"] = None
        if len(goal_diffs) >= 2 and np.any(goal_diffs != 0):
            w, p = sps.wilcoxon(goal_diffs)
            paired["goal_wilcoxon_W"], paired["goal_wilcoxon_p_two_sided"] = float(w), float(p)
        else:
            paired["goal_wilcoxon_W"] = paired["goal_wilcoxon_p_two_sided"] = None
        mechanism["paired_warm_vs_cold"] = paired

        # Exact paired stats on the subset of cells where both arms have the
        # steps/termination fields.
        exact_common = sorted(set(exact_cells.get("warm", {}))
                              & set(exact_cells.get("cold", {})))
        paired_exact = None
        if len(exact_common) >= 2:
            step_diffs = np.array([
                exact_cells["warm"][c]["mean_steps"]
                - exact_cells["cold"][c]["mean_steps"] for c in exact_common])
            inv_diffs = np.array([
                exact_cells["warm"][c]["termination_fractions"]["invalid"]
                - exact_cells["cold"][c]["termination_fractions"]["invalid"]
                for c in exact_common])
            paired_exact = {
                "n_pairs": len(exact_common),
                "cells": exact_common,
                "mean_steps_diff_warm_minus_cold": float(step_diffs.mean()),
                "mean_invalid_fraction_diff_warm_minus_cold": float(inv_diffs.mean()),
            }
            if np.any(step_diffs != 0):
                w, p = sps.wilcoxon(step_diffs)
                paired_exact["steps_wilcoxon_W"] = float(w)
                paired_exact["steps_wilcoxon_p_two_sided"] = float(p)
            else:
                paired_exact["steps_wilcoxon_W"] = None
                paired_exact["steps_wilcoxon_p_two_sided"] = None
            mechanism["paired_warm_vs_cold_exact"] = paired_exact

        # Verdict basis: exact fields when every paired cell has them,
        # otherwise the proxies (labeled either way).
        if paired_exact is not None and set(exact_common) == set(common):
            basis = "exact (steps/termination fields)"
            warm_longer = (paired_exact["steps_wilcoxon_p_two_sided"] is not None
                           and paired_exact["steps_wilcoxon_p_two_sided"] < 0.05
                           and paired_exact["mean_steps_diff_warm_minus_cold"] > 0)
            longer_claim = ("warm's early episodes take significantly more "
                            "steps than cold's")
        else:
            basis = "proxy-based"
            warm_longer = (paired["cost_wilcoxon_p_two_sided"] is not None
                           and paired["cost_wilcoxon_p_two_sided"] < 0.05
                           and paired["mean_nongoal_cost_diff_warm_minus_cold"] > 0)
            longer_claim = ("warm's early non-goal episodes accumulate "
                            "significantly more step cost than cold's")
        goal_not_worse = paired["mean_goal_rate_diff_warm_minus_cold"] >= 0
        if warm_longer and goal_not_worse:
            verdict = (
                f"CONFOUNDED ({basis}): {longer_claim} while its early "
                f"goal-reach rate is not lower — consistent with cold's "
                f"early-return edge coming from cheap terminations rather than "
                f"better behavior. Report early-phase progress (goal-reach), "
                f"not return."
            )
            if basis == "proxy-based":
                verdict += (" Confirm with explicit length/termination fields "
                            "before publishing the mechanism claim.")
        elif warm_longer:
            verdict = (
                f"PARTIAL ({basis}): {longer_claim}, but its early goal-reach "
                f"rate is below cold's, so the early-return deficit cannot be "
                f"attributed to reward design alone."
            )
        else:
            verdict = (
                f"NOT SUPPORTED ({basis}): warm's early episodes are not "
                f"significantly longer/costlier than cold's; the early-return "
                f"comparison stands as measured."
            )
        mechanism["verdict"] = verdict
        mechanism["verdict_basis"] = basis

    # --- 3. merge into out-json + summary table ------------------------------
    payload = {}
    if out_json.exists():
        payload = json.loads(out_json.read_text())
    payload.setdefault("metadata", {})["sample_efficiency_generated"] = (
        time.strftime("%Y-%m-%d %H:%M:%S"))
    payload["sample_efficiency"] = sample_efficiency
    payload["early_phase_advantage"] = early
    payload["early_phase_mechanism"] = mechanism
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    print(f"\n  Appended sample_efficiency + early_phase_advantage + "
          f"early_phase_mechanism to {out_json}\n")

    print(f"  Episodes to rolling goal-reach threshold (window={args.window}, "
          f"budget={budget} episodes)")
    print(f"  {'arm':<10} {'thr':>5} {'median [IQR]':>22} {'reached':>9} {'never':>7}")
    for arm, per_threshold in sample_efficiency["arms"].items():
        for thr, s in per_threshold.items():
            med = (f"{s['median']:.0f} [{s['iqr_low']:.0f}-{s['iqr_high']:.0f}]"
                   if s["median"] is not None and s["iqr_high"] is not None
                   else (f"{s['median']:.0f} [IQR undefined]" if s["median"] is not None
                         else "never (median)"))
            print(f"  {arm:<10} {float(thr):>4.0%} {med:>22} "
                  f"{s['n_reached']:>4}/{s['n_cells']:<4} {s['n_never']:>5}")
    print(f"\n  Early-phase mean return, episodes 1-{args.early_episodes}")
    for arm, s in early["arm_means"].items():
        print(f"  {arm:<10} mean={s['mean']:.2f}  (n={s['n_cells']})")
    if "paired_warm_vs_cold" in early:
        p = early["paired_warm_vs_cold"]
        pv = (f"{p['wilcoxon_p_two_sided']:.4g}"
              if p["wilcoxon_p_two_sided"] is not None else "n/a")
        print(f"  paired diff (warm-cold): mean={p['mean_diff_warm_minus_cold']:.2f} "
              f"median={p['median_diff_warm_minus_cold']:.2f} "
              f"Wilcoxon W={p['wilcoxon_W']} p={pv} (n={p['n_pairs']})")

    print(f"\n  Early-phase mechanism (episodes 1-{args.early_episodes}; "
          f"proxies for cells without steps/termination fields, "
          f"EXACT lines where recorded)")
    print(f"  {'arm':<10} {'goal rate':>10} {'==-5.0 frac':>12} "
          f"{'non-goal return Q1/med/Q3':>28}")
    for arm, s in mechanism["arms"].items():
        q = s["nongoal_return_quartiles"]
        qs = f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}" if q else "n/a"
        print(f"  {arm:<10} {s['goal_rate_mean']:>10.3f} "
              f"{s['minus5_fraction_mean']:>12.3f} {qs:>28}")
    for arm, s in mechanism["arms"].items():
        if "exact" in s:
            e = s["exact"]
            tf = e["termination_fractions_mean"]
            tf_str = " ".join(f"{tag}={tf[tag]:.2f}" for tag in TERMINATION_TAGS
                              if tf[tag] > 0)
            print(f"  {arm:<10} EXACT (n={e['n_cells_with_fields']} cells with "
                  f"fields): mean steps={e['mean_steps']:.1f}  {tf_str}")
    if "paired_warm_vs_cold_exact" in mechanism:
        m = mechanism["paired_warm_vs_cold_exact"]
        sp = (f"{m['steps_wilcoxon_p_two_sided']:.4g}"
              if m["steps_wilcoxon_p_two_sided"] is not None else "n/a")
        print(f"  paired steps diff (warm-cold, exact): "
              f"mean={m['mean_steps_diff_warm_minus_cold']:.1f} p={sp} "
              f"(n={m['n_pairs']}); invalid-frac diff "
              f"{m['mean_invalid_fraction_diff_warm_minus_cold']:+.3f}")
    if "paired_warm_vs_cold" in mechanism:
        m = mechanism["paired_warm_vs_cold"]
        cp = (f"{m['cost_wilcoxon_p_two_sided']:.4g}"
              if m["cost_wilcoxon_p_two_sided"] is not None else "n/a")
        gp = (f"{m['goal_wilcoxon_p_two_sided']:.4g}"
              if m["goal_wilcoxon_p_two_sided"] is not None else "n/a")
        print(f"  paired non-goal cost diff (warm-cold): "
              f"mean={m['mean_nongoal_cost_diff_warm_minus_cold']:.2f} p={cp} "
              f"(n={m['n_pairs_cost']})")
        print(f"  paired goal-rate diff (warm-cold): "
              f"mean={m['mean_goal_rate_diff_warm_minus_cold']:.3f} p={gp} "
              f"(n={m['n_pairs_goal']})")
        print(f"\n  VERDICT: {mechanism['verdict']}")


if __name__ == "__main__":
    main()
