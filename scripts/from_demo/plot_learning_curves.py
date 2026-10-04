"""Learning curves from per-episode training traces (measured data only).

Reads <trace-root>/seed<seed>_<scenario_id>/<arm>/episodes.jsonl (written by
train_gnn_dqn when trace_dir is set) and produces:

  - a JSON file with the plotted series (per-episode mean and 95% t-CI across
    cells) plus caption metadata, and
  - a two-panel matplotlib PDF:
      (a) raw episode return — mean +/- 95% CI band across cells, NO smoothing
      (b) goal-reach rate — per-cell rolling mean over a stated window (full
          windows only), then mean +/- 95% CI across cells

The only smoothing anywhere is the stated rolling window in panel (b); the
window size is recorded in the caption metadata and the figure caption.

Usage:
    uv run python scripts/plot_learning_curves.py                  # 25x25 defaults
    uv run python scripts/plot_learning_curves.py --trace-root runs/traces_50x50 \\
        --out-json runs/learning_curves_50x50.json \\
        --out-fig runs/figs/learning_curves_50x50.pdf
    uv run python scripts/plot_learning_curves.py --arms warm cold classical_only quantum_only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from scipy import stats as sps

ARM_LABELS = {
    "warm": "warm (full oracle pool)",
    "cold": "cold (no demonstrations)",
    "classical_only": "classical-only demonstrations",
    "quantum_only": "quantum-only demonstrations",
}
ARM_COLORS = {
    "warm": "#1f77b4",
    "cold": "#d62728",
    "classical_only": "#2ca02c",
    "quantum_only": "#9467bd",
}


def load_arm(trace_root: pathlib.Path, arm: str) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Stack per-cell episode series for one arm.

    Returns (returns[n_cells, T], goals[n_cells, T], cell_names,
    iteration_boundaries) where T is the minimum common episode count and the
    boundaries are the first global episode of each training iteration,
    measured from the traces themselves.
    """
    cells, returns, goals, boundaries = [], [], [], []
    for ep_file in sorted(trace_root.glob(f"*/{arm}/episodes.jsonl")):
        recs = [json.loads(line) for line in open(ep_file)]
        if not recs:
            continue
        cells.append(ep_file.parent.parent.name)
        returns.append([r["return"] for r in recs])
        goals.append([float(r["reached_goal"]) for r in recs])
        bnds = [i for i, r in enumerate(recs)
                if i == 0 or r["iteration"] != recs[i - 1]["iteration"]]
        if len(bnds) > len(boundaries):
            boundaries = bnds
    if not cells:
        return np.empty((0, 0)), np.empty((0, 0)), [], []
    # A complete run has the full episode count; shorter traces are crashed or
    # still-running cells and must not enter (or truncate) a measured figure.
    t_full = max(len(r) for r in returns)
    keep = [i for i, r in enumerate(returns) if len(r) == t_full]
    for i, r in enumerate(returns):
        if len(r) != t_full:
            print(f"  [{arm}] DROPPED incomplete cell {cells[i]} "
                  f"({len(r)}/{t_full} episodes)")
    cells = [cells[i] for i in keep]
    returns = np.array([returns[i] for i in keep])
    goals = np.array([goals[i] for i in keep])
    return returns, goals, cells, [b for b in boundaries if b < t_full]


def mean_ci(series: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Across-cell mean and t-based CI band, per episode. series: [n_cells, T]."""
    n = series.shape[0]
    mean = series.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = series.std(axis=0, ddof=1) / np.sqrt(n)
    half = sps.t.ppf(0.5 + confidence / 2, n - 1) * sem
    return mean, mean - half, mean + half


def rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    """Per-cell rolling mean, full windows only. [n_cells, T] -> [n_cells, T-window+1]."""
    kernel = np.ones(window) / window
    return np.array([np.convolve(row, kernel, mode="valid") for row in series])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=str, default="runs/traces_25x25")
    parser.add_argument("--out-json", type=str, default="runs/learning_curves_25x25.json")
    parser.add_argument("--out-fig", type=str, default="runs/figs/learning_curves_25x25.pdf")
    parser.add_argument("--arms", nargs="+", default=["warm", "cold"])
    parser.add_argument("--window", type=int, default=25,
                        help="Rolling window (episodes) for the goal-reach panel.")
    parser.add_argument("--title", type=str, default="25x25 sweep")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_root = pathlib.Path(args.trace_root)
    out_json = pathlib.Path(args.out_json)
    out_fig = pathlib.Path(args.out_fig)

    arm_data = {}
    for arm in args.arms:
        returns, goals, cells, bounds = load_arm(trace_root, arm)
        if len(cells) == 0:
            print(f"  [{arm}] no traces under {trace_root} — skipped")
            continue
        print(f"  [{arm}] {len(cells)} cells x {returns.shape[1]} episodes")
        arm_data[arm] = (returns, goals, cells, bounds)
    if not arm_data:
        raise SystemExit(f"No episode traces found under {trace_root}")

    n_cells_max = max(len(d[2]) for d in arm_data.values())
    grid_note = " (5 seeds × 5 scenarios)" if n_cells_max == 25 else ""
    caption = (
        f"Per-episode training curves, mean ± 95% t-CI across "
        f"{n_cells_max} cells{grid_note}. "
        f"Panel (a): raw episode return, no smoothing. "
        f"Panel (b): goal-reach rate, per-cell rolling mean over a "
        f"{args.window}-episode window (full windows only), then averaged "
        f"across cells. Vertical dashed lines mark training-iteration "
        f"boundaries measured from the traces."
    )

    payload = {
        "metadata": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace_root": str(trace_root),
            "rolling_window_episodes": args.window,
            "ci": "95% t-distribution across cells",
            "smoothing": f"none in panel (a); rolling window of {args.window} "
                         f"episodes in panel (b) only",
            "caption": caption,
        },
        "arms": {},
    }

    fig, (ax_ret, ax_goal) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, (returns, goals, cells, bounds) in arm_data.items():
        color = ARM_COLORS.get(arm, None)
        label = ARM_LABELS.get(arm, arm)
        t = np.arange(returns.shape[1])

        r_mean, r_lo, r_hi = mean_ci(returns)
        ax_ret.plot(t, r_mean, color=color, label=label, linewidth=1.0)
        ax_ret.fill_between(t, r_lo, r_hi, color=color, alpha=0.18, linewidth=0)

        g_roll = rolling_mean(goals, args.window)
        tg = np.arange(args.window - 1, args.window - 1 + g_roll.shape[1])
        g_mean, g_lo, g_hi = mean_ci(g_roll)
        ax_goal.plot(tg, g_mean, color=color, label=label, linewidth=1.4)
        ax_goal.fill_between(tg, g_lo, g_hi, color=color, alpha=0.18, linewidth=0)

        payload["arms"][arm] = {
            "n_cells": len(cells),
            "cells": cells,
            "n_episodes": int(returns.shape[1]),
            "iteration_boundaries": [int(b) for b in bounds],
            "episode": t.tolist(),
            "return_mean": r_mean.tolist(),
            "return_ci_low": r_lo.tolist(),
            "return_ci_high": r_hi.tolist(),
            "goal_episode": tg.tolist(),
            "goal_rolling_mean": g_mean.tolist(),
            "goal_rolling_ci_low": g_lo.tolist(),
            "goal_rolling_ci_high": g_hi.tolist(),
        }

    bounds = max((d[3] for d in arm_data.values()), key=len)
    for ax in (ax_ret, ax_goal):
        for b in bounds[1:]:
            ax.axvline(b, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_xlabel("training episode")
        ax.legend(fontsize=8)
    ax_ret.set_ylabel("episode return")
    ax_ret.set_title("(a) episode return (raw, mean ± 95% CI)", fontsize=10)
    ax_goal.set_ylabel(f"goal-reach rate (rolling {args.window}-episode window)")
    ax_goal.set_ylim(-0.02, 1.02)
    ax_goal.set_title("(b) rolling goal-reach rate (mean ± 95% CI)", fontsize=10)
    fig.suptitle(f"Learning curves — {args.title}", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.01, caption, fontsize=6.5, wrap=True)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, metadata={"Subject": caption,
                                   "Title": f"Learning curves — {args.title}"})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh)
    print(f"\n  Figure: {out_fig}\n  Data:   {out_json}")


if __name__ == "__main__":
    main()
