"""Re-analysis of all paired warm-vs-cold statistics across scales.

READ-ONLY on every input JSON: this script trains nothing, writes only
runs/stats_reanalysis.json, and never touches runs/demo_source_ablation*,
runs/sweep_phase3_traced.json, or runs/traces_25x25.

Infinite-cost rule (confirmed, matches the original analysis code):
  * cost = +inf for any cell where the agent fails to reach the goal
    (encoded as JSON ``Infinity`` in the 25x25/100x100 sweeps and the
    long-train subset, and as ``null`` in the 50x50 sweep and the cold-4x
    control -- both are mapped to +inf here).
  * win  = strict ``warm_cost < cold_cost``; a both-fail cell gives
    inf < inf = False, i.e. it counts as a NON-win and STAYS in the
    denominator.
  * Original paired t-test conventions differ by scale and are reproduced
    exactly per comparison:
      - 25x25 / 100x100 sweeps (``paired_seed_test`` in
        src/qwarm/eval/statistics.py): inf replaced by finite_max x 10,
        all N pairs kept.
      - 50x50 sweep (``_paired_t`` in scripts/run_sweep_50x50.py) and the
        unified benchmark (``paired_ttest`` in
        scripts/run_unified_benchmark_100x100.py): pairs dropped unless
        BOTH costs are finite (< 1e14, not None).
      - long-train 15-cell vs cold-4x control: the original analysis used
        continuity-corrected McNemar on reach only (no cost t-test); the
        t-test reported here is new, computed with the paired_seed_test
        convention.
  * Wilcoxon signed-rank and the bootstrap are computed uniformly:
      - Wilcoxon: inf replaced by finite_max x 10 (rank-preserving for any
        substitute larger than every finite cost), scipy default
        zero_method='wilcox' (both-fail cells give a zero diff and drop).
      - Bootstrap (10k resamples, seed 12345): per-cell diff
        d = warm - cold with single-fail giving +/-inf and both-fail
        giving 0 (a tie, consistent with "both-fail = non-win");
        percentile 95% CI on the MEDIAN diff (median is rank-based, so
        +/-inf entries are handled exactly).

Holm-Bonferroni is applied per scale across the comparisons feeding the
M1-M3 gate family at that scale, at the M3 threshold alpha = 0.01.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy import stats as scipy_stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_PATH = RUNS / "stats_reanalysis.json"

ALPHA = 0.01           # M3 gate threshold
WIN_GATE = 0.80        # M1 gate threshold
N_BOOT = 10_000
BOOT_SEED = 12345

INF_RULE = (
    "cost=inf for non-reach (null/Infinity in source JSONs both map to inf); "
    "win = strict warm_cost < cold_cost; both-fail -> inf < inf = False, "
    "counted as non-win and kept in the denominator"
)


# ── IO helpers ────────────────────────────────────────────────────────────────

def _load(rel: str, expect_fields: "set[str] | None" = None,
          sample_of_list: bool = True):
    """Load a JSON file read-only and verify expected fields.

    If the schema differs from expectation, report the actual fields and
    abort rather than guessing (sanity constraint)."""
    path = ROOT / rel
    with open(path) as fh:
        data = json.load(fh)
    if expect_fields:
        probe = data[0] if (sample_of_list and isinstance(data, list)) else data
        missing = expect_fields - set(probe)
        if missing:
            print(f"SCHEMA MISMATCH in {rel}:")
            print(f"  expected-but-missing fields: {sorted(missing)}")
            print(f"  actual fields: {sorted(probe)}")
            sys.exit(1)
    return data


def _fin(v) -> float:
    """None -> +inf (the original 50x50 ``_fin``); inf passes through."""
    return float("inf") if v is None else float(v)


# ── Original-method t-tests (reproduced exactly) ──────────────────────────────

def t_paired_seed_test(warm: list[float], cold: list[float]):
    """25x25/100x100 convention: inf -> finite_max*10, keep all pairs."""
    a = np.array(warm, dtype=float)
    b = np.array(cold, dtype=float)
    finite = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    finite_max = max(float(finite.max()) if finite.size else 0.0, 1.0)
    a = np.where(np.isinf(a), finite_max * 10, a)
    b = np.where(np.isinf(b), finite_max * 10, b)
    if len(a) < 2:
        return None, None, len(a)
    t, p = scipy_stats.ttest_rel(a, b)
    t, p = float(t), float(p)
    if np.isnan(p):
        return None, None, len(a)
    return t, p, len(a)


def t_drop_nonfinite(warm: list[float], cold: list[float]):
    """50x50 / unified convention: keep only pairs with BOTH costs finite."""
    pairs = [(x, y) for x, y in zip(warm, cold)
             if np.isfinite(x) and np.isfinite(y) and x < 1e14 and y < 1e14]
    if len(pairs) < 2:
        return None, None, len(pairs)
    xs, ys = zip(*pairs)
    t, p = scipy_stats.ttest_rel(list(xs), list(ys))
    t, p = float(t), float(p)
    if np.isnan(p):
        return None, None, len(pairs)
    return t, p, len(pairs)


T_METHODS = {
    "paired_seed_test_inf_to_10x_max": t_paired_seed_test,
    "drop_pairs_unless_both_finite": t_drop_nonfinite,
}


# ── New statistics ────────────────────────────────────────────────────────────

def wilcoxon_inf_substituted(warm: list[float], cold: list[float]):
    """Wilcoxon signed-rank with inf -> finite_max*10 (rank-preserving)."""
    a = np.array(warm, dtype=float)
    b = np.array(cold, dtype=float)
    finite = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    finite_max = max(float(finite.max()) if finite.size else 0.0, 1.0)
    a = np.where(np.isinf(a), finite_max * 10, a)
    b = np.where(np.isinf(b), finite_max * 10, b)
    diff = a - b
    if np.count_nonzero(diff) == 0:
        return None, None
    try:
        w, p = scipy_stats.wilcoxon(diff)
    except ValueError:
        return None, None
    return float(w), float(p)


def _inf_aware_diffs(warm: list[float], cold: list[float]) -> np.ndarray:
    """Per-cell warm-cold diff: both-fail -> 0 (tie), single-fail -> +/-inf."""
    out = []
    for w, c in zip(warm, cold):
        wi, ci = np.isinf(w), np.isinf(c)
        if wi and ci:
            out.append(0.0)
        elif wi:
            out.append(float("inf"))
        elif ci:
            out.append(float("-inf"))
        else:
            out.append(w - c)
    return np.array(out, dtype=float)


def bootstrap_median_ci(warm: list[float], cold: list[float],
                        n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Percentile 95% CI on the median per-cell cost diff (10k resamples)."""
    diffs = _inf_aware_diffs(warm, cold)
    n = len(diffs)
    if n == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(diffs[idx], axis=1)
    # method='nearest': endpoints are actual resample medians, so a -inf/+inf
    # endpoint stays -inf/+inf instead of NaN-ing out under interpolation.
    # (All comparison sizes here are odd, so each resample median is itself
    # an actual diff value, never an average of two.)
    lo, hi = np.quantile(medians, [0.025, 0.975], method="nearest")
    return float(np.median(diffs)), float(lo), float(hi)


def win_and_reach(warm: list[float], cold: list[float]):
    n = len(warm)
    wins = sum(1 for w, c in zip(warm, cold) if w < c)  # strict <; inf<inf False
    warm_reach = sum(1 for w in warm if np.isfinite(w))
    cold_reach = sum(1 for c in cold if np.isfinite(c))
    return {
        "n_cells": n,
        "win_rate": wins / n if n else None,
        "warm_reach_rate": warm_reach / n if n else None,
        "cold_reach_rate": cold_reach / n if n else None,
        "n_both_reached": sum(1 for w, c in zip(warm, cold)
                              if np.isfinite(w) and np.isfinite(c)),
    }


def holm(pvals: "list[float | None]") -> "list[float | None]":
    """Holm-Bonferroni step-down; None p-values are excluded from the family."""
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    m = len(indexed)
    indexed.sort(key=lambda ip: ip[1])
    out: list = [None] * len(pvals)
    running_max = 0.0
    for rank, (i, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        out[i] = running_max
    return out


# ── Comparison assembly ───────────────────────────────────────────────────────

def analyze(name: str, scale: str, warm: list[float], cold: list[float],
            solvable_mask: list[bool], t_method: str,
            original_stored_p: "float | None" = None,
            original_stored_stat_name: "str | None" = None,
            notes: str = "") -> dict:
    """Full stats for one paired comparison, all-cells and solvable-only."""
    tfun = T_METHODS[t_method]

    def block(w, c):
        t, p, n_used = tfun(w, c)
        w_stat, w_p = wilcoxon_inf_substituted(w, c)
        med, lo, hi = bootstrap_median_ci(w, c)
        return {
            **win_and_reach(w, c),
            "t_stat": t, "t_p": p, "t_n_pairs_used": n_used,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "median_diff": med,
            "bootstrap_ci95_median_diff": [lo, hi],
        }

    all_cells = block(warm, cold)
    sw = [w for w, m in zip(warm, solvable_mask) if m]
    sc = [c for c, m in zip(cold, solvable_mask) if m]
    solvable = block(sw, sc)

    if original_stored_p is not None and all_cells["t_p"] is not None:
        if abs(original_stored_p - all_cells["t_p"]) > 1e-9:
            notes += (f" [WARNING: reproduced t p={all_cells['t_p']:.6g} differs "
                      f"from stored {original_stored_stat_name}="
                      f"{original_stored_p:.6g}]")

    return {
        "comparison": name,
        "scale": scale,
        "inf_rule": INF_RULE,
        "original_t_method": t_method,
        "original_stored_p": original_stored_p,
        "original_stored_stat_name": original_stored_stat_name,
        "n_excluded_unsolvable": len(solvable_mask) - sum(solvable_mask),
        "all_cells": all_cells,
        "solvable_only": solvable,
        "notes": notes.strip(),
    }


def main() -> None:
    # ── Load audit and build unsolvable sets ─────────────────────────────────
    audit = _load("runs/eval_reachability_audit.json",
                  {"scale", "seed", "scenario_id", "reachable"})
    unsolvable: dict[str, set] = {"25x25": set(), "50x50": set(), "100x100": set()}
    unsolvable_longtrain: set = set()
    for r in audit:
        key = (r["seed"], r["scenario_id"])
        if r.get("tier") == "4x_long_train":
            if not r["reachable"]:
                unsolvable_longtrain.add(key)
        elif not r["reachable"]:
            unsolvable[r["scale"]].add(key)
    print("Unsolvable cells per audit (sweep grids): "
          + ", ".join(f"{s}: {len(v)}" for s, v in unsolvable.items())
          + f"; long-train 40-iter state: {len(unsolvable_longtrain)}")
    expected = {"25x25": 1, "50x50": 3, "100x100": 3}
    for s, n in expected.items():
        if len(unsolvable[s]) != n:
            print(f"  WARNING: audit shows {len(unsolvable[s])} unsolvable at {s}, "
                  f"task brief said {n}")

    def mask(records, scale, seed_key="seed", scen_key="scenario_id"):
        return [(r[seed_key], r[scen_key]) not in unsolvable[scale]
                for r in records]

    comparisons: list[dict] = []

    # ── 1. 25x25 warm vs cold (sweep_phase3_final) ───────────────────────────
    p3 = _load("runs/sweep_phase3_final.json",
               {"seed", "scenario_id", "warm_cost", "cold_cost",
                "warm_strict", "cold_strict"})
    warm = [_fin(r["warm_cost"]) for r in p3]
    cold = [_fin(r["cold_cost"]) for r in p3]
    comparisons.append(analyze(
        "25x25_warm_vs_cold", "25x25", warm, cold, mask(p3, "25x25"),
        "paired_seed_test_inf_to_10x_max",
        notes="Original analysis: paired_seed_test in "
              "scripts/run_multi_seed_warm_vs_cold.py (no aggregate p stored "
              "in the JSON; reproduced from raw cells)."))

    # ── 2-5. 50x50 1x/4x at budgets 300/1000 ─────────────────────────────────
    agg50 = _load("runs/sweep_50x50/aggregate_50x50.json", {"tier_1x", "tier_4x"},
                  sample_of_list=False)
    for tier, rel in (("1x", "runs/sweep_50x50/sweep_v1_50x50_1x.json"),
                      ("4x", "runs/sweep_50x50/sweep_v1_50x50_4x.json")):
        recs = _load(rel, {"seed", "scenario_id", "warm_cost", "cold_cost",
                           "warm_cost_1000", "cold_cost_1000"})
        for budget, wkey, ckey in (("300", "warm_cost", "cold_cost"),
                                   ("1000", "warm_cost_1000", "cold_cost_1000")):
            warm = [_fin(r[wkey]) for r in recs]
            cold = [_fin(r[ckey]) for r in recs]
            stored = agg50[f"tier_{tier}"][f"budget_{budget}"].get("paired_p")
            comparisons.append(analyze(
                f"50x50_{tier}_budget{budget}", "50x50", warm, cold,
                mask(recs, "50x50"), "drop_pairs_unless_both_finite",
                original_stored_p=stored,
                original_stored_stat_name=f"aggregate_50x50.tier_{tier}."
                                          f"budget_{budget}.paired_p",
                notes="Original analysis: _paired_t in scripts/run_sweep_50x50.py "
                      "(non-reached pairs silently dropped from the t-test)."))

    # ── 6. 100x100 warm vs cold (sweep_v1_on_100x100) ────────────────────────
    s100 = _load("runs/sweep_v1_on_100x100.json",
                 {"seed", "scenario_id", "warm_cost", "cold_cost"})
    warm = [_fin(r["warm_cost"]) for r in s100]
    cold = [_fin(r["cold_cost"]) for r in s100]
    comparisons.append(analyze(
        "100x100_warm_vs_cold", "100x100", warm, cold, mask(s100, "100x100"),
        "paired_seed_test_inf_to_10x_max",
        notes="Original analysis: paired_seed_test (same script as 25x25). "
              "Very low reach on both arms at this scale; t-test on "
              "substituted values is reported but nearly meaningless."))

    # ── 7. 100x100 long-train warm 4x vs cold 4x control (15 cells) ──────────
    lt = _load("runs/v1_long_train_subset_results.json",
               {"seed", "scenario_id", "warm_cost", "warm_strict"})
    cc = _load("runs/cold_4x_control_results.json",
               {"seed", "scenario_id", "cold_cost_1000", "cold_strict_1000"})
    lt_agg = _load("runs/v1_long_train_aggregate_15cell.json", {"mcnemar_p"},
                   sample_of_list=False)
    cc_agg = _load("runs/cold_4x_control_aggregate.json", {"budget_1000"},
                   sample_of_list=False)
    cold_by_cell = {(r["seed"], r["scenario_id"]): _fin(r["cold_cost_1000"])
                    for r in cc}
    pairs_missing = [(r["seed"], r["scenario_id"]) for r in lt
                     if (r["seed"], r["scenario_id"]) not in cold_by_cell]
    if pairs_missing:
        print(f"  WARNING: long-train cells missing in cold control: {pairs_missing}")
    lt = [r for r in lt if (r["seed"], r["scenario_id"]) in cold_by_cell]
    warm = [_fin(r["warm_cost"]) for r in lt]
    cold = [cold_by_cell[(r["seed"], r["scenario_id"])] for r in lt]
    lt_mask = [(r["seed"], r["scenario_id"]) not in unsolvable_longtrain
               for r in lt]
    stored_mcn = cc_agg["budget_1000"]["mcnemar_warm_vs_cold_p"]
    comparisons.append(analyze(
        "100x100_longtrain_warm4x_vs_cold4x", "100x100", warm, cold, lt_mask,
        "paired_seed_test_inf_to_10x_max",
        original_stored_p=None,  # original had NO cost t-test, only McNemar
        notes=f"Original analysis: continuity-corrected McNemar on reach only "
              f"(cold_4x_control_aggregate budget_1000 p={stored_mcn:.6g}; "
              f"v1_long_train_aggregate_15cell 1x-vs-4x mcnemar_p="
              f"{lt_agg['mcnemar_p']:.6g}). Cold@300 and cold@1000 are "
              f"identical in this control. The cost t-test here is NEW, "
              f"computed with the paired_seed_test convention. All 15 cells "
              f"are reachable in the matched 40-iteration env state per the "
              f"long-train audit tier, so solvable-only equals all-cells."))

    # ── 8. Unified benchmark: V1 (warm) vs Cold per regime ───────────────────
    uni = _load("runs/unified_benchmark_100x100.json",
                {"seed", "scenario_id", "condition", "regime",
                 "reached_goal", "total_cost"})
    by_key: dict = {}
    for r in uni:
        by_key[(r["condition"], r["regime"], r["seed"], r["scenario_id"])] = r
    regimes = sorted({r["regime"] for r in uni})
    cells = sorted({(r["seed"], r["scenario_id"]) for r in uni})
    for regime in regimes:
        warm, cold, cmask = [], [], []
        for seed, scen in cells:
            v1 = by_key.get(("V1", regime, seed, scen))
            cd = by_key.get(("Cold", regime, seed, scen))
            if v1 is None or cd is None:
                print(f"  WARNING: unified benchmark missing V1/Cold record "
                      f"for {seed}/{scen}/{regime}")
                continue
            warm.append(_fin(v1["total_cost"]))
            cold.append(_fin(cd["total_cost"]))
            cmask.append((seed, scen) not in unsolvable["100x100"])
        comparisons.append(analyze(
            f"unified_100x100_V1_vs_Cold_{regime}", "100x100", warm, cold,
            cmask, "drop_pairs_unless_both_finite",
            notes="Original aggregate reported only V3-vs-{V1,V2,Astar} paired "
                  "stats; V1-vs-Cold was never tested, so original_stored_p is "
                  "null. t-test uses the unified script's paired_ttest "
                  "convention (both-finite pairs only)."))

    # ── Optional: demo-source ablation ───────────────────────────────────────
    abl_path = RUNS / "demo_source_ablation_results.json"
    ablation_note = None
    if abl_path.exists():
        with open(abl_path) as fh:
            abl = json.load(fh)
        probe = abl[0] if isinstance(abl, list) and abl else abl
        ablation_note = ("demo_source_ablation_results.json exists but its "
                         f"schema was not pre-confirmed; actual fields: "
                         f"{sorted(probe) if isinstance(probe, dict) else type(probe).__name__}. "
                         "Not folded into the paired comparisons -- re-run once "
                         "the pipeline finishes and the schema is known.")
        print(f"NOTE: {ablation_note}")
    else:
        ablation_note = ("runs/demo_source_ablation_results.json absent at "
                         "analysis time (detached ablation pipeline still "
                         "running); skipped.")
        print(f"NOTE: {ablation_note}")

    # ── Holm-Bonferroni per scale (M1-M3 gate family) ────────────────────────
    by_scale: dict[str, list[int]] = {}
    for i, c in enumerate(comparisons):
        by_scale.setdefault(c["scale"], []).append(i)
    for scale, idxs in by_scale.items():
        for variant in ("all_cells", "solvable_only"):
            for key, out_key in (("t_p", "t_p_holm"),
                                 ("wilcoxon_p", "wilcoxon_p_holm")):
                adj = holm([comparisons[i][variant][key] for i in idxs])
                for i, a in zip(idxs, adj):
                    comparisons[i][variant][out_key] = a
        for i in idxs:
            comparisons[i]["holm_family"] = {
                "scale": scale,
                "members": [comparisons[j]["comparison"] for j in idxs],
                "alpha": ALPHA,
            }

    # ── Conclusion changes ───────────────────────────────────────────────────
    def sig(p):
        return (p is not None) and (p < ALPHA)

    for c in comparisons:
        ac, so = c["all_cells"], c["solvable_only"]
        base = sig(ac["t_p"])  # the original (M3-style) conclusion
        flags = {
            "nonparametric": sig(ac["wilcoxon_p"]) != base,
            "holm_correction": sig(ac["t_p_holm"]) != base,
            "solvable_only": sig(so["t_p"]) != base,
            "m1_win_gate_flips_solvable_only": (
                ac["win_rate"] is not None and so["win_rate"] is not None
                and (ac["win_rate"] >= WIN_GATE) != (so["win_rate"] >= WIN_GATE)),
        }
        c["conclusion_change_flags"] = flags
        c["conclusion_changed"] = any(flags.values())
        c["significant_at_001"] = {
            "original_t": base,
            "wilcoxon": sig(ac["wilcoxon_p"]),
            "t_holm": sig(ac["t_p_holm"]),
            "wilcoxon_holm": sig(ac["wilcoxon_p_holm"]),
            "solvable_only_t": sig(so["t_p"]),
            "solvable_only_wilcoxon": sig(so["wilcoxon_p"]),
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    payload = {
        "generated": "reanalyze_stats.py",
        "inf_rule": INF_RULE,
        "alpha_m3": ALPHA,
        "win_gate_m1": WIN_GATE,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                      "ci": "percentile 95% on median per-cell diff; "
                            "both-fail diff = 0, single-fail diff = +/-inf"},
        "unsolvable_excluded": {
            s: sorted([f"{seed}/{scen}" for seed, scen in v])
            for s, v in unsolvable.items()},
        "demo_source_ablation": ablation_note,
        "comparisons": comparisons,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)  # allow_nan: inf -> Infinity, as in
        # the existing run JSONs (v1_long_train_subset_results.json precedent)
    print(f"\nWrote {OUT_PATH}")

    # ── Summary table ────────────────────────────────────────────────────────
    def _p(v):
        if v is None:
            return "    n/a"
        return f"{v:7.1e}" if v < 1e-4 else f"{v:7.4f}"

    def _ci(ci):
        lo, hi = ci
        if lo is None:
            return "n/a"
        f = lambda x: "-inf" if x == float("-inf") else (
            "+inf" if x == float("inf") else f"{x:.0f}")
        return f"[{f(lo)},{f(hi)}]"

    hdr = (f"{'comparison':<38} {'orig_p':>7} {'wilcox':>7} {'holm_t':>7} "
           f"{'solv_t':>7} {'win%':>5}/{'solv%':<5} {'CI(median diff)':>18} chg")
    print("\n" + hdr)
    print("-" * len(hdr))
    for c in comparisons:
        ac, so = c["all_cells"], c["solvable_only"]
        win = f"{ac['win_rate']*100:4.0f}" if ac["win_rate"] is not None else " n/a"
        swin = f"{so['win_rate']*100:<4.0f}" if so["win_rate"] is not None else "n/a "
        chg = "**YES**" if c["conclusion_changed"] else "no"
        print(f"{c['comparison']:<38} {_p(ac['t_p'])} {_p(ac['wilcoxon_p'])} "
              f"{_p(ac['t_p_holm'])} {_p(so['t_p'])} {win}/{swin} "
              f"{_ci(ac['bootstrap_ci95_median_diff']):>18} {chg}")
    print("-" * len(hdr))
    changed = [c for c in comparisons if c["conclusion_changed"]]
    if changed:
        print(f"\n{len(changed)} comparison(s) change conclusion under at least "
              "one robustness variant:")
        for c in changed:
            reasons = [k for k, v in c["conclusion_change_flags"].items() if v]
            print(f"  - {c['comparison']}: {', '.join(reasons)}")
    else:
        print("\nNo comparison changes conclusion under any variant.")


if __name__ == "__main__":
    main()
