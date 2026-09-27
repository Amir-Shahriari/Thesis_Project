"""Across-seed test: does the goal-relative Q-head replicate?

Reads the per-seed re-evaluation artefacts written by reeval_fleet.py and asks
one question: within each seed, on identical held-out queries, does switching
the Q-head from plain concatenation to goal-relative change held-out reach --
and does it do so for the WARM arm specifically?

Seed 42 gave 48 -> 71 (McNemar p=0.0022) for warm and 68 -> 70 (p=0.89) for
cold. That is a single seed; this decides whether it is a result.

Three views are reported because each answers a different objection:

  per-seed      the raw effect, seed by seed. Shows whether the direction is
                consistent or driven by one outlier.
  seed-level    sign test over seeds, treating each seed as one observation.
                Conservative, but with 5 seeds the smallest attainable
                two-sided p is 0.0625 -- consistency of direction carries the
                argument, not the p-value.
  pooled        McNemar over every (seed, query) discordant pair. Far more
                powerful, but treats queries as independent across seeds, so
                it overstates confidence if seeds differ systematically. Read
                it alongside the per-seed table, never instead of it.

Usage:
    python scripts/aggregate_seed_grid.py
    python scripts/aggregate_seed_grid.py --scale 25x25 --variant _gr
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_rows(path: pathlib.Path, arm: str, split: str = "held_out") -> dict:
    d = json.loads(path.read_text())
    if arm not in d.get("rows", {}):
        return {}
    return {(r["source"], r["destination"]): r
            for r in d["rows"][arm][split]}


def compare(base: dict, var: dict) -> dict:
    """Paired reach/cost change on the solvable queries the two share."""
    keys = [k for k in (set(base) & set(var)) if base[k].get("solvable")]
    b = {k for k in keys if base[k]["reached"]}
    v = {k for k in keys if var[k]["reached"]}
    gained, lost = sorted(v - b), sorted(b - v)
    both = sorted(b & v)
    out = {
        "n_solvable": len(keys),
        "base_reach": len(b), "var_reach": len(v),
        "gained": len(gained), "lost": len(lost),
        "discordant": len(gained) + len(lost),
    }
    if out["discordant"]:
        out["mcnemar_p"] = float(
            binomtest(len(gained), out["discordant"], 0.5).pvalue)
    if len(both) >= 3:
        cb = np.array([base[k]["cost_ratio"] for k in both], dtype=float)
        cv = np.array([var[k]["cost_ratio"] for k in both], dtype=float)
        ok = np.isfinite(cb) & np.isfinite(cv)
        if ok.sum() >= 3:
            out["cost_n"] = int(ok.sum())
            out["cost_base_median"] = float(np.median(cb[ok]))
            out["cost_var_median"] = float(np.median(cv[ok]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="25x25")
    ap.add_argument("--variant", default="_gr",
                    help="tag suffix of the variant config (baseline is '')")
    ap.add_argument("--fleet-dir", default="runs/fleet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fleet = ROOT / args.fleet_dir if not pathlib.Path(args.fleet_dir).is_absolute() \
        else pathlib.Path(args.fleet_dir)

    # Discover seeds that have BOTH configs re-evaluated.
    seeds = []
    for p in sorted(fleet.glob(f"reeval_seed*_{args.scale}.json")):
        stem = p.stem                       # reeval_seed42_25x25
        seed = stem.split("_")[1].replace("seed", "")
        if (fleet / f"reeval_seed{seed}_{args.scale}{args.variant}.json").exists():
            seeds.append(int(seed))
    seeds.sort()
    if not seeds:
        raise SystemExit(
            f"no seed has both baseline and '{args.variant}' re-evaluations in "
            f"{fleet}. Run scripts\\run_seed_grid.bat first.")

    print("=" * 78)
    print(f"  goal-relative replication  |  scale {args.scale}  |  "
          f"variant '{args.variant}'  |  seeds {seeds}")
    print("=" * 78)

    results: dict = {"scale": args.scale, "variant": args.variant,
                     "seeds": seeds, "per_seed": {}}
    pooled = {"warm": [0, 0], "cold": [0, 0]}      # [gained, lost]

    for arm in ("warm", "cold"):
        print(f"\n  {arm.upper()}  (baseline reach -> variant reach, "
              f"per seed, identical queries)")
        print(f"  {'seed':>8s} {'base':>6s} {'var':>6s} {'gain':>6s} "
              f"{'lost':>6s} {'p':>8s}   {'cost base':>10s} {'cost var':>10s}")
        deltas = []
        for s in seeds:
            b = load_rows(fleet / f"reeval_seed{s}_{args.scale}.json", arm)
            v = load_rows(fleet / f"reeval_seed{s}_{args.scale}{args.variant}.json", arm)
            if not b or not v:
                print(f"  {s:>8d}   (missing {arm} arm; skipped)")
                continue
            c = compare(b, v)
            results["per_seed"].setdefault(str(s), {})[arm] = c
            pooled[arm][0] += c["gained"]
            pooled[arm][1] += c["lost"]
            deltas.append(c["var_reach"] - c["base_reach"])
            cb = c.get("cost_base_median")
            cv = c.get("cost_var_median")
            print(f"  {s:>8d} {c['base_reach']:>6d} {c['var_reach']:>6d} "
                  f"{c['gained']:>6d} {c['lost']:>6d} "
                  f"{c.get('mcnemar_p', float('nan')):>8.4f}   "
                  f"{('%10.2f' % cb) if cb is not None else '         -'} "
                  f"{('%10.2f' % cv) if cv is not None else '         -'}")

        if deltas:
            d = np.array(deltas, dtype=float)
            pos = int((d > 0).sum())
            sign_p = float(binomtest(pos, len(d), 0.5).pvalue) if len(d) else float("nan")
            try:
                w_p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")
            except ValueError:
                w_p = float("nan")
            g, l = pooled[arm]
            pool_p = float(binomtest(g, g + l, 0.5).pvalue) if (g + l) else float("nan")
            print(f"    seed-level : mean delta {d.mean():+.1f} reach, "
                  f"positive in {pos}/{len(d)} seeds, sign p={sign_p:.4f}"
                  + (f", wilcoxon p={w_p:.4f}" if np.isfinite(w_p) else ""))
            print(f"    pooled     : {g} gained vs {l} lost across all "
                  f"(seed, query) pairs, McNemar p={pool_p:.6f}")
            results.setdefault("summary", {})[arm] = {
                "mean_delta_reach": float(d.mean()),
                "seeds_positive": pos, "n_seeds": len(d),
                "sign_p": sign_p, "wilcoxon_p": w_p,
                "pooled_gained": g, "pooled_lost": l, "pooled_mcnemar_p": pool_p,
            }

    print("\n" + "-" * 78)
    sw = results.get("summary", {}).get("warm")
    sc = results.get("summary", {}).get("cold")
    if sw and sc:
        n = sw["n_seeds"]
        # A verdict on replication needs seeds to replicate ACROSS. Below three,
        # report the effect and withhold judgement rather than calling a
        # single-seed result "not replicating" -- that would be a statement
        # about missing runs, not about the intervention.
        warm_works = sw["seeds_positive"] >= max(4, n - 1)
        cold_flat = sc["seeds_positive"] <= sc["n_seeds"] / 2 + 0.5
        print("  VERDICT")
        print(f"    warm improves in {sw['seeds_positive']}/{n} seeds "
              f"(mean {sw['mean_delta_reach']:+.1f}), pooled p={sw['pooled_mcnemar_p']:.6f}")
        print(f"    cold improves in {sc['seeds_positive']}/{sc['n_seeds']} seeds "
              f"(mean {sc['mean_delta_reach']:+.1f}), pooled p={sc['pooled_mcnemar_p']:.6f}")
        if n < 3:
            print(f"    -> only {n} seed(s) available; no replication verdict yet. "
                  f"Run scripts\\run_seed_grid.bat, then re-run this script.")
        elif warm_works and cold_flat:
            print("    -> REPLICATES and is WARM-SPECIFIC: the intervention "
                  "improves the demonstration-warm-started agent and not the "
                  "cold control. This is a thesis result.")
        elif warm_works:
            print("    -> replicates for warm, but cold moves too: the change "
                  "is architecture-general, not a warm-start improvement. "
                  "Report it as such.")
        else:
            print("    -> does NOT replicate across seeds. The seed-42 effect "
                  "was seed-specific; cut the claim.")

    out = pathlib.Path(args.out) if args.out else (
        fleet / f"seed_grid_{args.scale}{args.variant}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
