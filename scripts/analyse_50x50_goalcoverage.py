"""Pooled warm-vs-cold analysis for the 50x50 goal-coverage replication.

Pairs are matched per (seed, source, destination) so warm and cold are compared
on the identical query, then pooled across seeds. Reach is compared with an
exact McNemar test on the discordant pairs; cost is compared with a Wilcoxon
signed-rank test restricted to queries BOTH arms reached (a cost comparison
over queries only one arm reached is not a paired comparison at all).
"""
import json
import math
import pathlib
from statistics import median

SCALE = "50x50"
VARIANT = "_mq"
RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs" / "fleet"


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial(b, b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank, normal approx with tie correction."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_pos = sum(ranks[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return None
    z = (w_pos - mu) / sd
    return math.erfc(abs(z) / math.sqrt(2))


def collect(split):
    """Return per-seed and pooled paired records for one split."""
    per_seed, pooled = [], []
    for p in sorted(RUNS.glob(f"reeval_seed*_{SCALE}{VARIANT}.json")):
        d = json.loads(p.read_text())
        seed = d["summary"]["tag"]
        warm = {(r["source"], r["destination"]): r
                for r in d["rows"]["warm"][split]}
        cold = {(r["source"], r["destination"]): r
                for r in d["rows"]["cold"][split]}
        recs = []
        for key in warm.keys() & cold.keys():
            w, c = warm[key], cold[key]
            if not (w.get("solvable") and c.get("solvable")):
                continue
            recs.append((w, c))
        per_seed.append((seed, recs))
        pooled.extend(recs)
    return per_seed, pooled


def describe(recs):
    b = sum(1 for w, c in recs if w["reached"] and not c["reached"])
    c_ = sum(1 for w, c in recs if c["reached"] and not w["reached"])
    both = sum(1 for w, c in recs if w["reached"] and c["reached"])
    wr = sum(1 for w, _ in recs if w["reached"])
    cr = sum(1 for _, c in recs if c["reached"])
    p = mcnemar_exact(b, c_)
    paired = [(w["cost"], c["cost"]) for w, c in recs
              if w["reached"] and c["reached"]
              and w["cost"] is not None and c["cost"] is not None]
    cost = None
    if paired:
        diffs = [cc - ww for ww, cc in paired]   # >0 means warm cheaper
        cost = {
            "n": len(paired),
            "warm_median": median(x for x, _ in paired),
            "cold_median": median(y for _, y in paired),
            "warm_better": sum(1 for x, y in paired if x < y),
            "p": wilcoxon(diffs),
        }
    return dict(n=len(recs), warm=wr, cold=cr, both=both,
                warm_only=b, cold_only=c_, p=p, cost=cost)


def show(title, per_seed, pooled):
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")
    print(f"  {'seed':<26} {'warm':>9} {'cold':>9} {'w-only':>7} "
          f"{'c-only':>7} {'McNemar p':>10}")
    sig = 0
    for seed, recs in per_seed:
        s = describe(recs)
        star = " *" if s["p"] < 0.05 else ""
        sig += s["p"] < 0.05
        print(f"  {seed:<26} {s['warm']:>4}/{s['n']:<4} "
              f"{s['cold']:>4}/{s['n']:<4} {s['warm_only']:>7} "
              f"{s['cold_only']:>7} {s['p']:>10.4g}{star}")
    s = describe(pooled)
    print(f"  {'-' * 72}")
    print(f"  {'POOLED':<26} {s['warm']:>4}/{s['n']:<4} "
          f"{s['cold']:>4}/{s['n']:<4} {s['warm_only']:>7} "
          f"{s['cold_only']:>7} {s['p']:>10.4g}"
          f"{' *' if s['p'] < 0.05 else ''}")
    print(f"\n  seeds significant at 0.05: {sig}/{len(per_seed)}")
    print(f"  pooled reach: warm {100*s['warm']/s['n']:.1f}%  "
          f"cold {100*s['cold']/s['n']:.1f}%  "
          f"(both {s['both']}, discordant {s['warm_only']+s['cold_only']})")
    if s["cost"]:
        c = s["cost"]
        pv = f"{c['p']:.4g}" if c["p"] is not None else "n<6, not tested"
        print(f"  pooled cost on {c['n']} commonly-reached: warm median "
              f"{c['warm_median']:.2f}  cold median {c['cold_median']:.2f}  "
              f"warm better {c['warm_better']}/{c['n']}  wilcoxon p={pv}")
    return s


print(f"50x50 multi-query goal-coverage replication  (variant {VARIANT})")
tr_ps, tr_pool = collect("train")
ho_ps, ho_pool = collect("held_out")
tr = show("TRAINING GOALS  (demonstrated during training)", tr_ps, tr_pool)
ho = show("HELD-OUT GOALS  (never demonstrated)", ho_ps, ho_pool)

print(f"\n{'=' * 74}\n  VERDICT\n{'=' * 74}")
train_win = tr["p"] < 0.05 and tr["warm_only"] > tr["cold_only"]
ho_null = ho["p"] >= 0.05
if train_win and ho_null:
    print("  Goal-coverage boundary REPLICATES at 50x50.")
    print("  Warm-start transfers to goals it was shown; it does not")
    print("  generalise to goals it was not. Same pattern as 25x25.")
elif train_win and not ho_null:
    print("  Warm wins on BOTH splits -- boundary does not hold at 50x50.")
else:
    print("  No warm advantage even on training goals -- budget-limited;")
    print("  not informative about goal coverage.")
