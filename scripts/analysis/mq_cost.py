"""Paired COST comparison on multi-query TRAINING goals.

Section 4.5 reports the reach half of the training-goal claim at both scales
and concedes the cost half is underpowered at 50x50 (only 10 both-reached
queries, p = 0.169). The 25x25 study has five seeds and a far weaker cold
collapse, so its both-reached set should be large enough to carry the test
that 50x50 cannot. This is re-analysis of existing artefacts; no retraining.
"""
import json
import math
import pathlib
from statistics import median

RUNS = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs\fleet")


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank with average ranks for ties and a
    normal approximation with tie correction. Returns (W, z, p, n)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    tie_groups = []
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        tie_groups.append(j - i + 1)
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4
    tie_term = sum(t ** 3 - t for t in tie_groups)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
    if sigma == 0:
        return w, 0.0, 1.0, n
    z = (w - mu + 0.5) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return w, z, p, n


def sign_test(diffs):
    b = sum(1 for d in diffs if d < 0)   # warm cheaper
    c = sum(1 for d in diffs if d > 0)   # cold cheaper
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, p


def collect(pattern, split):
    """Paired (warm_cost, cold_cost, dijkstra) on solvable, both-reached."""
    out = []
    for p in sorted(RUNS.glob(pattern)):
        d = json.loads(p.read_text())
        w = {(r["source"], r["destination"]): r for r in d["rows"]["warm"][split]}
        c = {(r["source"], r["destination"]): r for r in d["rows"]["cold"][split]}
        for k in w.keys() & c.keys():
            rw, rc = w[k], c[k]
            if not (rw.get("solvable") and rc.get("solvable")):
                continue
            if not (rw["reached"] and rc["reached"]):
                continue
            if rw["cost"] is None or rc["cost"] is None:
                continue
            out.append((rw["cost"], rc["cost"], rw.get("dijkstra_cost")))
    return out


for label, pattern in (("25x25 (5 seeds)", "reeval_seed*_25x25.json"),
                       ("50x50 (3 seeds)", "reeval_seed*_50x50_mq.json")):
    for split in ("train", "held_out"):
        rows = collect(pattern, split)
        print(f"\n=== {label} -- {split.upper()} goals, both-reached ===")
        if not rows:
            print("  no both-reached pairs")
            continue
        diffs = [wc - cc for wc, cc, _ in rows]
        b, c, sp = sign_test(diffs)
        _, z, wp, n = wilcoxon_signed_rank(diffs)
        wcosts = [r[0] for r in rows]
        ccosts = [r[1] for r in rows]
        print(f"  n = {len(rows)} paired queries")
        print(f"  median cost  warm {median(wcosts):10.1f}   "
              f"cold {median(ccosts):10.1f}")
        print(f"  warm cheaper on {b}/{b + c}   (cold cheaper {c})")
        print(f"  sign test p = {sp:.4g}")
        print(f"  Wilcoxon signed-rank: z = {z:.3f}, p = {wp:.4g}  (n={n})")
        ratios = [(r[0] / r[2], r[1] / r[2]) for r in rows
                  if r[2] and math.isfinite(r[2]) and r[2] > 0]
        if ratios:
            print(f"  median cost ratio vs Dijkstra: "
                  f"warm {median(x for x, _ in ratios):.2f}x   "
                  f"cold {median(y for _, y in ratios):.2f}x")
