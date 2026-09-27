"""Masked vs reactive multi-query (goal-coverage) control at 50x50.

Tests whether the positive half of the goal-coverage result -- warm wins on
DEMONSTRATED goals in 5/5 seeds, pooled 44 vs 8, p = 4.0e-7 -- survives the
masked action space, the same way section 4.4 tested the single-query result.
The held-out half is a null under both arms and is reported for completeness.
"""
import json
import math
import pathlib

RUNS = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs\fleet")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def analyse(pattern, label):
    files = sorted(RUNS.glob(pattern))
    if not files:
        print(f"\n### {label}: no files matching {pattern}")
        return
    print(f"\n{'=' * 74}\n### {label}   ({len(files)} seeds)\n{'=' * 74}")
    for split in ("train", "held_out"):
        tb = tc = tw = tcold = tn = 0
        wins = 0
        per = []
        for f in files:
            d = json.loads(f.read_text())
            w = {(r["source"], r["destination"]): r
                 for r in d["rows"]["warm"][split]}
            c = {(r["source"], r["destination"]): r
                 for r in d["rows"]["cold"][split]}
            keys = [k for k in w.keys() & c.keys()
                    if w[k].get("solvable") and c[k].get("solvable")]
            b = sum(1 for k in keys if w[k]["reached"] and not c[k]["reached"])
            cc = sum(1 for k in keys if c[k]["reached"] and not w[k]["reached"])
            wr = sum(1 for k in keys if w[k]["reached"])
            cr = sum(1 for k in keys if c[k]["reached"])
            p = mcnemar(b, cc)
            wins += wr > cr
            per.append((f.name.split("_")[1], len(keys), wr, cr, b, cc, p))
            tb += b
            tc += cc
            tw += wr
            tcold += cr
            tn += len(keys)
        print(f"\n  -- {split.upper()} goals --")
        print(f"  {'seed':<10}{'solv':>6}{'warm':>6}{'cold':>6}"
              f"{'w-only':>8}{'c-only':>8}{'p':>10}")
        for s, n, wr, cr, b, cc, p in per:
            print(f"  {s:<10}{n:>6}{wr:>6}{cr:>6}{b:>8}{cc:>8}{p:>10.4g}")
        print(f"  POOLED    {tn:>6}{tw:>6}{tcold:>6}{tb:>8}{tc:>8}"
              f"{mcnemar(tb, tc):>10.4g}")
        print(f"  warm leads in {wins}/{len(files)} seeds")


analyse("reeval_seed*_50x50_mq_masked.json", "MASKED action space (control)")
analyse("reeval_seed*_50x50_mq.json", "REACTIVE action space (published)")
