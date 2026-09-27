"""50x50 masked replication vs the published reactive sweep.

Tests whether the 88% vs 12% reach contrast survives the action space the MDP
specifies -- the last single-query reach claim in the thesis still resting on
the reactive revisit rule.
"""
import json
import math
import pathlib
from statistics import median

NEW = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs"
                   r"\sweep_50x50_masked\sweep_v1_50x50_1x.json")
OLD = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs"
                   r"\sweep_50x50\sweep_v1_50x50_1x.json")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    d = json.loads(p.read_text())
    return d if isinstance(d, list) else d.get("cells", d.get("rows", []))


def get(r, *names):
    for n in names:
        if n in r and r[n] is not None:
            return r[n]
    return None


def report(label, rows):
    n = len(rows)
    ws = [bool(get(r, "warm_strict", "warm_reached")) for r in rows]
    cs = [bool(get(r, "cold_strict", "cold_reached")) for r in rows]
    b = sum(1 for w, c in zip(ws, cs) if w and not c)
    c_ = sum(1 for w, c in zip(ws, cs) if c and not w)
    print(f"\n=== {label}  (n={n}) ===")
    print(f"  warm {sum(ws)}/{n} ({100*sum(ws)/n:.0f}%)   "
          f"cold {sum(cs)}/{n} ({100*sum(cs)/n:.0f}%)")
    print(f"  discordant: warm-only {b}, cold-only {c_}"
          f"   exact McNemar p = {mcnemar(b, c_):.4g}")
    both = [(get(r, "warm_cost"), get(r, "cold_cost")) for r, w, c
            in zip(rows, ws, cs) if w and c
            and get(r, "warm_cost") and get(r, "cold_cost")]
    if both:
        wins = sum(1 for x, y in both if x < y)
        print(f"  both-reached {len(both)}: warm cheaper on {wins}"
              f"   median warm {median(x for x, _ in both):.0f}"
              f" vs cold {median(y for _, y in both):.0f}")
    return sum(ws), sum(cs), n


if not NEW.exists():
    raise SystemExit(f"masked 50x50 not found: {NEW}")
w1, c1, n1 = report("MASKED  (mask_visited=True)", load(NEW))
if OLD.exists():
    w0, c0, n0 = report("PUBLISHED reactive (reference)", load(OLD))
    print("\n=== VERDICT ===")
    print(f"  cold reach: reactive {c0}/{n0}  ->  masked {c1}/{n1}")
    print(f"  warm reach: reactive {w0}/{n0}  ->  masked {w1}/{n1}")
    verdict = ("SURVIVES -- contrast remains"
               if (w1 - c1) > 0.5 * (w0 - c0) else
               "COLLAPSES -- contrast largely removed")
    print(f"  -> {verdict}")
    print("\n  NOTE: cross-run comparison (different codebase/torch/device).")
    print("  Treat as indicative; a same-environment reactive arm at 50x50")
    print("  would be needed for the clean A/B run at 25x25.")
