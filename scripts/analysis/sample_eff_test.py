"""Paired test for the sample-efficiency result of thesis section 4.2.

The chapter reports 17/24 warm cells crossing the 50% goal-reach threshold
against 1/24 cold, and 10/24 vs 1/24 at the 80% threshold, but attaches no
statistical test -- unlike every other paired comparison in the chapter.
The cells are paired by construction (same seed, same scenario, same
perturbation realisation), so exact McNemar applies directly.
"""
import json
import math
import pathlib

RUNS = pathlib.Path(r"C:\Users\amirh\Desktop\Demo\runs")
d = json.load(open(RUNS / "learning_curves_25x25.json"))
arms = d["sample_efficiency"]["arms"]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


for thr in sorted(arms["warm"], key=float):
    w = arms["warm"][thr]["per_cell"]
    c_ = arms["cold"][thr]["per_cell"]
    cells = sorted(set(w) & set(c_))
    b = sum(1 for k in cells if w[k] is not None and c_[k] is None)
    c = sum(1 for k in cells if c_[k] is not None and w[k] is None)
    both = sum(1 for k in cells if w[k] is not None and c_[k] is not None)
    p = mcnemar_exact(b, c)
    print(f"\nthreshold {thr}  ({len(cells)} paired cells)")
    print(f"  warm crossed {arms['warm'][thr]['n_reached']}, "
          f"cold crossed {arms['cold'][thr]['n_reached']}")
    print(f"  discordant: warm-only={b}  cold-only={c}  (both={both})")
    print(f"  exact McNemar p = {p:.3g}")
    # paired episodes-to-threshold on cells where BOTH crossed
    pairs = [(w[k], c_[k]) for k in cells
             if w[k] is not None and c_[k] is not None]
    print(f"  both-crossed pairs available for a signed-rank test: {len(pairs)}"
          + (f"  {pairs}" if pairs else ""))
