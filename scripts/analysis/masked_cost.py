"""Does anything survive masking? Test the residual cost advantage."""
import json
import math
import pathlib
from statistics import median

rows = json.loads(pathlib.Path(
    r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs\sweep_phase3_masked.json"
).read_text())
rows = rows if isinstance(rows, list) else rows.get("cells", [])


def sign_p(b, c):
    n = b + c
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(d):
    nz = [x for x in d if x != 0]
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    r = [0.0] * n
    i = 0
    ties = []
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        ties.append(j - i + 1)
        i = j + 1
    wp = sum(x for x, v in zip(r, nz) if v > 0)
    wm = sum(x for x, v in zip(r, nz) if v < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24
                   - sum(t ** 3 - t for t in ties) / 48)
    z = (w - mu + 0.5) / sd
    return z, math.erfc(abs(z) / math.sqrt(2)), n


pairs = [(r["warm_cost"], r["cold_cost"], r.get("dijkstra_cost"))
         for r in rows
         if r.get("warm_strict") and r.get("cold_strict")
         and r.get("warm_cost") and r.get("cold_cost")]
d = [w - c for w, c, _ in pairs]
b = sum(1 for x in d if x < 0)
c_ = sum(1 for x in d if x > 0)
z, p, n = wilcoxon(d)
print(f"both-reached pairs: {len(pairs)}")
print(f"  warm cheaper on {b}/{b + c_}   sign test p = {sign_p(b, c_):.4g}")
print(f"  Wilcoxon signed-rank z = {z:.3f}, p = {p:.4g}  (n={n})")
print(f"  median cost: warm {median(x for x, _, _ in pairs):.0f}"
      f"  cold {median(y for _, y, _ in pairs):.0f}")
rr = [(w / dj, c / dj) for w, c, dj in pairs if dj and math.isfinite(dj) and dj > 0]
if rr:
    print(f"  median ratio vs Dijkstra: warm {median(x for x, _ in rr):.2f}x"
          f"   cold {median(y for _, y in rr):.2f}x")
    print(f"  MEAN ratio vs Dijkstra:   warm "
          f"{sum(x for x, _ in rr) / len(rr):.2f}x  (M2 gate <= 5.0)")

reach = sum(bool(r.get("warm_strict")) for r in rows)
solv = [r for r in rows if r.get("dijkstra_cost") not in (None, float("inf"))]
print(f"\n  warm reach {reach}/{len(rows)}; cells with finite Dijkstra: {len(solv)}")
