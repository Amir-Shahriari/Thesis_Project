"""Recover the demonstration-source ablation from its run log.

The JSON artefact holds only 9 of 50 cells: the aggregation job aborted on a
path-cost reproducibility guard ("14/25 warm cells differ >1% from published,
baseline 5") before joining. The log records every cell's arm, seed, query,
cost and strict flag, so the comparison can be reconstructed from it.
"""
import math
import pathlib
import re
from statistics import median

LOG = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs"
                   r"\demo_source_ablation.log")
text = LOG.read_text(errors="replace")

pat = re.compile(
    r"\[\s*\d+/\d+\]\s+(\w+)\s+seed=(\d+)\s+(\S+)->(\S+)\s*\n"
    r"\s+cost=([\d.]+)\s+strict=(True|False)")
rows = [dict(arm=m[0], seed=int(m[1]), src=m[2], dst=m[3],
             cost=float(m[4]), strict=m[5] == "True")
        for m in pat.findall(text)]

print(f"cells recovered from log: {len(rows)}")
arms = sorted({r["arm"] for r in rows})
for a in arms:
    sub = [r for r in rows if r["arm"] == a]
    reach = sum(r["strict"] for r in sub)
    print(f"  {a:<16} {len(sub):>3} cells   strict reach {reach}/{len(sub)}"
          f"   median cost {median(r['cost'] for r in sub):>8.0f}")


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


if len(arms) >= 2:
    a1, a2 = arms[0], arms[1]
    d1 = {(r["seed"], r["src"], r["dst"]): r for r in rows if r["arm"] == a1}
    d2 = {(r["seed"], r["src"], r["dst"]): r for r in rows if r["arm"] == a2}
    shared = sorted(set(d1) & set(d2), key=str)
    print(f"\n=== {a1} vs {a2}  (paired on {len(shared)} cells) ===")
    b = sum(1 for k in shared if d1[k]["cost"] < d2[k]["cost"])
    c = sum(1 for k in shared if d2[k]["cost"] < d1[k]["cost"])
    ties = len(shared) - b - c
    print(f"  {a1} cheaper on {b}, {a2} on {c}, ties {ties}")
    print(f"  median cost: {a1} {median(d1[k]['cost'] for k in shared):.0f}"
          f"   {a2} {median(d2[k]['cost'] for k in shared):.0f}")
    print(f"  exact sign test p = {sign_p(b, c):.4g}")
    rb = sum(1 for k in shared if d1[k]["strict"] and not d2[k]["strict"])
    rc = sum(1 for k in shared if d2[k]["strict"] and not d1[k]["strict"])
    print(f"  reach discordant: {a1}-only {rb}, {a2}-only {rc}"
          f"  -> McNemar p = {sign_p(rb, rc):.4g}")
    print("  (reach at ceiling in both arms: p=1.0 means no power to resolve a"
          "\n   difference, not evidence that the sources are equivalent)")
