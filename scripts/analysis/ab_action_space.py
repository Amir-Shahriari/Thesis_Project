"""The clean A/B: reactive vs masked training action space, ONE environment.

Both runs use the current codebase, the same GPU/torch, the same
classical-only oracle pool, the same 5 seeds x 5 scenarios, the same budget.
The ONLY difference is mask_visited. This is the comparison that was missing
when the correction was first claimed on a published-vs-masked contrast that
also differed in codebase, torch, device and oracle pool.
"""
import json
import math
import pathlib
from statistics import median

R = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs")
D = pathlib.Path(r"C:\Users\amirh\Desktop\Demo\runs")


def load(p):
    d = json.loads(p.read_text())
    return d if isinstance(d, list) else d.get("cells", d.get("rows", []))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarise(label, rows):
    n = len(rows)
    w = sum(bool(r["warm_strict"]) for r in rows)
    c = sum(bool(r["cold_strict"]) for r in rows)
    b = sum(1 for r in rows if r["warm_strict"] and not r["cold_strict"])
    cc = sum(1 for r in rows if r["cold_strict"] and not r["warm_strict"])
    both = [(r["warm_cost"], r["cold_cost"]) for r in rows
            if r["warm_strict"] and r["cold_strict"]
            and r["warm_cost"] and r["cold_cost"]]
    wins = sum(1 for x, y in both if x < y)
    print(f"\n{label}")
    print(f"  warm {w}/{n} ({100*w/n:.0f}%)   cold {c}/{n} ({100*c/n:.0f}%)")
    print(f"  discordant  warm-only {b}   cold-only {cc}"
          f"   exact McNemar p = {mcnemar(b, cc):.4g}")
    if both:
        print(f"  both-reached {len(both)}: warm cheaper on {wins}"
              f"   median warm {median(x for x, _ in both):.0f}"
              f" vs cold {median(y for _, y in both):.0f}")
    return dict(n=n, w=w, c=c, b=b, cc=cc)


print("=" * 72)
print(" A/B ON ONE ENVIRONMENT  (current codebase, GPU, classical-only pool)")
print("=" * 72)
react = summarise("REACTIVE  (mask_visited=False)",
                  load(R / "sweep_phase3_unmasked_ctrl.json"))
mask = summarise("MASKED    (mask_visited=True)",
                 load(R / "sweep_phase3_masked.json"))

print("\n" + "=" * 72)
print(" FOR REFERENCE ONLY -- different codebase/torch/device/oracle pool")
print("=" * 72)
pub = summarise("PUBLISHED (reactive, June, full pool)",
                load(D / "sweep_phase3_final.json"))

print("\n" + "=" * 72)
print(" VERDICT")
print("=" * 72)
print(f"  cold reach, reactive -> masked : {react['c']}/{react['n']}"
      f"  ->  {mask['c']}/{mask['n']}")
print(f"  warm reach, reactive -> masked : {react['w']}/{react['n']}"
      f"  ->  {mask['w']}/{mask['n']}")
print(f"  warm-vs-cold p, reactive       : {mcnemar(react['b'], react['cc']):.4g}")
print(f"  warm-vs-cold p, masked         : {mcnemar(mask['b'], mask['cc']):.4g}")
print(f"\n  published cold {pub['c']}/{pub['n']} vs control-reactive"
      f" {react['c']}/{react['n']}"
      f"  -> environment reproduces the reactive baseline?"
      f" {'YES' if abs(pub['c']-react['c'])<=3 else 'NO'}")
