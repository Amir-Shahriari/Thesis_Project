"""Paired McNemar tests the thesis asserts but never reports.

Chapter 4 claims the V2 isolated effect (6/25 -> 10/25) is "statistically
meaningful" and the V3 effect (10/25 -> 12/25) is a "+20% improvement",
with no test attached. Both are paired on the SAME cells and the same V1
agent checkpoint, so exact McNemar is the right instrument -- the identical
one used everywhere else in the chapter.
"""
import json
import math
import pathlib

RUNS = pathlib.Path(r"C:\Users\amirh\Desktop\Demo\runs")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


v1 = json.load(open(RUNS / "sweep_v1_on_100x100.json"))
v3 = json.load(open(RUNS / "sweep_v3_adaptive.json"))

print("v1 record keys:", sorted(v1[0].keys()))

def key(r):
    # v1 sweep uses source/destination, v3 sweep uses src/dst; scenario_id is
    # shared and unique within a seed, so join on (seed, scenario_id).
    return (r["seed"], r["scenario_id"])

v1m = {key(r): r for r in v1}
v3m = {key(r): r for r in v3}
shared = sorted(set(v1m) & set(v3m), key=str)
print(f"\npaired cells: {len(shared)} (v1 {len(v1)}, v3 {len(v3)})")

if not shared:
    print("!! keys do not align; falling back to (seed, src, dst)")
    v1m = {(r["seed"], r.get("src"), r.get("dst")): r for r in v1}
    v3m = {(r["seed"], r.get("src"), r.get("dst")): r for r in v3}
    shared = sorted(set(v1m) & set(v3m), key=str)
    print(f"paired cells: {len(shared)}")


def strict(r, prefix):
    # the V1 sweep records the warm arm as warm_strict
    return bool(r["warm_strict"])


def report(name, arm_a, arm_b, get_a, get_b):
    b = c = both = neither = 0
    for k in shared:
        a_ok, b_ok = get_a(v1m[k], v3m[k]), get_b(v1m[k], v3m[k])
        if a_ok and b_ok:
            both += 1
        elif b_ok:
            b += 1          # arm_b only
        elif a_ok:
            c += 1          # arm_a only
        else:
            neither += 1
    na = both + c
    nb = both + b
    p = mcnemar_exact(b, c)
    print(f"\n=== {name} ===")
    print(f"  {arm_a}: {na}/{len(shared)}    {arm_b}: {nb}/{len(shared)}")
    print(f"  discordant: {arm_b}-only={b}  {arm_a}-only={c}  "
          f"(both={both}, neither={neither})")
    print(f"  exact McNemar p = {p:.4f}"
          f"   -> {'significant' if p < 0.05 else 'NOT significant'} at 0.05")


# V1 (no library) vs V2-control (same agent + library, fixed lambda=0.5)
report("V1 baseline vs V2-control (isolated library effect)",
       "V1", "V2-ctrl",
       lambda a, b: strict(a, "v1"),
       lambda a, b: bool(b["v2_repro_strict"]))

# V2-control vs V3 (same agent, same library, adaptive lambda)
report("V2-control vs V3 (isolated inference-rule effect)",
       "V2-ctrl", "V3",
       lambda a, b: bool(b["v2_repro_strict"]),
       lambda a, b: bool(b["v3_strict"]))

# V1 vs V3 for completeness
report("V1 baseline vs V3",
       "V1", "V3",
       lambda a, b: strict(a, "v1"),
       lambda a, b: bool(b["v3_strict"]))
