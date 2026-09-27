"""Re-analyse the demonstration-source ablation on ROUTE COST.

Reach saturated at this scale (every arm reaches every cell), so the McNemar
contrasts reported in the thesis have b=c=0 and p=1.0 -- no power, not
equivalence. Section 4.4 additionally shows the reach advantage belongs to the
revisit rule, so reach is the wrong metric for this comparison anyway. Cost is
the metric that survived the action-space control, so the question becomes:
do the three oracle pools produce measurably different ROUTE QUALITY?
"""
import json
import math
import pathlib
from statistics import median

RUNS = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs")
ABL = json.loads((RUNS / "demo_source_ablation_partial.json").read_text())
FULL = json.loads(
    (pathlib.Path(r"C:\Users\amirh\Desktop\Demo\runs\sweep_phase3_traced.json")
     ).read_text())


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


by_arm = {}
for r in ABL:
    by_arm.setdefault(r["arm"], {})[(r["seed"], r["scenario_id"])] = r

full = {(r["seed"], r["scenario_id"]): r for r in FULL}

print("ABLATION ARTEFACT CONTENTS")
for a, cells in by_arm.items():
    print(f"  {a:<16} {len(cells)} cells, seeds {sorted({k[0] for k in cells})}")
print(f"  full pool taken from sweep_phase3_traced.json: {len(full)} cells, "
      f"seeds {sorted({k[0] for k in full})}")

# --- paired classical vs quantum on the cells both ran -------------------
c_arm, q_arm = by_arm["classical_only"], by_arm["quantum_only"]
shared = sorted(set(c_arm) & set(q_arm))
print(f"\n=== classical_only vs quantum_only  (paired, n={len(shared)}) ===")
print(f"  {'cell':<26}{'classical':>12}{'quantum':>12}{'ratio c/q':>12}")
cw = qw = 0
diffs = []
for k in shared:
    cc, qq = c_arm[k]["cost"], q_arm[k]["cost"]
    diffs.append(cc - qq)
    cw += cc < qq
    qw += qq < cc
    print(f"  {k[1]:<26}{cc:>12.0f}{qq:>12.0f}{cc / qq:>12.2f}")
print(f"  classical cheaper on {cw}/{len(shared)}, quantum on {qw}")
print(f"  median cost: classical {median(c_arm[k]['cost'] for k in shared):.0f}"
      f"   quantum {median(q_arm[k]['cost'] for k in shared):.0f}")
print(f"  median ratio vs Dijkstra: "
      f"classical {median(c_arm[k]['cost_ratio'] for k in shared):.2f}x"
      f"   quantum {median(q_arm[k]['cost_ratio'] for k in shared):.2f}x")
print(f"  exact sign test p = {sign_p(cw, qw):.4g}")

# --- each ablation arm vs the full pool on the same cells ----------------
for name, arm in (("classical_only", c_arm), ("quantum_only", q_arm)):
    ks = sorted(set(arm) & set(full))
    if not ks:
        continue
    b = sum(1 for k in ks if arm[k]["cost"] < full[k]["warm_cost"])
    c = sum(1 for k in ks if full[k]["warm_cost"] < arm[k]["cost"])
    print(f"\n=== {name} vs FULL pool  (paired, n={len(ks)}) ===")
    print(f"  {name} cheaper on {b}/{len(ks)}, full pool on {c}")
    print(f"  median cost: {name} {median(arm[k]['cost'] for k in ks):.0f}"
          f"   full {median(full[k]['warm_cost'] for k in ks):.0f}")
    print(f"  exact sign test p = {sign_p(b, c):.4g}")

print("\n=== reach, for the record ===")
for a, cells in by_arm.items():
    print(f"  {a:<16} strict reach {sum(r['strict'] for r in cells.values())}"
          f"/{len(cells)}")
print("  every arm reaches every cell it was run on -> the reach comparison")
print("  is at ceiling; b=c=0 and p=1.0 mean NO POWER, not equivalence.")
