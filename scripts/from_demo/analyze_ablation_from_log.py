"""
READ-ONLY analysis script: reconstructs demo-source ablation verdict
from runs/demo_source_ablation.log and runs/demo_source_ablation_partial.json
plus the full-arm traced data in runs/sweep_phase3_traced.json.
Prints everything, writes nothing into runs/.
"""
import json, re, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG  = ROOT / "runs" / "demo_source_ablation.log"
PART = ROOT / "runs" / "demo_source_ablation_partial.json"
TRAC = ROOT / "runs" / "sweep_phase3_traced.json"

EXCLUDED = {("2024", "seed518677876_s1")}   # seed as str for uniform keying

# ── 1. Parse log ──────────────────────────────────────────────────────────────
log_text = LOG.read_text()

# Header: excluded cells
header_m = re.search(r"unreachable \(excluded from rates\): \[([^\]]*)\]", log_text)
print("=== LOG HEADER ===")
print(f"Excluded (unreachable) cells: {header_m.group(1) if header_m else 'none found'}")

# Per-cell lines:
#  [  1/50] classical_only seed=42 Node_537->Node_54
#          cost=550 strict=True  (179s)
cell_re = re.compile(
    r"\[\s*(\d+)/\d+\]\s+(\S+)\s+seed=(\d+)\s+(\S+->)\S+\s*\n\s+cost=(\S+)\s+strict=(True|False)",
    re.MULTILINE,
)

# Also need Node_A->Node_B on the header line, fix the regex
cell_re = re.compile(
    r"\[\s*\d+/\d+\]\s+(\S+)\s+seed=(\d+)\s+(Node_\d+->Node_\d+)\s*\n\s+cost=(\S+)\s+strict=(True|False)",
    re.MULTILINE,
)

log_cells = []
for m in cell_re.finditer(log_text):
    arm, seed, pair, cost_s, strict_s = m.groups()
    cost = math.inf if cost_s == "inf" else float(cost_s)
    log_cells.append({"arm": arm, "seed": seed, "pair": pair,
                       "cost": cost, "strict": strict_s == "True"})

by_arm = {}
for r in log_cells:
    by_arm.setdefault(r["arm"], []).append(r)

print("\n=== LOG PARSE COMPLETENESS ===")
for arm, rows in sorted(by_arm.items()):
    print(f"  {arm:20s}: {len(rows)} cells matched from log")
print(f"  TOTAL from log            : {len(log_cells)} lines")

# ── 2. Load partial JSON (classical_only + quantum_only) ─────────────────────
part = json.loads(PART.read_text())
print("\n=== PARTIAL JSON COMPLETENESS ===")
part_by_arm = {}
for r in part:
    part_by_arm.setdefault(r["arm"], []).append(r)
for arm, rows in sorted(part_by_arm.items()):
    print(f"  {arm:20s}: {len(rows)} records in partial JSON")

# ── 3. Load full-arm traced data ──────────────────────────────────────────────
traced = json.loads(TRAC.read_text())
print(f"\n=== TRACED JSON (full arm) ===")
print(f"  Records: {len(traced)}")

# Build unified per-cell dict keyed by scenario_id
# Key: scenario_id (unique across seeds because it encodes the seed)
cells = {}
for r in traced:
    sid = r["scenario_id"]
    seed = str(r["seed"])
    cells[sid] = {
        "seed": seed,
        "scenario_id": sid,
        "source": r["source"],
        "destination": r["destination"],
        "full_warm": r["warm_strict"],
        "full_cold": r["cold_strict"],
    }

# Merge classical_only and quantum_only from partial JSON
for r in part:
    sid = r["scenario_id"]
    arm = r["arm"]
    if sid not in cells:
        cells[sid] = {"seed": str(r["seed"]), "scenario_id": sid,
                      "source": r["source"], "destination": r["destination"]}
    cells[sid][arm] = r["strict"]

# Identify excluded cell
excl_sids = set()
for sid, c in cells.items():
    key = (c["seed"], sid)
    if key in EXCLUDED:
        excl_sids.add(sid)
# Also match by scenario_id directly
excl_sids |= {"seed518677876_s1"}

included = {sid: c for sid, c in cells.items() if sid not in excl_sids}
print(f"  Included cells (after excluding unreachable): {len(included)}")
print(f"  Excluded scenario IDs: {excl_sids & cells.keys()}")

# ── 4. Goal-reach table ───────────────────────────────────────────────────────
print("\n=== GOAL-REACH TABLE ===")
print(f"  (N = {len(included)} cells, 1 excluded as structurally unreachable)\n")

def rate(vals):
    n = len(vals)
    k = sum(vals)
    return k, n, f"{k}/{n} = {k/n:.3f}" if n else "0/0"

arms_order = [("full_warm", "full (warm)"),
              ("full_cold", "full (cold)"),
              ("classical_only", "classical_only"),
              ("quantum_only",   "quantum_only")]

for key, label in arms_order:
    vals = [c[key] for c in included.values() if key in c]
    k, n, s = rate(vals)
    print(f"  {label:20s}: {s}  (missing data for {len(included)-n} cells)")

# ── 5. McNemar test ───────────────────────────────────────────────────────────
def mcnemar(a_vals, b_vals, label_a, label_b):
    pairs = list(zip(a_vals, b_vals))
    b = sum(1 for x,y in pairs if x and not y)   # A=T, B=F
    c = sum(1 for x,y in pairs if not x and y)   # A=F, B=T
    n_conc = sum(1 for x,y in pairs if x == y)
    n = len(pairs)
    # McNemar with continuity correction (Yates)
    denom = b + c
    if denom == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / denom if denom > 1 else 0.0
        # chi2 to p via incomplete gamma (use scipy if available, else approx)
        try:
            from scipy.stats import chi2 as chi2_dist
            p = chi2_dist.sf(chi2, df=1)
        except ImportError:
            # fallback: rough normal approx
            import math
            z = math.sqrt(chi2)
            p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    print(f"\n  {label_a} vs {label_b}:")
    print(f"    N pairs={n}, concordant={n_conc}, b={b} ({label_a}+ {label_b}-), c={c} ({label_a}- {label_b}+)")
    print(f"    McNemar chi2 (Yates) = {chi2:.4f}, p = {p:.4f}")
    if denom == 0:
        print(f"    -> No discordant pairs; test undefined (p=1 by convention)")
    elif p < 0.05:
        winner = label_a if b > c else label_b
        print(f"    -> SIGNIFICANT (p<0.05): {winner} reaches more goals")
    else:
        print(f"    -> Not significant (p>=0.05): arms not distinguishable on goal-reach")

print("\n=== MCNEMAR CONTRASTS ===")

def get_paired(key_a, key_b):
    rows = [(c[key_a], c[key_b])
            for c in included.values()
            if key_a in c and key_b in c]
    return [r[0] for r in rows], [r[1] for r in rows]

a, b = get_paired("full_warm", "classical_only")
mcnemar(a, b, "full_warm", "classical_only")

a, b = get_paired("quantum_only", "classical_only")
mcnemar(a, b, "quantum_only", "classical_only")

a, b = get_paired("full_warm", "full_cold")
mcnemar(a, b, "full_warm", "full_cold")

# ── 6. Per-cell matrix ────────────────────────────────────────────────────────
print("\n=== PER-CELL STRICT-REACH MATRIX ===")
print(f"  Legend: T=True  F=False  -=no data")
print(f"  {'seed':>6}  {'scenario_id':25s}  {'full_w':6}  {'full_c':6}  {'class':6}  {'quant':6}  {'agree?'}")
print(f"  {'-'*6}  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

disagreements = 0
for sid in sorted(included.keys()):
    c = included[sid]
    fw  = "T" if c.get("full_warm")      else ("F" if "full_warm"      in c else "-")
    fc  = "T" if c.get("full_cold")      else ("F" if "full_cold"      in c else "-")
    cl  = "T" if c.get("classical_only") else ("F" if "classical_only" in c else "-")
    qu  = "T" if c.get("quantum_only")   else ("F" if "quantum_only"   in c else "-")
    vals = [v for v in [fw, fc, cl, qu] if v != "-"]
    agree = "ok" if len(set(vals)) <= 1 else "DIFF"
    if agree == "DIFF":
        disagreements += 1
    print(f"  {c['seed']:>6}  {sid:25s}  {fw:6}  {fc:6}  {cl:6}  {qu:6}  {agree}")

print(f"\n  Cells with any arm disagreement: {disagreements} / {len(included)}")

# ── 7. Diversity numbers ──────────────────────────────────────────────────────
print("\n=== DIVERSITY NUMBERS ===")
# Check partial JSON for diversity fields (present in classical_only/quantum_only)
div_found = False
for r in part[:3]:
    if "diversity" in r:
        div_found = True
        break

if div_found:
    print("  Diversity stats ARE present in demo_source_ablation_partial.json.")
    print("  Per-arm summary (from partial JSON, excluding excluded cell):\n")
    excl_scenario = "seed518677876_s1"
    for arm_name in ["classical_only", "quantum_only"]:
        arm_rows = [r for r in part if r["arm"] == arm_name
                    and r.get("scenario_id") != excl_scenario]
        d_list = [r["diversity"] for r in arm_rows if "diversity" in r]
        if not d_list:
            print(f"  {arm_name}: no diversity data")
            continue
        n_qaoa   = [d["n_qaoa_paths"]      for d in d_list]
        n_class  = [d["n_classical_paths"] for d in d_list]
        n_uniq   = [d["n_unique_paths"]    for d in d_list]
        jac_dist = [d["mean_pairwise_jaccard_distance"] for d in d_list]
        n_sa     = [d["n_unique_state_actions"] for d in d_list]
        print(f"  {arm_name} (N={len(d_list)} cells):")
        print(f"    n_qaoa_paths      : mean={sum(n_qaoa)/len(n_qaoa):.1f}  total={sum(n_qaoa)}")
        print(f"    n_classical_paths : mean={sum(n_class)/len(n_class):.1f}  total={sum(n_class)}")
        print(f"    n_unique_paths    : mean={sum(n_uniq)/len(n_uniq):.1f}  total={sum(n_uniq)}")
        print(f"    mean pairwise Jaccard dist: {sum(jac_dist)/len(jac_dist):.4f}")
        print(f"    n_unique_state_actions: mean={sum(n_sa)/len(n_sa):.1f}  total={sum(n_sa)}")
    print()
    print("  NOTE: 'full' arm diversity was aborted mid-recompute; only 12/25 cells")
    print("  completed [full diversity] recompute before sanity check fired.")
    print("  Jaccard OVERLAP (not distance) and full-arm coverage: NOT available.")
else:
    print("  No diversity fields found in partial JSON.")

# Check traced JSON for diversity
if any("diversity" in str(r) for r in traced[:3]):
    print("  Traced JSON also contains diversity data — inspect manually if needed.")
else:
    print("  Traced (full-arm) JSON: no diversity field (warm/cold costs only).")

print("\n=== DONE ===")
