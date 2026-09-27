"""Analysis for the masked action-space control.

Three things, all paired:
  A. warm vs cold reach and cost WITHIN the masked run (the control itself)
  B. the same contrast in the published reactive run, for comparison
  C. episodes-to-threshold from the masked training traces, replicating the
     sample-efficiency measurement of section 4.3 under masking
"""
import json
import math
import pathlib
from statistics import median

NEW = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs\sweep_phase3_masked.json")
OLD = pathlib.Path(r"C:\Users\amirh\Desktop\Demo\runs\sweep_phase3_final.json")
TRACES = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs\traces_25x25_masked")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    d = json.loads(p.read_text())
    return d if isinstance(d, list) else d.get("cells", d.get("rows", []))


def reach_report(label, rows):
    w = sum(bool(r.get("warm_strict")) for r in rows)
    c = sum(bool(r.get("cold_strict")) for r in rows)
    b = sum(1 for r in rows if r.get("warm_strict") and not r.get("cold_strict"))
    cc = sum(1 for r in rows if r.get("cold_strict") and not r.get("warm_strict"))
    both = sum(1 for r in rows if r.get("warm_strict") and r.get("cold_strict"))
    n = len(rows)
    print(f"\n=== {label}  (n={n}) ===")
    print(f"  strict reach: warm {w}/{n} ({100*w/n:.0f}%)   "
          f"cold {c}/{n} ({100*c/n:.0f}%)")
    print(f"  discordant: warm-only={b}  cold-only={cc}  both={both}")
    print(f"  exact McNemar p = {mcnemar(b, cc):.4g}")
    pairs = [(r["warm_cost"], r["cold_cost"]) for r in rows
             if r.get("warm_strict") and r.get("cold_strict")
             and r.get("warm_cost") and r.get("cold_cost")]
    if pairs:
        wins = sum(1 for x, y in pairs if x < y)
        print(f"  both-reached cost: warm cheaper on {wins}/{len(pairs)}"
              f"   median warm {median(p[0] for p in pairs):.0f}"
              f" vs cold {median(p[1] for p in pairs):.0f}")
    strict_win = sum(1 for r in rows
                     if (r.get("warm_cost") or float("inf"))
                     < (r.get("cold_cost") or float("inf")))
    print(f"  strict-win rate (warm cost < cold cost): "
          f"{strict_win}/{n} = {100*strict_win/n:.0f}%")
    return dict(n=n, warm=w, cold=c, b=b, c=cc, p=mcnemar(b, cc))


def etp(path, thr=0.5, win=25):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    hits = [bool(r["reached_goal"]) for r in rows]
    for i in range(win, len(hits) + 1):
        if sum(hits[i - win:i]) / win > thr:
            return i
    return None


new = load(NEW)
print(f"masked cells available: {len(new)}")
r_new = reach_report("MASKED action space (this control)", new)
if OLD.exists():
    r_old = reach_report("REACTIVE action space (published sweep)", load(OLD))

print("\n=== C. episodes-to-threshold under masking ===")
cells = sorted(d.name for d in TRACES.iterdir() if d.is_dir())
wc = cc2 = 0
wv, cv = [], []
paired = 0
disc_w = disc_c = 0
for name in cells:
    wp = TRACES / name / "warm" / "episodes.jsonl"
    cp = TRACES / name / "cold" / "episodes.jsonl"
    if not (wp.exists() and cp.exists()):
        continue
    paired += 1
    a, b = etp(wp), etp(cp)
    wc += a is not None
    cc2 += b is not None
    disc_w += (a is not None) and (b is None)
    disc_c += (b is not None) and (a is None)
    if a:
        wv.append(a)
    if b:
        cv.append(b)
print(f"  paired cells with complete traces: {paired} of {len(cells)} scanned")
print(f"  crossed 50% rolling threshold: warm {wc}/{paired}, cold {cc2}/{paired}")
print(f"  discordant: warm-only={disc_w}  cold-only={disc_c}  "
      f"exact McNemar p = {mcnemar(disc_w, disc_c):.4g}")
if wv:
    print(f"  median episodes-to-threshold: warm {median(wv):.0f}")
if cv:
    print(f"  median episodes-to-threshold: cold {median(cv):.0f}")
