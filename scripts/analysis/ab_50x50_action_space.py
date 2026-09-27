"""The 50x50 A/B: reactive vs masked, one environment, one variable.

Mirrors ab_action_space.py (25x25). The published June sweep is shown as a
reproduction check on the reactive arm, never as the reactive arm itself --
that substitution is what made the first 25x25 claim unsound.
"""
import json
import math
import os
import pathlib
from statistics import median

R = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs")


def _arm_dir(var, default):
    """Resolve an arm directory given as absolute, repo-relative or runs-relative.

    The driver exports these as "runs/<arm>" while the default is a bare
    "<arm>"; joining both onto R gave runs\\runs\\<arm> and a FileNotFoundError
    after the four-hour masked arm had already finished. Accept either form.
    """
    p = pathlib.Path(os.environ.get(var, default))
    if p.is_absolute():
        return p
    return (R.parent / p) if p.parts[0] == R.name else (R / p)


REACT_DIR = _arm_dir("QWARM_AB_REACT_DIR", "sweep_50x50_reactive_ctrl")
MASK_DIR = _arm_dir("QWARM_AB_MASK_DIR", "sweep_50x50_masked")
REACT = REACT_DIR / "sweep_v1_50x50_1x.json"
MASK = MASK_DIR / "sweep_v1_50x50_1x.json"
PUB = R / "sweep_50x50" / "sweep_v1_50x50_1x.json"


def stack_of(d):
    """Return (torch_version, device) recorded for an arm, or None if unrecorded."""
    p = d / "provenance.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return (r.get("torch_version"), r.get("resolved_device"))


def check_same_stack():
    """An A/B across two execution stacks measures the stack, not the flag.

    This is the check whose absence let a CPU/GPU torch swap go unnoticed
    between the two arms. Refuse to report a verdict without it.
    """
    a, b = stack_of(REACT_DIR), stack_of(MASK_DIR)
    print("\nstack provenance")
    print("  reactive : %s" % ("UNRECORDED" if a is None else "torch %s on %s" % a))
    print("  masked   : %s" % ("UNRECORDED" if b is None else "torch %s on %s" % b))
    if a is None or b is None:
        print("  -> WARNING: an arm has no provenance.json; same-stack claim"
              " NOT verifiable. Treat any difference as confounded.")
        return False
    if a != b:
        print("  -> WARNING: ARMS RAN ON DIFFERENT STACKS. The contrast below"
              " is confounded by torch/device and is not an action-space A/B.")
        return False
    print("  -> same stack confirmed; the action space is the only variable.")
    return True


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
    if not n:
        print(f"\n{label}: EMPTY")
        return None
    w = sum(bool(r.get("warm_strict")) for r in rows)
    c = sum(bool(r.get("cold_strict")) for r in rows)
    b = sum(1 for r in rows if r.get("warm_strict") and not r.get("cold_strict"))
    cc = sum(1 for r in rows if r.get("cold_strict") and not r.get("warm_strict"))
    both = [(r["warm_cost"], r["cold_cost"]) for r in rows
            if r.get("warm_strict") and r.get("cold_strict")
            and r.get("warm_cost") and r.get("cold_cost")]
    print(f"\n{label}")
    print(f"  warm {w}/{n} ({100*w/n:.0f}%)   cold {c}/{n} ({100*c/n:.0f}%)")
    print(f"  discordant  warm-only {b}  cold-only {cc}"
          f"   exact McNemar p = {mcnemar(b, cc):.4g}")
    if both:
        print(f"  both-reached {len(both)}: warm cheaper on "
              f"{sum(1 for x, y in both if x < y)}"
              f"   median warm {median(x for x, _ in both):.0f}"
              f" vs cold {median(y for _, y in both):.0f}")
    return dict(n=n, w=w, c=c, b=b, cc=cc, p=mcnemar(b, cc))


print("=" * 72)
print(" 50x50 A/B -- ONE ENVIRONMENT, ACTION SPACE THE ONLY VARIABLE")
print("=" * 72)
if not REACT.exists():
    raise SystemExit(f"reactive arm not found: {REACT}\nRun "
                     "scripts/run_50x50_reactive_parallel.sh first.")
react = summarise("REACTIVE  (mask_visited=False)", load(REACT))
mask = summarise("MASKED    (mask_visited=True)", load(MASK))

if PUB.exists():
    print("\n" + "-" * 72)
    pub = summarise("PUBLISHED June sweep -- reproduction check only", load(PUB))
    if pub and react:
        ok = abs(pub["c"] - react["c"]) <= 3
        print(f"\n  reactive arm reproduces published cold "
              f"({react['c']}/{react['n']} vs {pub['c']}/{pub['n']}): "
              f"{'YES' if ok else 'NO -- environment differs materially'}")

same_stack = check_same_stack()

print("\n" + "=" * 72)
print(" VERDICT")
print("=" * 72)
if not same_stack:
    print("  (stack not verified identical -- see warning above)")
if react and mask:
    print(f"  cold reach  reactive -> masked : {react['c']}/{react['n']}"
          f"  ->  {mask['c']}/{mask['n']}")
    print(f"  warm reach  reactive -> masked : {react['w']}/{react['n']}"
          f"  ->  {mask['w']}/{mask['n']}")
    print(f"  warm-vs-cold p                 : {react['p']:.4g}"
          f"  ->  {mask['p']:.4g}")
    survives = mask["p"] < 0.05
    print(f"\n  -> warm advantage under the specified action space: "
          f"{'SURVIVES' if survives else 'DOES NOT SURVIVE'}")
