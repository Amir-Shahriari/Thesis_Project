"""Measure WHY training episodes ended, reactive vs masked, from real runs.

The claim under test: that the reactive revisit rule (a visited neighbour stays
selectable and ends the episode with -5) is what prevents the cold agent from
learning. If true, cold episodes under the reactive rule should terminate on
'invalid' at a high rate. If they mostly hit 'step_cap' or 'goal' instead, the
rule is not the binding constraint and the claim is wrong.

train_gnn_dqn.py records termination in {goal, step_cap, invalid, dead_end,
other} per episode.
"""
import collections
import json
import pathlib
import sys

R = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs")


def audit(root, label):
    root = pathlib.Path(root)
    if not root.exists():
        print(f"\n{label}: MISSING ({root})")
        return
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    for arm in ("warm", "cold"):
        files = sorted(root.glob(f"*/{arm}/episodes.jsonl"))
        if not files:
            continue
        allc = collections.Counter()
        early = collections.Counter()
        late = collections.Counter()
        n = 0
        for f in files:
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            n += len(rows)
            for i, r in enumerate(rows):
                t = r.get("termination") or "none"
                allc[t] += 1
                (early if i < 50 else late)[t] += 1
        print(f"\n  {arm}  ({len(files)} cell(s), {n} episodes)")
        for name, c in (("ALL", allc), ("episodes 1-50", early),
                        ("episodes 51+", late)):
            tot = sum(c.values()) or 1
            parts = "  ".join(f"{k}={v} ({100*v/tot:.0f}%)"
                              for k, v in c.most_common())
            print(f"    {name:<16} {parts}")


audit(R / "traces_reactive_probe", "REACTIVE rule (mask_visited=False) -- probe")
audit(R / "traces_25x25_masked", "MASKED rule (mask_visited=True) -- 25 cells")
