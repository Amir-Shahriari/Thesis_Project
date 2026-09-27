"""Merge per-seed sweep shards into one artefact.

Usage: merge_sweep_shards.py <shard_root> <out.json>

Each shard is <shard_root>/seed<N>/sweep_v1_50x50_1x.json holding that seed's
cells. Merging concatenates them in the canonical seed order and verifies the
result: no duplicate (seed, scenario_id), and the expected cell count.
"""
import json
import pathlib
import sys
from collections import Counter

CANONICAL = [42, 1337, 2024, 7, 314159]
EXPECTED_PER_SEED = 5

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])

merged = []
missing = []
for s in CANONICAL:
    p = root / f"seed{s}" / "sweep_v1_50x50_1x.json"
    if not p.exists():
        missing.append(s)
        continue
    rows = json.loads(p.read_text())
    rows = rows if isinstance(rows, list) else rows.get("cells", [])
    print(f"  seed {s:<7} {len(rows)} cells   ({p})")
    merged.extend(rows)

if missing:
    print(f"\n  WARNING: no shard for seed(s) {missing} -- merge is incomplete")

keys = [(r.get("seed"), r.get("scenario_id")) for r in merged]
dupes = [k for k, n in Counter(keys).items() if n > 1]
print(f"\n  merged {len(merged)} cells; duplicates: {dupes or 'none'}")
if dupes:
    raise SystemExit("ABORT: duplicate cells across shards, refusing to write")

expected = EXPECTED_PER_SEED * (len(CANONICAL) - len(missing))
if len(merged) != expected:
    print(f"  WARNING: expected {expected} cells, got {len(merged)}")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(merged, indent=2))
print(f"  wrote {out}")

w = sum(1 for r in merged if r.get("warm_strict"))
c = sum(1 for r in merged if r.get("cold_strict"))
print(f"  warm {w}/{len(merged)}   cold {c}/{len(merged)}")
