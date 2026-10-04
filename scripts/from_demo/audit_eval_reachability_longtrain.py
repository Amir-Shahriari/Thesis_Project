"""Extend the eval-reachability audit to the 15 long-train (4x) subset cells.

The V1 long-train entry point is NOT in the repo (only its consumers are), so
the eval graph state is determined EMPIRICALLY: for each cell, the graph is
rebuilt from its grid seed and advanced through update_graph(), computing the
env-cost Dijkstra at candidate states (after 5, 10 and 40 iterations). The
state whose Dijkstra matches the cell's RECORDED dijkstra_cost (to 1e-6 rel.)
is the state the 4x evaluation actually used. The 1x sweep's recorded
dijkstra_cost for the same cell (from runs/sweep_v1_on_100x100.json, validated
as the iteration-10 state by audit_eval_reachability.py) identifies the 1x
eval state for comparison.

Reports:
  (a) per-cell reachability on the 4x eval state,
  (b) whether any residual 4x failure (warm_strict=False) is an unreachable cell,
  (c) whether the 1x and 4x tiers were evaluated on different graph states.

Appends the 15 cells to runs/eval_reachability_audit.json with
tier="4x_long_train" (existing tier records are replaced on rerun; the
original 75 sweep records are untouched).

Read-only analysis: no training; writes only the audit JSON under runs/.

Usage:
    uv run python scripts/audit_eval_reachability_longtrain.py
"""
from __future__ import annotations

import heapq
import json
import math
import pathlib

from qwarm.env.dynamic_graph import DynamicGraph

SUBSET_JSON = pathlib.Path("runs/v1_long_train_subset_results.json")
SWEEP_1X_JSON = pathlib.Path("runs/sweep_v1_on_100x100.json")
AUDIT_JSON = pathlib.Path("runs/eval_reachability_audit.json")

GRID_CFG = dict(grid_width=100, grid_height=100, extra_edges=4, deactivate_prob=0.30)
CANDIDATE_ITERS = [5, 10, 40]
TIER = "4x_long_train"


def _grid_seed(scenario_id: str) -> int:
    return int(scenario_id.split("_s")[0].replace("seed", ""))


def _dijkstra(graph: dict, nodes: dict, src: str, dst: str) -> float:
    q: list[tuple[float, str]] = [(0.0, src)]
    best: dict[str, float] = {src: 0.0}
    vis: set[str] = set()
    while q:
        c, n = heapq.heappop(q)
        if n in vis:
            continue
        vis.add(n)
        if n == dst:
            return c
        for nb, e in graph[n].items():
            if not e["active"] or not nodes[nb]["active"]:
                continue
            nc = c + e["distance"] + 0.1 * e["time"] + nodes[nb]["node_penalty"]
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(q, (nc, nb))
    return float("inf")


def _norm(v) -> "float | None":
    if v is None:
        return None
    f = float(v)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _matches(a: "float | None", b: "float | None") -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def main() -> None:
    cells = json.loads(SUBSET_JSON.read_text())
    sweep_1x = {(r["seed"], r["scenario_id"]): _norm(r.get("dijkstra_cost"))
                for r in json.loads(SWEEP_1X_JSON.read_text())}

    print(f"{len(cells)} long-train subset cells; candidate eval states: "
          f"iteration {CANDIDATE_ITERS}")
    hdr = (f"{'cell':<26} {'strict4x':>8} {'recorded':>9} "
           + "".join(f" {'dij@'+str(k):>9}" for k in CANDIDATE_ITERS)
           + f" {'matches':>8} {'1x dij':>8} {'1x!=4x':>7}")
    print(hdr)
    print("-" * len(hdr))

    new_records: list[dict] = []
    match_counts: dict[int, int] = {k: 0 for k in CANDIDATE_ITERS}
    unmatched: list[str] = []
    failures_unreachable: list[str] = []
    state_diff_count = 0
    state_diff_n = 0

    for cell in cells:
        seed, sid = cell["seed"], cell["scenario_id"]
        src, dst = cell["source"], cell["destination"]
        recorded = _norm(cell.get("dijkstra_cost"))
        warm_strict = bool(cell.get("warm_strict", False))

        g = DynamicGraph(**GRID_CFG, seed=_grid_seed(sid))
        dij_at: dict[int, "float | None"] = {}
        for i in range(1, max(CANDIDATE_ITERS) + 1):
            g.update_graph(iteration=i)
            if i in CANDIDATE_ITERS:
                d = _dijkstra(g.graph, g.nodes, src, dst)
                dij_at[i] = d if d < float("inf") else None

        matched = [k for k in CANDIDATE_ITERS if _matches(dij_at[k], recorded)]
        for k in matched:
            match_counts[k] += 1
        if not matched:
            unmatched.append(f"{seed}/{sid}: recorded={recorded} vs {dij_at}")

        dij_4x = dij_at[40]
        reachable_4x = dij_4x is not None
        if not warm_strict and not reachable_4x:
            failures_unreachable.append(f"{seed}/{sid}")

        dij_1x = sweep_1x.get((seed, sid))
        in_1x_sweep = (seed, sid) in sweep_1x
        if in_1x_sweep:
            state_diff_n += 1
            if not _matches(dij_1x, recorded):
                state_diff_count += 1

        mstr = ",".join(f"i{k}" for k in matched) if matched else "NONE"
        def _f(v):
            return f"{v:9.2f}" if v is not None else f"{'unreach':>9}"
        print(f"{str(seed)+'/'+sid:<26} {str(warm_strict):>8} {_f(recorded)}"
              + "".join(" " + _f(dij_at[k]) for k in CANDIDATE_ITERS)
              + f" {mstr:>8} {_f(dij_1x) if in_1x_sweep else '   (n/a)'}"
              + f" {('YES' if in_1x_sweep and not _matches(dij_1x, recorded) else 'no') if in_1x_sweep else '-':>7}")

        new_records.append({
            "scale": "100x100",
            "tier": TIER,
            "seed": seed,
            "scenario_id": sid,
            "reachable": reachable_4x,
            "dijkstra_cost": dij_4x,
            "warm_strict": warm_strict,
            "recorded_dijkstra_cost": recorded,
            "recon_matches_recorded": 40 in matched,
            "matched_iterations": matched,
            "dijkstra_1x_state": dij_1x,
        })

    # ── Findings ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"Eval-state identification: matches per candidate state "
          f"(of {len(cells)} cells): "
          + ", ".join(f"iter-{k}: {match_counts[k]}" for k in CANDIDATE_ITERS))
    if unmatched:
        print(f"!! {len(unmatched)} cells matched NO candidate state:")
        for m in unmatched:
            print(f"   {m}")

    n_unreach_4x = sum(1 for r in new_records if not r["reachable"])
    n_fail = sum(1 for r in new_records if not r["warm_strict"])
    n_strict = len(new_records) - n_fail
    print(f"\n(a) Reachability on 4x (iteration-40) state: "
          f"{len(new_records) - n_unreach_4x}/{len(new_records)} reachable")
    if failures_unreachable:
        print(f"(b) Residual 4x failures that are UNREACHABLE cells: "
              f"{len(failures_unreachable)} -> {failures_unreachable}")
        n_solvable = len(new_records) - n_unreach_4x
        print(f"    Revised: {n_strict}/{len(new_records)} -> "
              f"{n_strict}/{n_solvable} over solvable cells")
    else:
        print(f"(b) None of the {n_fail} residual 4x failures is an unreachable "
              f"cell -- {n_strict}/{len(new_records)} stands on solvable cells.")
    print(f"(c) 1x vs 4x eval states differ on {state_diff_count}/{state_diff_n} "
          f"cells present in both files "
          f"(different recorded Dijkstra costs => different graph states).")

    # ── Append to audit JSON (replace any previous tier records) ─────────────
    audit = json.loads(AUDIT_JSON.read_text())
    audit = [r for r in audit if r.get("tier") != TIER]
    audit.extend(new_records)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2))
    print(f"\nAppended {len(new_records)} records (tier={TIER}) to {AUDIT_JSON} "
          f"({len(audit)} total).")


if __name__ == "__main__":
    main()
