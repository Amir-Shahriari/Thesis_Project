#!/usr/bin/env python3
"""
verify_oracle_checkpoints.py -- Gate check before building oracle-source demo mode.

Evaluates quantum_only / classical_only / full_pool on the 25x25 evaluated
scenarios (s0-s4) through the same inference path the demo server uses, then:

  1. Prints arm x scenario table: reached (strict), cost_ratio.
  2. Runs exact McNemar test on all arm pairs (warm-vs-warm and each-vs-cold).
  3. Checks the paper's source-invariance claim: every warm arm should reach
     every solvable cell (McNemar p=1.0 on all warm-vs-warm comparisons).
  4. Confirms full_pool results match the manifest's recorded warm outcomes.

Exit code 0 = gate passes; 1 = gate fails.  Pass --no-exit to suppress sys.exit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.metrics import evaluate_with_reasonableness
from qwarm.utils.device import resolve_device

# ---------------------------------------------------------------------------
# Manifest / scenario config
# ---------------------------------------------------------------------------

MANIFEST_PATH = ROOT / "demo_agents" / "manifest.json"
DEMO_AGENTS_DIR = ROOT / "demo_agents"
TRACES_ROOT = ROOT / "runs" / "traces_25x25"
SEED = 42   # training seed used for all demo scenarios in the ablation

SCENARIOS_25X25 = [
    ("s0", "Node_537", "Node_54"),
    ("s1", "Node_476", "Node_449"),
    ("s2", "Node_81",  "Node_525"),
    ("s3", "Node_274", "Node_101"),
    ("s4", "Node_427", "Node_398"),
]
GRID_SEED = 191664964
N_ITERS   = 5

def _build_graph() -> DynamicGraph:
    g = DynamicGraph(
        grid_width=25, grid_height=25,
        extra_edges=2, deactivate_prob=0.15,
        seed=GRID_SEED,
    )
    for i in range(1, N_ITERS + 1):
        g.update_graph(iteration=i)
    return g


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors server._load_checkpoint)
# ---------------------------------------------------------------------------

def _load(path: pathlib.Path, device: torch.device) -> GNNDQN:
    ck = torch.load(str(path), map_location=device, weights_only=False)
    if "encoder_raw_state_dict" in ck:
        enc_sd  = ck["encoder_raw_state_dict"]
        qh_sd   = ck["q_head_state_dict"]
        hidden  = ck["hidden_dim"]
        node_in = ck["node_in_dim"]
    else:
        enc_sd  = ck["encoder"]
        qh_sd   = ck["q_head"]
        w = enc_sd.get("conv1.lin_l.weight") or enc_sd.get("conv1.lin.weight")
        hidden  = w.shape[0] if w is not None else 64
        node_in = w.shape[1] if w is not None else 4
    agent = GNNDQN(node_in_dim=node_in, hidden_dim=hidden, device=device, seed=0)
    agent._encoder_raw.load_state_dict(enc_sd)
    agent._q_head_raw.load_state_dict(qh_sd)
    agent.update_target()
    agent._encoder_raw.eval()
    agent._q_head_raw.eval()
    return agent


def _checkpoint_paths(traces_root: pathlib.Path, seed: int) -> dict[str, list[pathlib.Path]]:
    """Return {arm: [path_s0, path_s1, ...]} for the three new arms + cold."""
    scen_ids = [f"seed{GRID_SEED}_s{i}" for i in range(5)]
    manifest = json.loads(MANIFEST_PATH.read_text())
    sc25 = manifest["25x25"]

    paths: dict[str, list[pathlib.Path]] = {}

    # full_pool: existing demo warm agents
    paths["full_pool"] = [
        DEMO_AGENTS_DIR / sc25["scenarios"][i]["warm"]
        for i in range(5)
    ]

    # cold: existing demo cold agents
    paths["cold"] = [
        DEMO_AGENTS_DIR / sc25["scenarios"][i]["cold"]
        for i in range(5)
    ]

    # classical_only and quantum_only: from traces dir (retrained with seed=42)
    for arm in ("classical_only", "quantum_only"):
        arm_paths = []
        for sid in scen_ids:
            p = traces_root / f"seed{seed}_{sid}" / arm / "agent.pt"
            arm_paths.append(p)
        paths[arm] = arm_paths

    return paths


# ---------------------------------------------------------------------------
# McNemar exact test (identical to ablation script)
# ---------------------------------------------------------------------------

def _mcnemar(reach_a: list[bool], reach_b: list[bool]) -> dict:
    b = sum(1 for x, y in zip(reach_a, reach_b) if x and not y)
    c = sum(1 for x, y in zip(reach_a, reach_b) if y and not x)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binom
        p = min(1.0, 2.0 * float(binom.cdf(min(b, c), n, 0.5)))
    return {"b": b, "c": c, "n_disc": n, "p": p}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", default=str(TRACES_ROOT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-exit", action="store_true",
                        help="Print results but do not call sys.exit(1) on failure.")
    args = parser.parse_args()

    traces_root = pathlib.Path(args.traces_root)
    device = resolve_device("auto")
    print(f"Device: {device}\n")

    paths = _checkpoint_paths(traces_root, args.seed)

    # Verify all checkpoints exist before starting
    missing: list[str] = []
    for arm, arm_paths in paths.items():
        for i, p in enumerate(arm_paths):
            if not p.exists():
                missing.append(f"  {arm:<16} s{i}  {p}")
    if missing:
        print("ERROR: missing checkpoints:", flush=True)
        print("\n".join(missing))
        return 1

    # Build graph once (same for all scenarios and arms)
    print("Building 25x25 demo graph (seed=191664964, 5 iterations)...", flush=True)
    g = _build_graph()
    data = dynamic_graph_to_pyg(g, device=device)

    # ---------------------------------------------------------------------------
    # Evaluate all arms x scenarios
    # ---------------------------------------------------------------------------
    ARMS = ["full_pool", "classical_only", "quantum_only", "cold"]
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}

    for arm in ARMS:
        print(f"\n[{arm}]", flush=True)
        for i, (sc_tag, src, dst) in enumerate(SCENARIOS_25X25):
            agent = _load(paths[arm][i], device)
            v = evaluate_with_reasonableness(g, agent, src, dst, k_threshold=3.0, data=data)
            ratio_str = f"{v.cost_ratio:.2f}x" if v.cost_ratio is not None else " inf "
            reached_str = "REACH" if v.reached_goal_strict else "FAIL "
            print(f"  {sc_tag}  {reached_str}  cost={v.cost:>8.1f}  ratio={ratio_str}  "
                  f"src={src}  dst={dst}", flush=True)
            results[arm].append({
                "sc": sc_tag,
                "src": src, "dst": dst,
                "reached": v.reached_goal_strict,
                "cost": v.cost,
                "ratio": v.cost_ratio,
            })
        del agent  # free memory between arms

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  ORACLE-SOURCE VERIFICATION TABLE  (25x25, seed=42)")
    print("=" * 72)
    hdr = f"{'scenario':<10}" + "".join(f"  {a:<16}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    for i, (sc_tag, src, dst) in enumerate(SCENARIOS_25X25):
        row = f"{sc_tag:<10}"
        for arm in ARMS:
            r = results[arm][i]
            tag = "REACH" if r["reached"] else "FAIL "
            ratio = f"{r['ratio']:.1f}x" if r["ratio"] is not None else " inf"
            row += f"  {tag} {ratio:<7}    "
        print(row)
    print("=" * 72)

    # ---------------------------------------------------------------------------
    # Gate checks
    # ---------------------------------------------------------------------------
    gate_ok = True
    print("\n--- Gate checks ---\n")

    # 1. Full_pool consistency with manifest
    manifest = json.loads(MANIFEST_PATH.read_text())
    sc25 = manifest["25x25"]["scenarios"]
    fp_mismatches = []
    for i, r in enumerate(results["full_pool"]):
        expected = sc25[i]["warm_reached"]
        actual   = r["reached"]
        if expected != actual:
            fp_mismatches.append(f"  s{i}: manifest says warm_reached={expected}, got {actual}")
    if fp_mismatches:
        print("FAIL  full_pool does NOT match manifest warm_reached:")
        for m in fp_mismatches: print(m)
        gate_ok = False
    else:
        print("PASS  full_pool results match manifest warm_reached on all 5 scenarios.")

    # 2. Source-invariance: every warm arm reaches every solvable cell
    #    Paper claim: McNemar p=1.0 on all warm-vs-warm pairs.
    #    A "solvable" cell is one where at least one warm arm reaches.
    solvable = [
        i for i in range(5)
        if any(results[a][i]["reached"] for a in ["full_pool", "classical_only", "quantum_only"])
    ]
    print(f"\nSolvable cells: {[f's{i}' for i in solvable]}  (n={len(solvable)})")

    warm_pairs = [
        ("full_pool", "classical_only"),
        ("full_pool", "quantum_only"),
        ("classical_only", "quantum_only"),
    ]
    for a, b in warm_pairs:
        reach_a = [results[a][i]["reached"] for i in solvable]
        reach_b = [results[b][i]["reached"] for i in solvable]
        mc = _mcnemar(reach_a, reach_b)
        status = "PASS" if mc["p"] == 1.0 else "FAIL"
        print(f"{status}  {a} vs {b}: McNemar p={mc['p']:.4f}  "
              f"(discordant pairs: {a} only={mc['b']}, {b} only={mc['c']})")
        if status == "FAIL":
            gate_ok = False

    # 3. Each warm arm vs cold
    print()
    for arm in ("full_pool", "classical_only", "quantum_only"):
        reach_w = [results[arm][i]["reached"] for i in solvable]
        reach_c = [results["cold"][i]["reached"] for i in solvable]
        mc = _mcnemar(reach_w, reach_c)
        n_w = sum(reach_w)
        n_c = sum(reach_c)
        print(f"  {arm} vs cold:  reach={n_w}/{len(solvable)} vs {n_c}/{len(solvable)}  "
              f"McNemar p={mc['p']:.4f}")

    # 4. Any warm arm that fails to reach where it should
    print()
    for arm in ("full_pool", "classical_only", "quantum_only"):
        fails = [f"s{i}" for i in solvable if not results[arm][i]["reached"]]
        if fails:
            print(f"FAIL  {arm} fails to reach on solvable cells: {fails}")
            gate_ok = False
        else:
            print(f"PASS  {arm} reaches all {len(solvable)} solvable cells.")

    print("\n" + "=" * 72)
    if gate_ok:
        print("  GATE: PASSED -- safe to build oracle-source UI.")
    else:
        print("  GATE: FAILED -- do NOT build oracle-source UI. See failures above.")
    print("=" * 72)

    if not gate_ok and not args.no_exit:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
