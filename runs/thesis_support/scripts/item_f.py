"""Measure the per-event re-encode cost of the within-trajectory hook (thesis ch3:279, '~20 ms on the test hardware').

Times exactly what the rollout does when env.graph_changed is set (train_gnn_dqn.py:278-281,
run_v3_realtime_eval.py:165-167): dynamic_graph_to_pyg(graph, device) + agent.encode(data), with CUDA synchronisation.
Uses the 25x25 preset (extra_edges=2), hidden_dim 64 as in run_multi_seed_warm_vs_cold.py, and a real 25x25 checkpoint
from demo_agents/ when loadable. Read-only on Thesis_Project (src imported, nothing written there).
Run (from the repo root, with a CUDA torch build, e.g. torch 2.11.0+cu128):
    python -B runs/thesis_support/scripts/item_f.py
Output: runs/thesis_support/item_f_reencode_timing.json (produced 2026-10-10 on the RTX 5080 by this code).
"""
import json, platform, statistics, sys, time, pathlib
sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import torch
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.agents.gnn_dqn import GNNDQN

OUT = pathlib.Path(__file__).resolve().parents[1] / "item_f_reencode_timing.json"
res = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "python": platform.python_version(), "platform": platform.platform(), "results": {}}


def run(device, n=300, warm=30):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=2, seed=191664964) if "seed" in DynamicGraph.__init__.__code__.co_varnames else DynamicGraph(grid_width=25, grid_height=25, extra_edges=2)
    agent = GNNDQN(node_in_dim=4, hidden_dim=64, seed=0, device=device)
    ck = pathlib.Path(r"C:\Users\amirh\Desktop\Thesis_Project\demo_agents\warm_25x25_s0.pt")
    loaded = False
    if ck.exists():
        try:
            sd = torch.load(ck, map_location=agent.device, weights_only=False)
            enc = sd.get("encoder") or sd.get("encoder_state_dict")
            if enc is not None:
                agent._encoder_raw.load_state_dict(enc)
                loaded = True
        except Exception as e:  # noqa: BLE001
            loaded = f"not loaded: {type(e).__name__}"
    agent._encoder_raw.eval()
    conv, total = [], []
    for k in range(warm + n):
        if hasattr(g, "update_graph"):
            g.update_graph()  # a perturbation event changes the graph, as in the hook
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        data = dynamic_graph_to_pyg(g, device=agent.device)
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            agent.encode(data)
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if k >= warm:
            conv.append(1000 * (t1 - t0))
            total.append(1000 * (t2 - t0))
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    return {"n": n, "checkpoint_loaded": loaded, "nodes": data.num_nodes, "edges": int(data.edge_index.shape[1]),
            "convert_ms_median": round(statistics.median(conv), 3),
            "total_ms_median": round(statistics.median(total), 3), "total_ms_p10": round(q(total, .1), 3),
            "total_ms_p90": round(q(total, .9), 3), "total_ms_mean": round(statistics.mean(total), 3)}


if torch.cuda.is_available():
    res["results"]["cuda"] = run("cuda")
res["results"]["cpu"] = run("cpu")
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
