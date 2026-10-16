"""item_h: trainable-parameter counts of the thesis agent (GraphSAGE encoder + Q-head).

Hidden size 64 is the 25x25 configuration (scripts/run_multi_seed_warm_vs_cold.py,
TRAIN_CFG["hidden_dim"]); 128 is the 50x50/100x100 configuration (configs/default.yaml,
scripts/run_sweep_50x50.py). Input node features: 4 (pyg_adapter). Q-head without the
goal-relative variant (3 concatenated embedding blocks). Deterministic; no training.
Writes runs/thesis_support/item_h_param_counts.json.
"""
import json, pathlib, sys, datetime

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch
from qwarm.agents.gnn_dqn import GraphSAGEEncoder, QHead

out = {"item": "h", "description": "Trainable parameter counts of the agent (encoder + Q-head), by hidden size.",
       "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "torch": torch.__version__,
       "node_in_dim": 4, "results": {}}
for d, scale in ((64, "25x25"), (128, "50x50 and 100x100")):
    enc, head = GraphSAGEEncoder(node_in_dim=4, hidden_dim=d), QHead(hidden_dim=d)
    pe = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    ph = sum(p.numel() for p in head.parameters() if p.requires_grad)
    gr = QHead(hidden_dim=d, goal_relative=True)
    out["results"][str(d)] = {"scale": scale, "encoder": pe, "q_head": ph, "total": pe + ph,
                              "q_head_goal_relative": sum(p.numel() for p in gr.parameters())}
path = ROOT / "runs" / "thesis_support" / "item_h_param_counts.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out["results"], indent=2))
