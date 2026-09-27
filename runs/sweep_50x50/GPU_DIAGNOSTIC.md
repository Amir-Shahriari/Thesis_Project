# GPU Diagnostic — 50x50 Sweep
**Measured:** 2026-06-07 during live sweep (cell 2/25 active, PID 36508)  
**Analyst:** Static code analysis + nvidia-smi snapshots (no GPU kernels launched by diagnostic)

---

## Q1 — Device Placement

### Software stack
| Item | Value |
|------|-------|
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| Device count | 1 |
| Device name | NVIDIA GeForce RTX 5080 |
| VRAM total | 16303 MiB |
| `torch.compile` | **disabled on Windows** (triton unavailable; `maybe_compile` returns module unchanged when `sys.platform == "win32"`) |

**`torch.cuda.is_available()` → `True`**

### Model parameter device
The sweep's python.exe process (PID 36508) is listed as a CUDA compute app by nvidia-smi and holds ≥9 GB of VRAM during training. Because `GNNDQN.device` is determined at construction from `torch.cuda.is_available()`, and CUDA is available, all `nn.Module` parameters (`GraphSAGEEncoder`, `QHead`, target copies) are on `cuda:0`.

### PyG tensor device
`dynamic_graph_to_pyg` is called with `device=agent.device` (confirmed in `train_gnn_dqn.py:138`). All node-feature and edge-index tensors land on `cuda:0`.

---

## Q2 — Time Breakdown of `learn_from_batch`

**STATUS: DEFERRED — cannot safely profile while sweep is active.**

At measurement time the GPU was at 88–94% SM utilization with 30% memory-bandwidth. Running torch.profiler or timed `learn_from_batch` calls concurrently would:
- Preempt the live sweep's CUDA streams
- Produce measurements dominated by context-switch overhead

Live profiling will be valid only in a gap between cells or after the sweep completes.

**Static lower-bound estimates** (based on code structure, see Q4):

| Phase | Operations per grad step | Expected GPU time |
|-------|--------------------------|-------------------|
| GNN encoder (online) | 1 GraphSAGEEncoder forward on 2500 nodes | ~5–15 ms |
| GNN encoder (target) | 1 target encoder forward | ~5–15 ms |
| Q-head loop | **256 individual `_q_head_raw` forwards** | ~10–50 ms compute + ~2.56 ms Python overhead floor |
| Backward | 1 `.backward()` | ~5–15 ms |
| **Per-step total** | | **~25–95 ms estimated** |

At 7916 total grad steps for warm alone, pure compute would be 198–751 s. The 11 702 s wall time suggests the bulk of time is in **rollout + oracle calls**, not grad steps (see Q5).

---

## Q3 — GPU Utilization During Active Training

Measurements taken with `nvidia-smi` while cell 2/25 was running:

| Sample | SM util | MEM bandwidth | VRAM used / total |
|--------|---------|---------------|-------------------|
| 1 | 90% | 24% | 9048 / 16303 MiB |
| 2 | 94% | 29% | — |
| 3 | 88% | 25% | — |
| 4 | 89% | 30% | 9161 / 16303 MiB |

**The GPU IS being used.** SM utilization is high (88–94%). Memory bandwidth is moderate (24–30%), consistent with small-tensor operations (tiny Q-head inputs) rather than large matrix multiplications.

### Critical finding: GPU is shared with Rainbow Six Siege
`nvidia-smi --query-compute-apps` showed **PID 7016 = `RainbowSix.exe`** in the CUDA compute context simultaneously with PID 36508 (`python.exe`). A running game competes for:
- GPU time slices (OS CUDA scheduler preempts training kernels every ~16 ms for frame rendering)
- VRAM (game framebuffer + assets consume 3–6 GB, reducing what PyTorch can cache)
- PCIe bandwidth (game + training both doing CPU↔GPU transfers)

**Estimated impact:** with Rainbow Six consuming 20–40% of GPU throughput, training wall time is inflated by 1.25–1.67× relative to isolated execution. At cell 1's 11 702 s, isolated training time is likely **7 000–9 400 s** — still very slow (see Q5).

---

## Q4 — Batch Structure: GNN Forward Count

**Answer: GNN encoder runs ONCE per batch. Q-head runs ONCE PER TRANSITION (Python loop).**

Confirmed from `src/qwarm/agents/gnn_dqn.py`, `learn_from_batch`:

```python
# --- ENCODER: one forward pass on the full graph ---
self._encoder_raw.train()
emb = self._encoder_raw(x, ei)          # 1 forward, all 2500 nodes

with torch.no_grad():
    self.target_encoder.eval()           # S4 fix
    t_emb = self.target_encoder(x, ei)  # 1 forward, all 2500 nodes

# --- Q-HEAD: 256 individual forwards in a Python loop ---
for t in batch:                          # len(batch) == 256
    h_cur  = emb[node_idx[t.state_node]]
    h_act  = emb[node_idx[t.action_node]]
    h_goal = emb[node_idx[goal_idx]]
    q_pred = self._q_head_raw(h_cur, h_act, h_goal).squeeze()   # ← separate kernel per transition
    ...
    if t.is_expert and ...:
        q_bc = self._q_head_raw(h_cur, h_act_bc, h_goal).squeeze()  # possible second call
    loss_terms.append(F.mse_loss(q_pred, target_val.detach()))

td_loss = torch.stack(loss_terms).mean()
```

Each `_q_head_raw(h_cur, h_act, h_goal)` call:
- Input: three vectors of shape `[hidden_dim=128]`, concatenated to `[384]`
- Passes through 3 linear layers in the MLP Q-head
- Returns a scalar `[1]` tensor
- Launches **~6 CUDA kernels** (matmul + bias + activation for each of 3 layers)

For batch_size=256: **≥1536 tiny CUDA kernel launches per grad step** — each for a 384-element input. At this scale, the per-kernel launch overhead (~5–10 µs each on modern CUDA) rivals or exceeds the actual computation time.

**`torch.compile` cannot fuse these** because it is disabled on Windows (the only fusion path that could coalesce the loop).

---

## Q5 — Verdict

| Question | Answer |
|----------|--------|
| GPU or CPU? | **GPU** — RTX 5080 via CUDA 12.8, confirmed by nvidia-smi and CUDA availability |
| Compute-bound or host-overhead-bound? | **Host-overhead-bound + Oracle-bound** (see below) |

### Primary bottlenecks (estimated, in priority order)

**1. Oracle calls — likely dominant for warm agent**

`train_gnn_dqn` calls each oracle in `re_seed_experts_each_iteration` once per iteration. With 3 oracles and 10 iterations:
- ClassicalAStar: ~negligible
- QuantumInspiredStochasticOracle: O(seconds) per call
- FaithfulSimulatedQAOA: O(10–300 s) per call × 10 iterations = **100–3000 s** per warm agent

Pre-seeding (`discover_all_paths` with n_perturbation_states=3): 3 more oracle calls per oracle type = additional **30–900 s**.

**Total oracle overhead per warm cell: 130–3900 s (estimated)**

**2. Python loop in Q-head — host-overhead-bound**

256 individual `_q_head_raw` forwards per grad step × ≥8000 grad steps per agent × 2 agents:
- Python loop iterations: ≥4.1 million
- CUDA kernel launches: ≥6.1 million tiny launches
- Python interpreter overhead: ~10–50 µs × 4.1M = **41–205 s** just from Python overhead
- Actual computation: minor (384-element MLP), dominated by launch overhead

**3. Rainbow Six Siege GPU preemption**

Game running concurrently inflates wall time by ~25–67% due to CUDA scheduler preemption.

**4. Rollout inefficiency (minor)**

Rollout uses cached embeddings (`agent.encode()` once per iteration), so each `choose_action` only runs the Q-head on ~8 valid actions — fast. Rollout is not the bottleneck.

### Why cell 1 took 11 702 s

Best estimate for warm agent breakdown:
| Component | Time estimate |
|-----------|--------------|
| FaithfulQAOA oracle calls (warm only) | 500–2000 s |
| Q-head Python loop (grad steps) | 100–400 s |
| GNN encoder forward passes | 50–150 s |
| Rollout (200 eps × 40 steps × 10 iters) | 200–800 s |
| Graph perturbation + PyG conversion | 20–50 s |
| **Warm subtotal** | ~870–3400 s |
| **Cold agent (no oracle overhead)** | ~370–1400 s |
| **Rainbow Six preemption (~30% inflation)** | ~370–1440 s |
| **Total estimated** | **1600–6240 s** |

The upper end of this estimate (6 240 s) is still short of the measured 11 702 s. The remaining gap suggests FaithfulQAOA calls take longer than the upper estimates (potentially 100–300 s each), or the rollout has additional overhead not accounted for (e.g., graph perturbation touching NetworkX data structures in Python).

### Recommendation

**Close Rainbow Six Siege during the sweep** — this is the only immediately actionable fix. It will reduce wall time by an estimated 25–67% per cell and eliminate CUDA preemption noise.

For the remaining cells (23 cells × 11 702 s ≈ 3.1 days at current rate, ≈ 1.8–2.3 days with game closed), the sweep will complete within the 5-day deadline.

After the sweep completes, profiling FaithfulQAOA and Q-head loop timing in isolation will confirm these estimates and identify whether to restructure the Q-head into a batched forward pass for future runs.

---

## Appendix: Live Measurement Plan (post-sweep)

When the sweep finishes and the GPU is free, run:

```python
import torch, time
from qwarm.agents.gnn_dqn import GNNDQN
# ... build minimal 50x50 graph, prefill buffer ...

for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = agent.learn_from_batch(batch, data, goal_node=goal)
    torch.cuda.synchronize()
    print(f"learn_from_batch: {(time.perf_counter()-t0)*1000:.1f} ms")
```

And torch.profiler with `ProfilerActivity.CUDA` to split CPU vs CUDA time precisely.
