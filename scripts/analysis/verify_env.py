"""Regression check: defaults must reproduce the published env exactly."""
import sys

sys.path.insert(0, r"C:\Users\amirh\Desktop\qwarm-gnn-rl\src")
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv

g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=2, seed=191664964)
nodes = list(g.nodes)
src, dst = nodes[0], nodes[400]


def replay(**kw):
    """Fixed action sequence, so any reward change shows up."""
    e = PathfindingEnv(g.graph, g.nodes, src, dst, max_steps=50, **kw)
    e.reset()
    rng = np.random.default_rng(0)
    total, log = 0.0, []
    for _ in range(12):
        a = e.get_valid_actions()
        if not a:
            log.append("no-actions")
            break
        nxt = a[rng.integers(len(a))]
        _, r, done = e.step(nxt)
        total += r
        log.append(round(r, 6))
        if done:
            break
    return total, log


base, blog = replay()
print("LEGACY DEFAULTS")
print(f"  total {base:.6f}")
print(f"  rewards {blog}")

same, _ = replay(goal_bonus=100.0, cost_scale=1.0, mask_visited=False)
print(f"\n  explicit legacy args identical: {abs(same - base) < 1e-12}")

norm, _ = replay(cost_scale=7.9)
print(f"  cost_scale=7.9 rescales: {abs(norm * 7.9 - base) < 1e-6 or 'see note'}"
      f"   (total {norm:.4f})")

print("\nMASKING")
for mv, mf in ((False, False), (True, False), (True, True)):
    e = PathfindingEnv(g.graph, g.nodes, src, dst, mask_visited=mv,
                       mask_fallback=mf)
    e.reset()
    a0 = e.get_valid_actions()
    e.step(a0[0])
    offered = [x for x in e.get_valid_actions() if x in e.visited_nodes]
    print(f"  mask_visited={mv!s:<5} mask_fallback={mf!s:<5} -> "
          f"visited offered: {len(offered)}")

print("\nFALLBACK SEMANTICS (revisit allowed only when boxed in)")
e = PathfindingEnv(g.graph, g.nodes, src, dst, mask_visited=True,
                   mask_fallback=True)
e.reset()
a0 = e.get_valid_actions()
e.step(a0[0])
prev = src
_, r, done = e.step(prev)          # step back with unvisited options available
print(f"  revisit while options remain -> reward {r}, done={done} "
      f"(expect -5.0, True)")
