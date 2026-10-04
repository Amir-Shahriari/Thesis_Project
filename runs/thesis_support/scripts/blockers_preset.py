"""Quantify the two pre-pilot blockers.

B1: get_valid_actions does NOT filter visited nodes, so a revisit costs -5
    and ends the episode. How long does a random rollout survive, by degree?
B2: a fixed +100 bonus against normalised step cost makes the
    reach-vs-efficiency weighting vary along the density axis being swept.
    Does bonus = beta * E[C*] remove that?
"""
import heapq
import sys

sys.path.insert(0, r"C:\Users\amirh\Desktop\Thesis_Project\src")
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph

STEP = lambda g, u, v: (g.graph[u][v]["distance"]
                        + 0.1 * g.graph[u][v]["time"]
                        + g.nodes[v]["node_penalty"])


def dijkstra(g, src, dst):
    dist, prev, pq, seen = {src: 0.0}, {}, [(0.0, src)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, e in g.graph[u].items():
            if not (e["active"] and g.nodes[v]["active"]):
                continue
            nd = d + STEP(g, u, v)
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, None
    path, u = [dst], dst
    while u != src:
        u = prev[u]
        path.append(u)
    return path[::-1], dist[dst]


def rollout(g, src, dst, masked, rng, max_steps=400):
    """Uniform-random rollout, i.e. the eps=1 limit of eps-greedy."""
    cur, visited, steps = src, {src}, 0
    while steps < max_steps:
        acts = [n for n, e in g.graph[cur].items()
                if e["active"] and g.nodes[n]["active"]]
        if masked:
            acts = [a for a in acts if a not in visited]
        if not acts:
            return "dead-end", steps
        a = acts[rng.integers(len(acts))]
        if a in visited:                      # reactive -5, terminal
            return "revisit", steps
        steps += 1
        cur = a
        visited.add(a)
        if cur == dst:
            return "goal", steps
    return "timeout", steps


print("#" * 76)
print("# BLOCKER 1 -- random-rollout outcome, reactive revisit vs masked")
print("#" * 76)
for nc, lab in ((1250, "n_chords=1250  degree 7.8, diam 5"),
                (60, "n_chords=  60  degree 4.0, diam 15"),
                (0, "n_chords=   0  degree 3.8, diam 47")):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=(2 if nc == 1250 else 0), n_chords=(None if nc == 1250 else nc), seed=191664964)
    nodes = list(g.nodes)
    rng = np.random.default_rng(3)
    pairs = []
    while len(pairs) < 20:
        s, d = rng.choice(nodes, 2, replace=False)
        if dijkstra(g, str(s), str(d))[0]:
            pairs.append((str(s), str(d)))
    for masked in (False, True):
        out, steps = {}, []
        for s, d in pairs:
            for _ in range(100):
                o, k = rollout(g, s, d, masked, rng)
                out[o] = out.get(o, 0) + 1
                steps.append(k)
        n = sum(out.values())
        tag = "MASKED  " if masked else "REACTIVE"
        print(f"  {lab} | {tag}  goal {100*out.get('goal',0)/n:5.1f}%  "
              f"revisit-death {100*out.get('revisit',0)/n:5.1f}%  "
              f"dead-end {100*out.get('dead-end',0)/n:5.1f}%  "
              f"median steps {np.median(steps):.0f}")
    print()

print("#" * 76)
print("# BLOCKER 2 -- bonus/cost balance across the density axis")
print("#" * 76)
print(f"\n  {'n_chords':>9} {'E[C*] (norm)':>13} {'fixed bonus 100':>17}"
      f" {'beta=5 bonus':>14} {'ret b=5':>9} {'>-5':>6}")
for nc in (0, 60, 125, 400, 1250):
    g = DynamicGraph(grid_width=25, grid_height=25, extra_edges=(2 if nc == 1250 else 0), n_chords=(None if nc == 1250 else nc), seed=191664964)
    nodes = list(g.nodes)
    rng = np.random.default_rng(7)
    paths = []
    while len(paths) < 25:
        s, d = rng.choice(nodes, 2, replace=False)
        p, c = dijkstra(g, str(s), str(d))
        if p:
            paths.append(p)
    ms = np.mean([STEP(g, u, v) for p in paths for u, v in zip(p, p[1:])])
    cstar = np.mean([sum(STEP(g, u, v) for u, v in zip(p, p[1:])) / ms
                     for p in paths])
    bonus = 5.0 * cstar
    rets = []
    for p in paths:
        R, disc = 0.0, 1.0
        for u, v in zip(p, p[1:]):
            r = -STEP(g, u, v) / ms + (bonus if v == p[-1] else 0.0)
            R += disc * r
            disc *= 0.99
        rets.append(R)
    print(f"  {nc:>9} {cstar:>13.1f} {100/cstar:>16.1f}x {bonus:>14.0f}"
          f" {np.median(rets):>9.1f} {sum(r>-5 for r in rets):>4d}/25")
print("\n  'fixed bonus 100' column = bonus-to-optimal-cost ratio; it must be"
      "\n  constant across the axis or the reward is a confound.")
