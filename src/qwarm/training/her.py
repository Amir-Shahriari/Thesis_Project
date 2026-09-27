"""Hindsight Experience Replay for goal-conditioned dynamic-graph navigation.

Motivation
----------
The fleet-mode pilot showed a warm agent reaching 16/20 of the goals it trained
on and 5/20 of held-out goals: it memorises its training destinations rather
than learning to route. Two things in the training loop make that the path of
least resistance -- expert demonstrations exist only for the training goals, and
the warm-only goal-adjacent seeding injects a "+100 at the doorstep" transition
for each training destination and for no other node.

HER attacks the data side of that. During training the agent reaches its
assigned goal on 1-3% of rollouts; the other 97% still reach *somewhere*. Those
trajectories are relabelled as successful demonstrations for the node they
actually arrived at, which turns a fixed set of training goals into a dense
sample of the node set at zero additional environment or oracle cost.

Why relabelling is exact here
-----------------------------
The reward of Eq. (reward) decomposes into a goal-independent step cost and a
goal-dependent terminal bonus:

    r(s, a, g) = -(w_dist + 0.1 w_time + c(v'))  +  100 * 1[v' = g]

so a stored step reward IS the step cost, and relabelling to goal g' only
requires adding 100 to the transition that arrives at g' and marking it
terminal. Nothing has to be re-simulated. Transitions whose next_state equals
their state are invalid-action terminations carrying the flat -5 penalty rather
than a step cost, so they are excluded -- relabelling them would fabricate a
step cost that never occurred.

Strategies
----------
    "future" (default)  for transition i, sample a relabel goal from the states
                        reached AFTER i in the same trajectory. The standard
                        choice in Andrychowicz et al. (2017); gives each
                        transition several goals at varying difficulty.
    "final"             use the last state of the trajectory as the goal for
                        every transition. Cheaper, lower coverage.
"""
from __future__ import annotations

import numpy as np

from qwarm.replay.expert_replay_buffer import Transition

GOAL_BONUS = 100.0


def relabel_trajectory(
    traj: list[tuple[str, str, float, str, list[str]]],
    k: int = 4,
    strategy: str = "future",
    rng: np.random.Generator | None = None,
    iteration: int = 0,
) -> list[Transition]:
    """Produce hindsight transitions from one failed (or any) trajectory.

    Args:
        traj: ordered list of (state, action, reward, next_state, valid_next)
            as executed. `reward` must be the environment's step reward.
        k: how many relabel goals to draw per transition ("future" strategy).
            k <= 0 disables relabelling entirely.
        strategy: "future" or "final".
        rng: generator for future-goal sampling; a fresh default is used if None.
        iteration: stamped onto the produced transitions for staleness tracking.

    Returns:
        New Transition objects tagged is_expert=False. They are ordinary online
        experience about a different goal, not demonstrations -- marking them
        expert would let them into the DQfD margin loss, which is reserved for
        oracle-verified actions.
    """
    if k <= 0 or len(traj) < 2:
        return []
    rng = rng or np.random.default_rng()

    # Real moves only: next_state == state marks an invalid-action termination
    # whose -5 reward is a penalty, not a traversal cost.
    moves = [(s, a, r, ns, vn) for (s, a, r, ns, vn) in traj if ns != s]
    if len(moves) < 2:
        return []

    out: list[Transition] = []
    n = len(moves)

    for i, (s, a, r, ns, vn) in enumerate(moves):
        if strategy == "final":
            targets = [n - 1]
        else:
            # j == i is allowed and is the valuable case: it makes the step
            # that arrived at this node a terminal success for that node as a
            # goal. The last transition is included for exactly that reason --
            # the node the agent actually ended at is the most informative
            # relabel the trajectory offers.
            hi = n - 1
            targets = rng.integers(i, hi + 1, size=min(k, hi - i + 1)).tolist()

        for j in set(int(t) for t in targets):
            goal = moves[j][3]          # next_state of the chosen future step
            if goal == s:               # goal == current state is degenerate
                continue
            reached = (j == i)
            out.append(Transition(
                state_node=s,
                action_node=a,
                reward=r + (GOAL_BONUS if reached else 0.0),
                next_state_node=ns,
                done=reached,
                valid_next_actions=[] if reached else vn,
                is_expert=False,
                iteration_added=iteration,
                goal_node=goal,
            ))
    return out
