"""Hindsight relabelling must not add the goal bonus twice."""
import numpy as np

from qwarm.training.her import GOAL_BONUS, relabel_trajectory

STEP = -8.0


def _success_traj():
    # A -> B -> C -> D, where D is the episode's goal: the last stored reward
    # already includes the +100 bonus.
    return [
        ("A", "B", STEP, "B", ["C"]),
        ("B", "C", STEP, "C", ["D"]),
        ("C", "D", STEP + GOAL_BONUS, "D", []),
    ]


def test_relabel_skips_original_goal():
    out = relabel_trajectory(_success_traj(), k=4, rng=np.random.default_rng(0),
                             original_goal="D")
    assert out
    assert all(t.goal_node != "D" for t in out)
    assert max(t.reward for t in out) == STEP + GOAL_BONUS


def test_relabel_without_original_goal_double_counts():
    # Documents the behaviour the original_goal argument exists to prevent.
    out = relabel_trajectory(_success_traj(), k=4, strategy="final",
                             original_goal=None)
    final = [t for t in out if t.done and t.goal_node == "D"]
    assert final and final[0].reward == STEP + 2 * GOAL_BONUS


def test_failed_trajectory_unaffected():
    traj = [("A", "B", STEP, "B", ["C"]), ("B", "C", STEP, "C", ["E"])]
    a = relabel_trajectory(traj, k=4, rng=np.random.default_rng(1))
    b = relabel_trajectory(traj, k=4, rng=np.random.default_rng(1), original_goal="D")
    assert [(t.state_node, t.goal_node, t.reward) for t in a] == \
           [(t.state_node, t.goal_node, t.reward) for t in b]
    assert all(t.reward in (STEP, STEP + GOAL_BONUS) for t in b)
