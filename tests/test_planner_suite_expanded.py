import numpy as np

from uavbench.planners import PLANNERS


def test_six_planners_registered():
    expected = {"astar", "theta_star", "dstar_lite", "ad_star", "dwa", "mppi"}
    assert set(PLANNERS.keys()) == expected


def test_all_planners_basic_pathfinding():
    h = np.zeros((20, 20), dtype=float)
    nfz = np.zeros((20, 20), dtype=bool)
    # small obstacle wall with a gap
    h[10, 2:18] = 1.0
    h[10, 10] = 0.0

    for planner_id in sorted(PLANNERS.keys()):
        planner = PLANNERS[planner_id](h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success, f"{planner_id} failed: {result.reason}"
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (19, 19)
