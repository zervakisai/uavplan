import numpy as np

from uavbench.planners import PLANNERS


def test_new_planners_return_planresult():
    heightmap = np.zeros((20, 20), dtype=np.float32)
    no_fly = np.zeros((20, 20), dtype=bool)
    start = (0, 0)
    goal = (19, 19)

    planner_ids = ["dstar_lite", "ad_star", "dwa", "mppi"]

    for pid in planner_ids:
        planner = PLANNERS[pid](heightmap, no_fly)
        result = planner.plan(start, goal)
        assert hasattr(result, "success")
        assert hasattr(result, "path")
        assert isinstance(result.path, list)
