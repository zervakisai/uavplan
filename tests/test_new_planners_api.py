import numpy as np

from uavbench.planners import PLANNERS


def test_new_planners_return_planresult():
    heightmap = np.zeros((20, 20), dtype=np.float32)
    no_fly = np.zeros((20, 20), dtype=bool)
    start = (0, 0)
    goal = (19, 19)

    planner_ids = [
        "dstar_lite",
        "lpa_star",
        "ad_star",
        "ara_star",
        "local_dwa",
        "local_teb_lite",
        "hybrid_dstar_dwa",
        "hybrid_dstar_teb_lite",
        "risk_mpc",
    ]

    for pid in planner_ids:
        planner = PLANNERS[pid](heightmap, no_fly)
        result = planner.plan(start, goal)
        assert hasattr(result, "success")
        assert hasattr(result, "path")
        assert isinstance(result.path, list)
