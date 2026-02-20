import warnings

import numpy as np

from uavbench.planners import PLANNERS, PAPER_PLANNERS


def test_paper_planners_return_planresult():
    """All paper-suite planners return PlanResult with expected fields."""
    heightmap = np.zeros((20, 20), dtype=np.float32)
    no_fly = np.zeros((20, 20), dtype=bool)
    start = (0, 0)
    goal = (19, 19)

    for pid in PAPER_PLANNERS:
        planner = PLANNERS[pid](heightmap, no_fly)
        result = planner.plan(start, goal)
        assert hasattr(result, "success"), f"{pid}: missing 'success'"
        assert hasattr(result, "path"), f"{pid}: missing 'path'"
        assert isinstance(result.path, list), f"{pid}: path is not list"


def test_deprecated_planners_still_work():
    """Deprecated planner keys still return working planners (with warning)."""
    heightmap = np.zeros((20, 20), dtype=np.float32)
    no_fly = np.zeros((20, 20), dtype=bool)

    deprecated_ids = ["dstar_lite", "ad_star", "dwa", "mppi"]
    for pid in deprecated_ids:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            planner = PLANNERS[pid](heightmap, no_fly)
        result = planner.plan((0, 0), (19, 19))
        assert hasattr(result, "success")
        assert hasattr(result, "path")
        assert isinstance(result.path, list)
