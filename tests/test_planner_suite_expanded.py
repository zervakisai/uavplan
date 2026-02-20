import warnings

import numpy as np

from uavbench.planners import (
    PLANNERS,
    PAPER_PLANNERS,
    DEPRECATED_ALIASES,
    DEPRECATED_PLANNERS,
)


def test_paper_suite_has_six_planners():
    """Paper suite contains exactly 6 planners."""
    assert len(PAPER_PLANNERS) == 6


def test_registry_has_paper_plus_deprecated():
    """Registry = paper suite + deprecated planner + deprecated aliases."""
    expected = set(PAPER_PLANNERS) | DEPRECATED_PLANNERS | set(DEPRECATED_ALIASES)
    assert set(PLANNERS.keys()) == expected


def test_all_paper_planners_basic_pathfinding():
    """All paper-suite planners solve a walled grid with gap."""
    h = np.zeros((20, 20), dtype=float)
    nfz = np.zeros((20, 20), dtype=bool)
    h[10, 2:18] = 1.0
    h[10, 10] = 0.0  # gap

    for planner_id in PAPER_PLANNERS:
        planner = PLANNERS[planner_id](h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success, f"{planner_id} failed: {result.reason}"
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (19, 19)


def test_deprecated_planners_still_solve():
    """Deprecated planners still work (with warning) for one release cycle."""
    h = np.zeros((20, 20), dtype=float)
    nfz = np.zeros((20, 20), dtype=bool)

    all_deprecated = sorted(set(DEPRECATED_ALIASES) | DEPRECATED_PLANNERS)
    for planner_id in all_deprecated:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            planner = PLANNERS[planner_id](h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success, f"deprecated planner {planner_id} failed: {result.reason}"
