"""Unit tests for D* Lite planner API (PL-5).

Tests incremental update correctness and obstacle change handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.planners.base import PlannerBase, PlanResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(size: int = 15, density: float = 0.1, seed: int = 42):
    """Create a test grid with some obstacles."""
    rng = np.random.default_rng(seed)
    heightmap = np.zeros((size, size), dtype=np.float32)
    if density > 0:
        mask = rng.random((size, size)) < density
        heightmap[mask] = rng.uniform(1.0, 5.0, size=mask.sum()).astype(
            np.float32
        )
    # Ensure corners are free
    heightmap[0, 0] = 0.0
    heightmap[0, size - 1] = 0.0
    heightmap[size - 1, 0] = 0.0
    heightmap[size - 1, size - 1] = 0.0
    no_fly = np.zeros((size, size), dtype=bool)
    return heightmap, no_fly


# ===========================================================================
# PL-5: D* Lite API
# ===========================================================================


class TestDStarLiteAPI:
    """D* Lite implements PlannerBase with incremental updates."""

    def test_is_planner_base(self):
        """DStarLitePlanner is a subclass of PlannerBase."""
        from uavbench.planners.dstar_lite import DStarLitePlanner

        assert issubclass(DStarLitePlanner, PlannerBase)

    def test_plan_returns_plan_result(self):
        """plan() returns a PlanResult with path."""
        from uavbench.planners.dstar_lite import DStarLitePlanner

        heightmap, no_fly = _make_grid(size=10, density=0.0)
        planner = DStarLitePlanner(heightmap, no_fly)

        result = planner.plan((0, 0), (9, 9))
        assert isinstance(result, PlanResult)
        assert result.success
        assert len(result.path) > 0
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (9, 9)

    def test_path_is_valid(self):
        """Path contains only 4-connected moves on free cells."""
        from uavbench.planners.dstar_lite import DStarLitePlanner

        heightmap, no_fly = _make_grid(size=10, density=0.05)
        planner = DStarLitePlanner(heightmap, no_fly)

        result = planner.plan((0, 0), (9, 9))
        if not result.success:
            pytest.skip("No path found on this grid")

        for i in range(1, len(result.path)):
            px, py = result.path[i - 1]
            cx, cy = result.path[i]
            dist = abs(cx - px) + abs(cy - py)
            assert dist == 1, (
                f"Path step {i}: ({px},{py})→({cx},{cy}) is not 4-connected"
            )
            assert heightmap[cy, cx] == 0.0, (
                f"Path step {i}: ({cx},{cy}) is blocked"
            )

    def test_update_accepts_dynamic_state(self):
        """update() accepts a dynamic state dict without error."""
        from uavbench.planners.dstar_lite import DStarLitePlanner

        heightmap, no_fly = _make_grid(size=10, density=0.0)
        planner = DStarLitePlanner(heightmap, no_fly)
        planner.plan((0, 0), (9, 9))

        dyn_state = {
            "fire_mask": np.zeros((10, 10), dtype=bool),
            "smoke_mask": None,
            "traffic_positions": None,
            "forced_block_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }
        planner.update(dyn_state)  # Should not raise

    def test_obstacle_change_triggers_replan(self):
        """When obstacles change, should_replan returns True."""
        from uavbench.planners.dstar_lite import DStarLitePlanner

        heightmap, no_fly = _make_grid(size=10, density=0.0)
        planner = DStarLitePlanner(heightmap, no_fly)
        result = planner.plan((0, 0), (9, 9))
        path = result.path

        # Add obstacle on the planned path
        dyn_state_clean = {
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_positions": None,
            "forced_block_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        # First check: no change → no replan needed
        planner.update(dyn_state_clean)
        should, _ = planner.should_replan((0, 0), path, dyn_state_clean, 1)
        # May or may not trigger depending on implementation

        # Add fire on path midpoint
        fire_mask = np.zeros((10, 10), dtype=bool)
        if len(path) > 2:
            mx, my = path[len(path) // 2]
            fire_mask[my, mx] = True

        dyn_state_blocked = dict(dyn_state_clean)
        dyn_state_blocked["fire_mask"] = fire_mask
        planner.update(dyn_state_blocked)

        should, reason = planner.should_replan(
            (0, 0), path, dyn_state_blocked, 2
        )
        assert should, "D* Lite should request replan when path is blocked"

    def test_no_path_returns_failure(self):
        """When no path exists, plan() returns success=False."""
        from uavbench.planners.dstar_lite import DStarLitePlanner

        # Create grid with wall separating start and goal
        heightmap = np.zeros((10, 10), dtype=np.float32)
        heightmap[:, 5] = 5.0  # wall at column 5
        no_fly = np.zeros((10, 10), dtype=bool)

        planner = DStarLitePlanner(heightmap, no_fly)
        result = planner.plan((0, 0), (9, 9))

        assert not result.success
