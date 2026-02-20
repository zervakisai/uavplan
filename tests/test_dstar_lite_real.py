"""Tests for DStarLiteRealPlanner — true incremental D* Lite.

Validates:
  1. Basic pathfinding correctness on open and walled grids.
  2. Incremental update: after obstacle insertion, replan finds detour.
  3. Obstacle removal: path improves after obstacle is cleared.
  4. No-path detection (goal surrounded).
  5. BasePlanner API compliance (plan returns PlanResult).
  6. Incremental efficiency: fewer expansions than full re-init.
  7. should_replan() / replan() harness-compatible API.
  8. Determinism: same inputs → same path.
  9. Edge cases: start==goal, adjacent start/goal.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.planners.dstar_lite_real import DStarLiteRealPlanner
from uavbench.planners.base import PlanResult, PlannerConfig


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def open_grid():
    """20×20 completely open grid."""
    h = np.zeros((20, 20), dtype=float)
    nfz = np.zeros((20, 20), dtype=bool)
    return h, nfz


@pytest.fixture
def walled_grid():
    """20×20 grid with a wall at row 10 with a gap at col 10."""
    h = np.zeros((20, 20), dtype=float)
    nfz = np.zeros((20, 20), dtype=bool)
    h[10, 0:20] = 1.0  # full wall
    h[10, 10] = 0.0     # gap at (10, 10)
    return h, nfz


# ── 1. Basic pathfinding ─────────────────────────────────────────

class TestBasicPathfinding:
    def test_open_grid_success(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))

        assert isinstance(result, PlanResult)
        assert result.success is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (19, 19)
        # Optimal Manhattan path length = 38 steps + start = 39 cells
        assert len(result.path) == 39

    def test_walled_grid_finds_gap(self, walled_grid):
        h, nfz = walled_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))

        assert result.success is True
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (19, 19)
        # Path must pass through the gap at (10, 10)
        assert (10, 10) in result.path

    def test_blocked_goal_returns_failure(self, open_grid):
        h, nfz = open_grid
        # Surround (15, 15) with obstacles
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            h[15 + dy, 15 + dx] = 1.0
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (15, 15))
        assert result.success is False
        assert result.path == [] or result.path[-1] != (15, 15)

    def test_nfz_respected(self, open_grid):
        h, nfz = open_grid
        # Block column 5 entirely with nfz
        nfz[:, 5] = True
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        # Can't cross col 5, so no path on 4-connected grid
        # Actually start is at x=0 which is left of the wall, goal at x=19 right of it
        assert result.success is False


# ── 2. Start == Goal edge case ────────────────────────────────────

class TestEdgeCases:
    def test_start_equals_goal(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((5, 5), (5, 5))
        assert result.success is True
        assert result.path == [(5, 5)]

    def test_adjacent_start_goal(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((5, 5), (6, 5))
        assert result.success is True
        assert result.path == [(5, 5), (6, 5)]

    def test_out_of_bounds_raises(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        with pytest.raises(ValueError):
            planner.plan((-1, 0), (5, 5))
        with pytest.raises(ValueError):
            planner.plan((5, 5), (20, 20))


# ── 3. Incremental obstacle insertion ────────────────────────────

class TestIncrementalUpdates:
    def test_obstacle_insertion_causes_detour(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)

        # Initial plan: straight path
        result1 = planner.plan((0, 0), (19, 0))
        assert result1.success
        # All cells on row 0: (0,0), (1,0), ..., (19,0)
        assert all(p[1] == 0 for p in result1.path)

        # Insert obstacle at (10, 0)
        dyn = np.zeros((20, 20), dtype=bool)
        dyn[0, 10] = True  # row=0, col=10 → cell (10, 0) blocked
        planner.update_obstacles(dyn)
        result2 = planner.replan_incremental((0, 0))

        assert result2.success
        assert result2.path[0] == (0, 0)
        assert result2.path[-1] == (19, 0)
        # Must detour around (10, 0)
        assert (10, 0) not in result2.path

    def test_obstacle_removal_shortens_path(self, walled_grid):
        h, nfz = walled_grid
        planner = DStarLiteRealPlanner(h, nfz)

        result1 = planner.plan((0, 0), (19, 19))
        assert result1.success
        path_len_1 = len(result1.path)

        # Remove wall: clear all obstacles from heightmap via dynamic mask
        # Actually we simulate removal by giving a new dynamic mask that
        # doesn't add anything. The wall is static heightmap so we can't
        # remove it via dynamic. Instead, let's test obstacle INSERTION first
        # then REMOVAL of the dynamic obstacle.
        dyn = np.zeros((20, 20), dtype=bool)
        dyn[5, 5] = True  # add a dynamic obstacle
        planner.update_obstacles(dyn)
        result_with_obs = planner.replan_incremental((0, 0))

        # Now remove it
        dyn2 = np.zeros((20, 20), dtype=bool)
        planner.update_obstacles(dyn2)
        result_cleared = planner.replan_incremental((0, 0))

        assert result_cleared.success
        # After clearing, path should be at most as long as with obstacle
        assert len(result_cleared.path) <= len(result_with_obs.path) or not result_with_obs.success

    def test_update_returns_changed_count(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        planner.plan((0, 0), (19, 19))

        dyn = np.zeros((20, 20), dtype=bool)
        dyn[5, 5] = True
        dyn[6, 6] = True
        changed = planner.update_obstacles(dyn)
        assert changed == 2  # two cells changed

    def test_no_change_returns_zero(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        planner.plan((0, 0), (19, 19))

        dyn = np.zeros((20, 20), dtype=bool)
        changed = planner.update_obstacles(dyn)
        assert changed == 0


# ── 4. Incremental efficiency ────────────────────────────────────

class TestIncrementalEfficiency:
    def test_incremental_fewer_expansions_than_full(self, open_grid):
        """After a small obstacle change, incremental replan should expand
        fewer vertices than a full re-initialization."""
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)

        # Full initial plan
        result_full = planner.plan((0, 0), (19, 19))
        assert result_full.success
        full_expansions = result_full.expansions

        # Small obstacle change: block one cell off-path
        dyn = np.zeros((20, 20), dtype=bool)
        dyn[5, 15] = True  # far from optimal path
        planner.update_obstacles(dyn)
        result_incr = planner.replan_incremental((0, 0))

        assert result_incr.success
        # Incremental should expand much fewer vertices than full init
        assert result_incr.expansions < full_expansions

    def test_on_path_obstacle_still_succeeds(self, open_grid):
        """Blocking a cell on the current path forces incremental recomputation
        but should still find a detour."""
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success

        # Block a cell that's on the current path
        mid = result.path[len(result.path) // 2]
        dyn = np.zeros((20, 20), dtype=bool)
        dyn[mid[1], mid[0]] = True
        planner.update_obstacles(dyn)

        result2 = planner.replan_incremental((0, 0))
        assert result2.success
        assert mid not in result2.path


# ── 5. Harness-compatible API ────────────────────────────────────

class TestHarnessAPI:
    def test_should_replan_detects_blocked_path(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success

        # Create fire mask blocking a cell on the path
        mid = result.path[5]
        fire = np.zeros((20, 20), dtype=bool)
        fire[mid[1], mid[0]] = True

        should, reason = planner.should_replan(
            current_pos=result.path[0],
            fire_mask=fire,
        )
        assert should is True
        assert "obstacle" in reason.lower()

    def test_should_replan_false_when_clear(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        planner.plan((0, 0), (19, 19))

        should, reason = planner.should_replan(
            current_pos=(0, 0),
            fire_mask=np.zeros((20, 20), dtype=bool),
        )
        assert should is False

    def test_replan_returns_valid_path(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success

        # Block midpoint
        mid = result.path[10]
        fire = np.zeros((20, 20), dtype=bool)
        fire[mid[1], mid[0]] = True

        new_path = planner.replan(
            current_pos=(0, 0),
            goal=(19, 19),
            fire_mask=fire,
        )
        assert len(new_path) > 0
        assert new_path[0] == (0, 0)
        assert new_path[-1] == (19, 19)
        assert mid not in new_path

    def test_get_replan_metrics(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        planner.plan((0, 0), (19, 19))

        dyn = np.zeros((20, 20), dtype=bool)
        dyn[5, 5] = True
        planner.update_obstacles(dyn)
        planner.replan_incremental((0, 0))

        metrics = planner.get_replan_metrics()
        assert "total_replans" in metrics
        assert "total_expansions" in metrics
        assert "incremental_updates" in metrics
        assert metrics["total_replans"] == 1


# ── 6. Determinism ───────────────────────────────────────────────

class TestDeterminism:
    def test_same_inputs_same_path(self, open_grid):
        h, nfz = open_grid
        paths = []
        for _ in range(3):
            planner = DStarLiteRealPlanner(h, nfz)
            result = planner.plan((0, 0), (19, 19))
            paths.append(result.path)
        assert paths[0] == paths[1] == paths[2]

    def test_incremental_determinism(self, open_grid):
        h, nfz = open_grid
        paths = []
        for _ in range(3):
            planner = DStarLiteRealPlanner(h, nfz)
            planner.plan((0, 0), (19, 19))
            dyn = np.zeros((20, 20), dtype=bool)
            dyn[5, 5] = True
            planner.update_obstacles(dyn)
            result = planner.replan_incremental((0, 0))
            paths.append(result.path)
        assert paths[0] == paths[1] == paths[2]


# ── 7. BasePlanner API compliance ────────────────────────────────

class TestAPICompliance:
    def test_subclasses_base_planner(self):
        from uavbench.planners.base import BasePlanner
        assert issubclass(DStarLiteRealPlanner, BasePlanner)

    def test_plan_result_fields(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert hasattr(result, "path")
        assert hasattr(result, "success")
        assert hasattr(result, "compute_time_ms")
        assert hasattr(result, "expansions")
        assert hasattr(result, "replans")
        assert hasattr(result, "reason")
        assert result.compute_time_ms >= 0

    def test_registered_in_planners(self):
        from uavbench.planners import PLANNERS
        assert "incremental_dstar_lite" in PLANNERS
        assert PLANNERS["incremental_dstar_lite"] is DStarLiteRealPlanner

    def test_current_path_property(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert planner.current_path == result.path

    def test_docstring_cites_koenig(self):
        doc = DStarLiteRealPlanner.__doc__
        assert doc is not None
        assert "koenig" in doc.lower() or "d* lite" in doc.lower()


# ── 8. Path validity ─────────────────────────────────────────────

class TestPathValidity:
    def test_path_is_connected(self, open_grid):
        """Every consecutive pair in the path must be 4-connected neighbors."""
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success
        for i in range(len(result.path) - 1):
            ax, ay = result.path[i]
            bx, by = result.path[i + 1]
            assert abs(ax - bx) + abs(ay - by) == 1, (
                f"Disconnected: {result.path[i]} → {result.path[i+1]}"
            )

    def test_path_avoids_obstacles(self, walled_grid):
        h, nfz = walled_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success
        for (x, y) in result.path:
            assert h[y, x] == 0, f"Path crosses obstacle at ({x},{y})"
            assert not nfz[y, x], f"Path crosses NFZ at ({x},{y})"

    def test_incremental_path_avoids_dynamic(self, open_grid):
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        planner.plan((0, 0), (19, 19))

        dyn = np.zeros((20, 20), dtype=bool)
        dyn[10, 10] = True
        planner.update_obstacles(dyn)
        result = planner.replan_incremental((0, 0))
        assert result.success
        assert (10, 10) not in result.path


# ── 9. Multiple sequential replans ───────────────────────────────

class TestMultipleReplans:
    def test_three_sequential_obstacles(self, open_grid):
        """Simulate 3 sequential obstacle insertions with incremental replans."""
        h, nfz = open_grid
        planner = DStarLiteRealPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success

        obstacle_cells = [(5, 5), (10, 10), (15, 15)]
        for i, (ox, oy) in enumerate(obstacle_cells):
            dyn = np.zeros((20, 20), dtype=bool)
            for cx, cy in obstacle_cells[:i + 1]:
                dyn[cy, cx] = True
            planner.update_obstacles(dyn)
            result = planner.replan_incremental((0, 0))
            assert result.success, f"Failed after obstacle #{i+1}"
            for cx, cy in obstacle_cells[:i + 1]:
                assert (cx, cy) not in result.path

        metrics = planner.get_replan_metrics()
        assert metrics["total_replans"] == 3
