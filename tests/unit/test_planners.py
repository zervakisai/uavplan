"""Unit tests: Planner registry, naming honesty, basic pathfinding.

Merges & deduplicates:
  - test_sanity.TestPlanners (3 tests)
  - test_planner_suite_expanded (4 tests)
  - test_new_planners_api (2 tests)
  - test_planner_naming_honesty (56 tests)  ← kept as-is, no overlap
  - test_vv_contracts.TestPlausibility (partial)

Total: ~45 unique tests.  Runtime: < 1 s.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from uavbench.planners import (
    AStarPlanner,
    ThetaStarPlanner,
    PLANNERS,
    PAPER_PLANNERS,
    DEPRECATED_ALIASES,
    DEPRECATED_PLANNERS,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def open_grid_20():
    h = np.zeros((20, 20), dtype=np.float32)
    nfz = np.zeros((20, 20), dtype=bool)
    return h, nfz


@pytest.fixture
def walled_grid_20():
    h = np.zeros((20, 20), dtype=np.float32)
    nfz = np.zeros((20, 20), dtype=bool)
    h[10, 2:18] = 1.0
    h[10, 10] = 0.0  # gap
    return h, nfz


# ── Registry ──────────────────────────────────────────────────

class TestPlannerRegistry:
    """Planner registry is complete and consistent."""

    def test_paper_suite_has_six_planners(self):
        assert len(PAPER_PLANNERS) == 6

    def test_paper_suite_expected_keys(self):
        expected = {
            "astar", "theta_star", "periodic_replan",
            "aggressive_replan", "incremental_dstar_lite", "grid_mppi",
        }
        assert set(PAPER_PLANNERS) == expected

    def test_registry_has_paper_plus_deprecated(self):
        expected = set(PAPER_PLANNERS) | DEPRECATED_PLANNERS | set(DEPRECATED_ALIASES)
        assert set(PLANNERS.keys()) == expected

    def test_paper_suite_excludes_greedy_local(self):
        assert "greedy_local" not in PAPER_PLANNERS

    def test_paper_suite_excludes_legacy_aliases(self):
        for alias in DEPRECATED_ALIASES:
            assert alias not in PAPER_PLANNERS

    @pytest.mark.parametrize("key", list(PAPER_PLANNERS))
    def test_paper_planner_in_registry(self, key):
        assert key in PLANNERS


# ── Deprecation Warnings ─────────────────────────────────────

class TestDeprecationWarnings:
    """Legacy aliases emit DeprecationWarning; paper keys don't."""

    @pytest.mark.parametrize("key", sorted(DEPRECATED_ALIASES))
    def test_legacy_alias_warns(self, key):
        with pytest.warns(DeprecationWarning):
            _ = PLANNERS[key]

    @pytest.mark.parametrize("key", sorted(DEPRECATED_ALIASES))
    def test_legacy_alias_suggests_replacement(self, key):
        with pytest.warns(DeprecationWarning, match=DEPRECATED_ALIASES[key]):
            _ = PLANNERS[key]

    @pytest.mark.parametrize("key", sorted(DEPRECATED_PLANNERS))
    def test_deprecated_planner_warns(self, key):
        with pytest.warns(DeprecationWarning):
            _ = PLANNERS[key]

    @pytest.mark.parametrize("key", list(PAPER_PLANNERS))
    def test_paper_planner_no_warning(self, key):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _ = PLANNERS[key]


# ── Docstring Honesty ─────────────────────────────────────────

_HONEST_NAMES = {
    "periodic_replan": ("periodic", "NOT", "D\\* Lite"),
    "aggressive_replan": ("aggressive", "NOT", "AD\\*"),
    "grid_mppi": ("MPPI", "discretized", "4 cardinal"),
}


class TestDocstringHonesty:
    """Planner docstrings disclose simplifications."""

    @pytest.mark.parametrize("key", list(PAPER_PLANNERS))
    def test_paper_planner_has_docstring(self, key):
        cls = PLANNERS[key]
        assert cls.__doc__, f"{key} has no docstring"

    @pytest.mark.parametrize("name,keywords", [
        (n, kw) for n, kw in _HONEST_NAMES.items()
    ])
    def test_docstring_discloses_simplification(self, name, keywords):
        doc = PLANNERS[name].__doc__ or ""
        for kw in keywords:
            assert kw.lower().replace("\\", "") in doc.lower() or kw in doc, (
                f"{name} docstring missing keyword '{kw}'"
            )


# ── Basic Pathfinding ─────────────────────────────────────────

class TestBasicPathfinding:
    """All planners solve trivial and walled grids."""

    @pytest.mark.parametrize("pid", list(PAPER_PLANNERS))
    def test_all_paper_planners_solve_open_grid(self, pid, open_grid_20):
        h, nfz = open_grid_20
        planner = PLANNERS[pid](h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success, f"{pid} failed on open grid"
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (19, 19)

    @pytest.mark.parametrize("pid", list(PAPER_PLANNERS))
    def test_all_paper_planners_solve_walled_grid(self, pid, walled_grid_20):
        h, nfz = walled_grid_20
        planner = PLANNERS[pid](h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success, f"{pid} failed walled grid"

    def test_all_paper_planners_return_planresult(self, open_grid_20):
        h, nfz = open_grid_20
        for pid in PAPER_PLANNERS:
            planner = PLANNERS[pid](h, nfz)
            result = planner.plan((0, 0), (19, 19))
            assert hasattr(result, "success")
            assert hasattr(result, "path")
            assert isinstance(result.path, list)

    def test_deprecated_planners_still_solve(self, open_grid_20):
        h, nfz = open_grid_20
        for pid in sorted(set(DEPRECATED_ALIASES) | DEPRECATED_PLANNERS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                planner = PLANNERS[pid](h, nfz)
            result = planner.plan((0, 0), (19, 19))
            assert result.success, f"deprecated planner {pid} failed"

    def test_astar_optimal_on_empty_grid(self):
        """A* on a clear grid returns Manhattan-length + 1 nodes."""
        h = np.zeros((20, 20), dtype=np.float32)
        nfz = np.zeros((20, 20), dtype=bool)
        planner = AStarPlanner(h, nfz)
        result = planner.plan((0, 0), (19, 19))
        assert result.success
        expected = abs(19 - 0) + abs(19 - 0) + 1  # Manhattan + 1
        assert len(result.path) == expected

    def test_theta_star_shorter_or_equal_to_astar(self):
        """Theta* produces path ≤ A* in waypoint count."""
        h = np.zeros((20, 20), dtype=np.float32)
        h[10, 5:15] = 1
        nfz = np.zeros((20, 20), dtype=bool)
        a = AStarPlanner(h, nfz).plan((0, 5), (19, 15))
        t = ThetaStarPlanner(h, nfz).plan((0, 5), (19, 15))
        assert a.success and t.success
        assert len(t.path) <= len(a.path)

    def test_planner_timeout(self):
        from uavbench.planners import AStarConfig
        h = np.zeros((100, 100), dtype=np.float32)
        nfz = np.zeros((100, 100), dtype=bool)
        cfg = AStarConfig(max_planning_time_ms=0.1)
        planner = AStarPlanner(h, nfz, cfg)
        result = planner.plan((0, 0), (99, 99))
        assert result.compute_time_ms <= 10.0


# ── AdaptiveAStarPlanner: _path_idx progress tracking (V&V Fix) ──

class TestAdaptiveAStarPathIdx:
    """Verify that should_replan() does not trigger on already-passed cells."""

    def _make_planner(self):
        from uavbench.planners.adaptive_astar import AdaptiveAStarPlanner
        h = np.zeros((20, 20), dtype=np.float32)
        nfz = np.zeros((20, 20), dtype=bool)
        return AdaptiveAStarPlanner(h, nfz)

    def test_path_idx_initialised_zero(self):
        p = self._make_planner()
        p.plan((0, 0), (10, 0))
        assert p._path_idx == 0

    def test_no_replan_on_obstacle_behind_uav(self):
        """Fire behind the UAV must NOT trigger path_blocked."""
        p = self._make_planner()
        p.plan((0, 0), (15, 0))

        # Simulate UAV having advanced to cell (10, 0)
        p._path_idx = 10

        # Place fire at (5, 0) — behind the UAV
        fire = np.zeros((20, 20), dtype=bool)
        fire[0, 5] = True  # fire_mask[y, x] = fire_mask[row, col]

        should, reason = p.should_replan(
            current_pos=(10, 0, 1),
            fire_mask=fire,
        )
        assert not (should and reason == "path_blocked"), (
            "path_blocked triggered by obstacle BEHIND the UAV — path_idx fix missing"
        )

    def test_replan_on_obstacle_ahead(self):
        """Fire ahead of the UAV MUST trigger path_blocked."""
        from uavbench.planners.adaptive_astar import AdaptiveAStarConfig
        h = np.zeros((30, 30), dtype=np.float32)
        nfz = np.zeros((30, 30), dtype=bool)
        cfg = AdaptiveAStarConfig(base_interval=1000, lookahead_steps=10)
        from uavbench.planners.adaptive_astar import AdaptiveAStarPlanner
        p = AdaptiveAStarPlanner(h, nfz, cfg)
        p.plan((0, 0), (20, 0))

        # UAV at start (0,0); fire immediately ahead at (3, 0)
        fire = np.zeros((30, 30), dtype=bool)
        fire[0, 3] = True  # y=0, x=3

        should, reason = p.should_replan(current_pos=(0, 0, 1), fire_mask=fire)
        assert should and reason == "path_blocked"

    def test_path_idx_reset_after_replan(self):
        p = self._make_planner()
        p.plan((0, 0), (10, 0))
        p._path_idx = 5
        p.replan(current_pos=(5, 0, 1), goal=(10, 0))
        assert p._path_idx == 0


# ── DStarLiteRealPlanner: _path_idx progress tracking (V&V Fix) ──

class TestDStarLitePathIdx:
    """Verify DStarLiteRealPlanner's path-prefix fix."""

    def _make_planner(self):
        from uavbench.planners.dstar_lite_real import DStarLiteRealPlanner
        h = np.zeros((20, 20), dtype=np.float32)
        nfz = np.zeros((20, 20), dtype=bool)
        return DStarLiteRealPlanner(h, nfz)

    def test_path_idx_attribute_exists(self):
        p = self._make_planner()
        assert hasattr(p, "_path_idx"), "_path_idx missing from DStarLiteRealPlanner"
        assert p._path_idx == 0

    def test_path_idx_reset_after_replan(self):
        p = self._make_planner()
        p.plan((0, 0), (10, 0))
        p._path_idx = 5
        p.replan(current_pos=(5, 0, 1), goal=(10, 0))
        assert p._path_idx == 0
