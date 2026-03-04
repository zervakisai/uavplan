"""Contract tests for Sanity Check (SC-1 through SC-4).

Uses synthetic results to verify that the sanity checker correctly
detects violations and passes clean results.
6 planners: astar, theta_star (static), periodic_replan, aggressive_replan, dstar_lite, apf.
"""

from __future__ import annotations

import pytest

from uavbench.benchmark.sanity_check import (
    SanityReport,
    SanityViolation,
    Severity,
    ViolationType,
    run_sanity_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(
    scenario_id: str,
    planner_rates: dict[str, float],
    n_seeds: int = 10,
) -> list[dict]:
    """Generate synthetic episode results with given success rates."""
    results = []
    for planner_id, rate in planner_rates.items():
        n_success = int(round(rate * n_seeds))
        for i in range(n_seeds):
            results.append({
                "scenario_id": scenario_id,
                "planner_id": planner_id,
                "success": i < n_success,
            })
    return results


# ===========================================================================
# SC-1: Adaptive > Static in fire scenarios
# ===========================================================================


class TestSC1_AdaptiveBeatsStatic:
    """Static planner beating all adaptive in fire = VIOLATION."""

    def test_static_beats_adaptive_in_fire_detected(self):
        """SC-1: A* 50% beats all adaptive in fire → violation."""
        results = _make_results(
            "gov_fire_delivery_medium",
            {
                "astar": 0.5,
                "periodic_replan": 0.1,
                "aggressive_replan": 0.1,
                "dstar_lite": 0.1,
            },
        )
        report = run_sanity_check(results)
        sc1 = [v for v in report.violations
               if v.violation_type == ViolationType.ADAPTIVE_BEHIND_STATIC]
        assert len(sc1) >= 1, "Should detect static > adaptive in fire"
        assert sc1[0].severity == Severity.ERROR

    def test_adaptive_beats_static_in_fire_passes(self):
        """SC-1: Adaptive > static → no violation."""
        results = _make_results(
            "gov_fire_delivery_medium",
            {
                "astar": 0.0,
                "periodic_replan": 0.5,
                "aggressive_replan": 0.5,
                "dstar_lite": 0.3,
            },
        )
        report = run_sanity_check(results)
        sc1 = [v for v in report.violations
               if v.violation_type == ViolationType.ADAPTIVE_BEHIND_STATIC]
        assert len(sc1) == 0, "No violation when adaptive > static"

    def test_non_fire_scenario_not_checked(self):
        """SC-1: flood scenario with static > adaptive → no SC-1."""
        results = _make_results(
            "gov_flood_rescue_medium",
            {
                "astar": 0.5,
                "periodic_replan": 0.1,
                "aggressive_replan": 0.1,
                "dstar_lite": 0.1,
            },
        )
        report = run_sanity_check(results)
        sc1 = [v for v in report.violations
               if v.violation_type == ViolationType.ADAPTIVE_BEHIND_STATIC]
        assert len(sc1) == 0, "SC-1 only applies to fire scenarios"


# ===========================================================================
# SC-2: Difficulty ordering
# ===========================================================================


class TestSC2_DifficultyOrdering:
    """success(medium) >= success(hard) for each planner."""

    def test_hard_better_than_medium_detected(self):
        """SC-2: Hard > medium for a planner → violation."""
        results = (
            _make_results("gov_fire_delivery_medium", {"astar": 0.2})
            + _make_results("gov_fire_delivery_hard", {"astar": 0.8})
        )
        report = run_sanity_check(results)
        sc2 = [v for v in report.violations
               if v.violation_type == ViolationType.DIFFICULTY_ORDERING]
        assert len(sc2) >= 1, "Should detect hard > medium"

    def test_medium_better_than_hard_passes(self):
        """SC-2: Medium > hard → no violation."""
        results = (
            _make_results("gov_fire_delivery_medium", {"astar": 0.8})
            + _make_results("gov_fire_delivery_hard", {"astar": 0.2})
        )
        report = run_sanity_check(results)
        sc2 = [v for v in report.violations
               if v.violation_type == ViolationType.DIFFICULTY_ORDERING]
        assert len(sc2) == 0

    def test_similar_rates_tolerated(self):
        """SC-2: Hard slightly > medium within 5% tolerance → no violation."""
        results = (
            _make_results("gov_fire_delivery_medium", {"astar": 0.5})
            + _make_results("gov_fire_delivery_hard", {"astar": 0.5})
        )
        report = run_sanity_check(results)
        sc2 = [v for v in report.violations
               if v.violation_type == ViolationType.DIFFICULTY_ORDERING]
        assert len(sc2) == 0, "Equal rates should pass within tolerance"


# ===========================================================================
# SC-4: D*Lite >= A*
# ===========================================================================


class TestSC4_DStarLitePosition:
    """D*Lite should perform >= A* everywhere."""

    def test_dstar_behind_astar_detected(self):
        """SC-4: D*Lite < A* → violation (possible implementation bug)."""
        results = _make_results(
            "gov_fire_delivery_medium",
            {"astar": 0.5, "dstar_lite": 0.1},
        )
        report = run_sanity_check(results)
        sc4 = [v for v in report.violations
               if v.violation_type == ViolationType.DSTAR_BEHIND_ASTAR]
        assert len(sc4) >= 1, "Should detect D*Lite < A*"

    def test_dstar_beats_astar_passes(self):
        """SC-4: D*Lite >= A* → no violation."""
        results = _make_results(
            "gov_fire_delivery_medium",
            {"astar": 0.1, "dstar_lite": 0.3},
        )
        report = run_sanity_check(results)
        sc4 = [v for v in report.violations
               if v.violation_type == ViolationType.DSTAR_BEHIND_ASTAR]
        assert len(sc4) == 0

    def test_both_zero_passes(self):
        """SC-4: Both at 0% → no violation (within tolerance)."""
        results = _make_results(
            "gov_fire_delivery_hard",
            {"astar": 0.0, "dstar_lite": 0.0},
        )
        report = run_sanity_check(results)
        sc4 = [v for v in report.violations
               if v.violation_type == ViolationType.DSTAR_BEHIND_ASTAR]
        assert len(sc4) == 0


# ===========================================================================
# Overall report
# ===========================================================================


class TestSanityReport:
    """SanityReport correctly aggregates violations."""

    def test_clean_results_pass(self):
        """Well-behaved results produce a passing report."""
        results = (
            _make_results(
                "gov_fire_delivery_medium",
                {
                    "astar": 0.0,
                    "periodic_replan": 0.7,
                    "aggressive_replan": 0.7,
                    "dstar_lite": 0.3,
                },
            )
            + _make_results(
                "gov_fire_delivery_hard",
                {
                    "astar": 0.0,
                    "periodic_replan": 0.3,
                    "aggressive_replan": 0.3,
                    "dstar_lite": 0.1,
                },
            )
        )
        report = run_sanity_check(results)
        assert report.passed, (
            f"Clean results should pass sanity check. "
            f"Violations: {[v.details for v in report.violations]}"
        )

    def test_report_passed_false_on_error(self):
        """Report.passed is False when there are ERROR-level violations."""
        results = _make_results(
            "gov_fire_delivery_medium",
            {
                "astar": 0.5,
                "periodic_replan": 0.0,
                "aggressive_replan": 0.0,
                "dstar_lite": 0.0,
            },
        )
        report = run_sanity_check(results)
        assert not report.passed, "Should fail with ERROR-level violations"

    def test_empty_results(self):
        """Empty results produce a passing report with no violations."""
        report = run_sanity_check([])
        assert report.passed
        assert len(report.violations) == 0
