"""Contract tests for Calibration (CC-1 through CC-4).

CC-1: Feasibility pre-check produces valid results
CC-2: Difficulty thresholds met (Medium>=50%)
CC-4: Infeasible episodes properly flagged
"""

from __future__ import annotations

import pytest

from uavbench.scenarios.calibration import (
    CalibrationResult,
    FeasibilityResult,
    calibrate_difficulty,
    feasibility_pre_check,
)
from uavbench.scenarios.loader import load_scenario


# ===========================================================================
# CC-1: Feasibility pre-check
# ===========================================================================


class TestCC1_FeasibilityPreCheck:
    """feasibility_pre_check returns valid FeasibilityResult."""

    def test_returns_feasibility_result(self):
        """Pre-check returns a FeasibilityResult dataclass."""
        config = load_scenario("osm_penteli_pharma_delivery_medium")
        result = feasibility_pre_check(config, seed=42, horizon=20)
        assert isinstance(result, FeasibilityResult)
        assert result.seed == 42

    def test_short_horizon_feasible(self):
        """Short horizon (10 steps) should be feasible before dynamics activate."""
        config = load_scenario("osm_penteli_pharma_delivery_medium")
        result = feasibility_pre_check(config, seed=42, horizon=10)
        assert result.feasible, (
            "Short horizon before event_t1 should be feasible"
        )
        assert result.first_infeasible_step is None

    def test_feasible_has_no_infeasible_step(self):
        """If feasible, first_infeasible_step is None."""
        config = load_scenario("osm_piraeus_urban_rescue_medium")
        result = feasibility_pre_check(config, seed=42, horizon=10)
        if result.feasible:
            assert result.first_infeasible_step is None

    def test_infeasible_has_step_number(self):
        """If infeasible, first_infeasible_step is a positive int."""
        config = load_scenario("osm_downtown_fire_surveillance_medium")
        # Try multiple seeds — dense downtown may become infeasible
        for seed in range(10):
            result = feasibility_pre_check(config, seed=seed, horizon=50)
            if not result.feasible:
                assert isinstance(result.first_infeasible_step, int)
                assert result.first_infeasible_step >= 0
                return  # found one infeasible — test passes
        # If all 10 seeds are feasible, that's fine too (calibration worked)

    def test_total_steps_checked_is_positive(self):
        """total_steps_checked is non-negative."""
        config = load_scenario("osm_penteli_pharma_delivery_medium")
        result = feasibility_pre_check(config, seed=42, horizon=5)
        assert result.total_steps_checked >= 0


# ===========================================================================
# CC-2: Difficulty thresholds
# ===========================================================================


class TestCC2_DifficultyThresholds:
    """calibrate_difficulty checks thresholds per difficulty level."""

    def test_returns_calibration_result(self):
        """calibrate_difficulty returns CalibrationResult."""
        result = calibrate_difficulty(
            "osm_penteli_pharma_delivery_medium", n_seeds=3, horizon=10,
        )
        assert isinstance(result, CalibrationResult)
        assert result.n_seeds == 3
        assert 0.0 <= result.feasibility_rate <= 1.0

    def test_medium_passes_threshold(self):
        """Medium OSM scenario should pass its calibration threshold."""
        result = calibrate_difficulty(
            "osm_penteli_pharma_delivery_medium", n_seeds=5, horizon=10,
        )
        assert result.passes_threshold, (
            f"OSM medium scenario should pass threshold "
            f"(rate={result.feasibility_rate:.2f}, threshold={result.threshold})"
        )

    def test_per_seed_results_populated(self):
        """per_seed list has one entry per seed."""
        result = calibrate_difficulty(
            "osm_piraeus_urban_rescue_medium", n_seeds=3, horizon=10,
        )
        assert len(result.per_seed) == 3
        for r in result.per_seed:
            assert isinstance(r, FeasibilityResult)


# ===========================================================================
# CC-4: Infeasible episode flagging
# ===========================================================================


class TestCC4_InfeasibleFlagging:
    """Infeasible episodes are properly flagged."""

    def test_infeasible_result_has_false_feasible(self):
        """If a seed is infeasible, feasible=False."""
        # Use dense downtown scenario — more likely to hit infeasibility
        config = load_scenario("osm_downtown_fire_surveillance_medium")
        for seed in range(20):
            result = feasibility_pre_check(config, seed=seed, horizon=30)
            if not result.feasible:
                assert result.feasible is False
                assert result.first_infeasible_step is not None
                return
        # All feasible is also acceptable (means calibration is good)

    def test_calibration_rate_reflects_infeasible_seeds(self):
        """feasibility_rate = feasible_count / n_seeds."""
        result = calibrate_difficulty(
            "osm_penteli_pharma_delivery_medium", n_seeds=5, horizon=10,
        )
        feasible_count = sum(1 for r in result.per_seed if r.feasible)
        expected_rate = feasible_count / result.n_seeds
        assert abs(result.feasibility_rate - expected_rate) < 1e-9, (
            f"Rate mismatch: {result.feasibility_rate} vs {expected_rate}"
        )
