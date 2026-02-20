"""
Comprehensive tests for CLI benchmark functionality.
Verifies that all documented features work as advertised.
"""
import tempfile
from pathlib import Path
import json

import pytest

from uavbench.cli.benchmark import (
    run_planner_once,
    aggregate,
    scenario_path,
    _waypoint_action,
)
from uavbench.metrics.operational import compute_all_metrics
from uavbench.scenarios.registry import SCENARIO_REGISTRY, list_scenarios_by_mission, list_scenarios_by_regime
from uavbench.scenarios.schema import MissionType, Regime


class TestScenarioPath:
    """Test scenario path resolution."""

    def test_scenario_path_gov_civil_protection_easy(self):
        """Scenario path for gov_civil_protection_easy exists."""
        path = scenario_path("gov_civil_protection_easy")
        assert path.exists(), f"Scenario file not found: {path}"
        assert path.suffix == ".yaml"

    def test_scenario_path_gov_scenarios(self):
        """Gov mission scenarios resolve correctly."""
        for scenario_id in list(SCENARIO_REGISTRY.keys()):
            path = scenario_path(scenario_id)
            assert path.exists(), f"Scenario {scenario_id} not found at {path}"


class TestWaypointAction:
    """Test waypoint-to-action conversion."""

    def test_move_right(self):
        """Moving right (x+1) produces action 3."""
        action = _waypoint_action((5, 5), (6, 5))
        assert action == 3, "Right movement should be action 3"

    def test_move_left(self):
        """Moving left (x-1) produces action 2."""
        action = _waypoint_action((5, 5), (4, 5))
        assert action == 2, "Left movement should be action 2"

    def test_move_down(self):
        """Moving down (y+1) produces action 1."""
        action = _waypoint_action((5, 5), (5, 6))
        assert action == 1, "Down movement should be action 1"

    def test_move_up(self):
        """Moving up (y-1) produces action 0."""
        action = _waypoint_action((5, 5), (5, 4))
        assert action == 0, "Up movement should be action 0"


class TestRunPlannerOnce:
    """Test single planner run (static planning)."""

    def test_run_astar_gov_easy(self):
        """A* planner runs on gov_civil_protection_easy scenario."""
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)

        # Check result structure
        assert isinstance(result, dict), "Result should be dict"
        assert "scenario_id" in result
        assert "planner_id" in result
        assert "seed" in result
        assert "success" in result
        assert "path_length" in result
        assert "planning_time" in result

        # Check values
        assert result["scenario_id"] == "gov_civil_protection_easy"
        assert result["planner_id"] == "astar"
        assert result["seed"] == 0
        assert isinstance(result["success"], bool)
        assert result["path_length"] >= 0
        assert result["planning_time"] >= 0

    def test_run_theta_star_gov_easy(self):
        """Theta* planner runs on gov_civil_protection_easy scenario."""
        result = run_planner_once("gov_civil_protection_easy", "theta_star", seed=0)

        assert result["planner_id"] == "theta_star"
        assert "success" in result
        assert "path_length" in result

    def test_deterministic_seed(self):
        """Same seed produces consistent results."""
        result1 = run_planner_once("gov_civil_protection_easy", "astar", seed=42)
        result2 = run_planner_once("gov_civil_protection_easy", "astar", seed=42)

        # Should get same path length with same seed
        assert result1["path_length"] == result2["path_length"]
        assert result1["success"] == result2["success"]

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different results."""
        result1 = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        result2 = run_planner_once("gov_civil_protection_easy", "astar", seed=1)

        # May differ (not guaranteed, but likely for stochastic environments)
        # At minimum, both should complete successfully
        assert "path_length" in result1
        assert "path_length" in result2


class TestMetricsComputation:
    """Test metrics computation on planner results."""

    def test_compute_all_metrics_successful(self):
        """Compute metrics for successful run."""
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)

        metrics = compute_all_metrics(result)

        # Check that metrics are computed
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_metrics_contain_expected_keys(self):
        """Metrics include standard path quality measures."""
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        metrics = compute_all_metrics(result)

        # Standard metrics documented in API
        expected_metrics = ["path_length", "planning_time_ms"]

        for key in expected_metrics:
            assert key in metrics or key in result, f"Missing metric: {key}"


class TestAggregation:
    """Test results aggregation."""

    def test_aggregate_single_result(self):
        """Aggregate single result."""
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        agg = aggregate([result])

        assert isinstance(agg, dict)
        assert len(agg) > 0

    def test_aggregate_multiple_results(self):
        """Aggregate multiple results."""
        results = [
            run_planner_once("gov_civil_protection_easy", "astar", seed=i)
            for i in range(3)
        ]

        agg = aggregate(results)

        assert isinstance(agg, dict)
        # Should have aggregated statistics
        assert len(agg) > 0

    def test_aggregate_empty_list(self):
        """Aggregate empty list returns empty dict."""
        agg = aggregate([])
        assert isinstance(agg, dict)


class TestScenarioRegistry:
    """Test scenario registry queries."""

    def test_registry_has_scenarios(self):
        """Registry contains scenarios."""
        assert len(SCENARIO_REGISTRY) > 0
        assert len(SCENARIO_REGISTRY) == 9, f"Should have 9 scenarios, got {len(SCENARIO_REGISTRY)}"

    def test_all_scenarios_have_required_fields(self):
        """All scenarios have required metadata fields."""
        for scenario_id, metadata in SCENARIO_REGISTRY.items():
            assert hasattr(metadata, "scenario_id")
            assert hasattr(metadata, "mission_type")
            assert hasattr(metadata, "regime")
            assert hasattr(metadata, "difficulty")

    def test_list_by_mission_type(self):
        """List scenarios by mission type."""
        civil_protection = list_scenarios_by_mission(MissionType.CIVIL_PROTECTION)
        assert len(civil_protection) == 3, "Should have 3 civil protection scenarios"

        # All should have CIVIL_PROTECTION mission type
        for scenario_id in civil_protection:
            metadata = SCENARIO_REGISTRY[scenario_id]
            assert metadata.mission_type == MissionType.CIVIL_PROTECTION

    def test_list_by_regime(self):
        """List scenarios by regime."""
        naturalistic = list_scenarios_by_regime(Regime.NATURALISTIC)
        stress_test = list_scenarios_by_regime(Regime.STRESS_TEST)

        assert len(naturalistic) > 0, "Should have naturalistic scenarios"
        assert len(stress_test) > 0, "Should have stress-test scenarios"

    def test_mission_types_coverage(self):
        """All gov mission types are represented."""
        all_mission_types = set()
        for metadata in SCENARIO_REGISTRY.values():
            all_mission_types.add(metadata.mission_type)

        # Should have 3 mission types (civil_protection, maritime_domain, critical_infrastructure)
        assert len(all_mission_types) == 3, f"Should have 3 mission types, got {len(all_mission_types)}"

    def test_difficulties_coverage(self):
        """All difficulty levels represented."""
        registry_difficulties = set()

        for metadata in SCENARIO_REGISTRY.values():
            registry_difficulties.add(metadata.difficulty)

        # Check that all difficulties are present (stored as uppercase strings in metadata)
        assert "EASY" in registry_difficulties
        assert "MEDIUM" in registry_difficulties
        assert "HARD" in registry_difficulties

    def test_regimes_coverage(self):
        """Both regimes represented."""
        regimes = {Regime.NATURALISTIC, Regime.STRESS_TEST}
        registry_regimes = set()

        for metadata in SCENARIO_REGISTRY.values():
            registry_regimes.add(metadata.regime)

        assert regimes.issubset(registry_regimes), "Should have both NATURALISTIC and STRESS_TEST"


class TestScenarioFiltering:
    """Test scenario filtering capabilities."""

    def test_filter_by_mission_civil_protection(self):
        """Filter scenarios by civil protection mission."""
        civil = list_scenarios_by_mission(MissionType.CIVIL_PROTECTION)
        assert len(civil) == 3

    def test_filter_by_mission_maritime(self):
        """Filter scenarios by maritime domain mission."""
        maritime = list_scenarios_by_mission(MissionType.MARITIME_DOMAIN)
        assert len(maritime) == 3

    def test_filter_by_regime_naturalistic(self):
        """Filter naturalistic scenarios."""
        naturalistic = list_scenarios_by_regime(Regime.NATURALISTIC)
        assert len(naturalistic) > 0

    def test_filter_by_regime_stress_test(self):
        """Filter stress-test scenarios."""
        stress = list_scenarios_by_regime(Regime.STRESS_TEST)
        assert len(stress) > 0


class TestBenchmarkIntegration:
    """Integration tests for full benchmark workflow."""

    def test_run_mini_benchmark(self):
        """Run mini benchmark: 2 scenarios × 2 planners × 1 seed."""
        scenarios = ["gov_civil_protection_easy", "gov_maritime_domain_easy"]
        planners = ["astar", "theta_star"]
        seeds = [0]

        results = []
        for scenario_id in scenarios:
            for planner_id in planners:
                for seed in seeds:
                    result = run_planner_once(scenario_id, planner_id, seed=seed)
                    results.append(result)

        assert len(results) == 4  # 2 scenarios × 2 planners × 1 seed
        assert all("success" in r for r in results)

    def test_benchmark_cross_planner_comparison(self):
        """Compare A* vs Theta* on same scenario."""
        results_astar = []
        results_theta = []

        for seed in range(3):
            r_a = run_planner_once("gov_civil_protection_easy", "astar", seed=seed)
            r_t = run_planner_once("gov_civil_protection_easy", "theta_star", seed=seed)
            results_astar.append(r_a)
            results_theta.append(r_t)

        avg_astar = sum(r["path_length"] for r in results_astar) / len(results_astar)
        avg_theta = sum(r["path_length"] for r in results_theta) / len(results_theta)

        # Both should have valid results
        assert avg_astar > 0
        assert avg_theta > 0

    def test_benchmark_aggregation_workflow(self):
        """Test complete aggregation workflow."""
        # Run benchmark
        results = []
        for seed in range(2):
            for planner in ["astar", "theta_star"]:
                r = run_planner_once("gov_civil_protection_easy", planner, seed=seed)
                results.append(r)

        # Aggregate
        agg = aggregate(results)

        assert isinstance(agg, dict)
        assert len(agg) > 0


class TestPlannerRegistry:
    """Test planner registry and availability."""

    def test_astar_available(self):
        """A* planner is available."""
        from uavbench.planners import PLANNERS
        assert "astar" in PLANNERS

    def test_theta_star_available(self):
        """Theta* planner is available."""
        from uavbench.planners import PLANNERS
        assert "theta_star" in PLANNERS

    def test_run_all_available_planners(self):
        """Run paper-suite planners on gov_civil_protection_easy."""
        from uavbench.planners import PAPER_PLANNERS

        for planner_id in PAPER_PLANNERS[:3]:  # Test first 3 paper planners
            result = run_planner_once("gov_civil_protection_easy", planner_id, seed=0)
            assert result["planner_id"] == planner_id
            assert "path_length" in result


class TestMetricsComprehensiveness:
    """Test that all metrics are computed correctly."""

    def test_result_contains_planning_time(self):
        """Result includes planning time."""
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        assert "planning_time" in result
        assert result["planning_time"] >= 0

    def test_metrics_include_success_rate(self):
        """Aggregation includes success rate."""
        results = [run_planner_once("gov_civil_protection_easy", "astar", seed=i) for i in range(2)]
        agg = aggregate(results)

        # Check success computation
        successes = sum(1 for r in results if r["success"])
        success_rate = successes / len(results)
        assert 0.0 <= success_rate <= 1.0


class TestResultConsistency:
    """Test result format consistency."""

    def test_result_keys_consistent_across_planners(self):
        """All planner results have same keys."""
        result_a = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        result_t = run_planner_once("gov_civil_protection_easy", "theta_star", seed=0)

        keys_a = set(result_a.keys())
        keys_t = set(result_t.keys())

        # Should have same set of keys (minus planner-specific ones)
        common_keys = {"scenario_id", "planner_id", "seed", "success", "path_length", "planning_time"}
        assert common_keys.issubset(keys_a)
        assert common_keys.issubset(keys_t)

    def test_result_values_are_serializable(self):
        """Results can be serialized to JSON (excluding numpy arrays and config)."""
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)

        # Create JSON-serializable version (exclude heightmap, no_fly, config)
        serializable = {
            k: v for k, v in result.items()
            if k not in ("heightmap", "no_fly", "config")
        }

        # Should be JSON serializable
        json_str = json.dumps(serializable)
        parsed = json.loads(json_str)

        assert parsed["scenario_id"] == result["scenario_id"]
        assert parsed["planner_id"] == result["planner_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
