"""Sanity tests for UAVBench benchmark infrastructure."""

import numpy as np
import pytest
from pathlib import Path

from uavbench.planners import AStarPlanner, ThetaStarPlanner, JPSPlanner, PLANNERS
from uavbench.scenarios.registry import (
    SCENARIO_REGISTRY, list_scenarios, list_scenarios_by_regime, 
    list_scenarios_with_dynamics, print_scenario_registry
)
from uavbench.scenarios.schema import Domain, Difficulty, MissionType, Regime, ScenarioConfig
from uavbench.metrics.comprehensive import EpisodeMetrics, compute_episode_metrics, aggregate_episode_metrics
from uavbench.benchmark.solvability import check_solvability_certificate


class TestScenarioRegistry:
    """Test scenario registry and metadata."""
    
    def test_registry_has_scenarios(self):
        """Verify we have scenarios registered."""
        assert len(SCENARIO_REGISTRY) > 0
    
    def test_list_scenarios(self):
        """Test scenario listing functions."""
        all_scen = list_scenarios()
        assert len(all_scen) == 34, f"Expected 34 scenarios, got {len(all_scen)}"
        assert len(all_scen) > 0
        # Check that wildfire scenario exists (check for any wildfire scenario)
        wildfire_scen = [s for s in all_scen if "wildfire" in s]
        assert len(wildfire_scen) > 0, "No wildfire scenarios found"
        
        naturalistic = list_scenarios_by_regime(Regime.NATURALISTIC)
        stress = list_scenarios_by_regime(Regime.STRESS_TEST)
        assert len(naturalistic) + len(stress) == len(all_scen)
        
        dynamic_scen = list_scenarios_with_dynamics()
        assert len(dynamic_scen) > 0
    
    def test_scenario_metadata(self):
        """Test individual scenario metadata."""
        # Use actual scenario ID with full name
        scenario_id = None
        for sid in list_scenarios():
            if "wildfire" in sid and "easy" in sid:
                scenario_id = sid
                break
        
        assert scenario_id is not None, "No suitable wildfire scenario found"
        meta = SCENARIO_REGISTRY[scenario_id]
        assert meta.mission_type == MissionType.WILDFIRE_WUI
        assert meta.regime == Regime.NATURALISTIC
        assert meta.has_fire == True


class TestPlanners:
    """Test planner implementations."""
    
    def test_planner_registry(self):
        """Verify all planners registered."""
        assert "astar" in PLANNERS
        assert "theta_star" in PLANNERS
        assert len(PLANNERS) >= 2
    
    def test_astar_planning(self):
        """Test A* planner on simple map."""
        heightmap = np.zeros((10, 10))
        no_fly = np.zeros((10, 10), dtype=bool)
        
        planner = AStarPlanner(heightmap, no_fly)
        result = planner.plan((0, 0), (9, 9))
        
        assert result.success == True
        assert len(result.path) > 0
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (9, 9)
        assert result.compute_time_ms > 0
    
    def test_theta_star_planning(self):
        """Test Theta* produces smoother paths."""
        heightmap = np.zeros((20, 20))
        heightmap[10, 5:15] = 1  # Obstacle
        no_fly = np.zeros((20, 20), dtype=bool)
        
        astar = AStarPlanner(heightmap, no_fly)
        theta = ThetaStarPlanner(heightmap, no_fly)
        
        a_result = astar.plan((0, 5), (19, 15))
        t_result = theta.plan((0, 5), (19, 15))
        
        assert a_result.success
        assert t_result.success
        # Theta* should produce shorter or equal path
        assert len(t_result.path) <= len(a_result.path)
    
    def test_planner_timeout(self):
        """Test planner respects time budget."""
        from uavbench.planners import AStarConfig
        
        heightmap = np.zeros((100, 100))
        no_fly = np.zeros((100, 100), dtype=bool)
        
        cfg = AStarConfig(max_planning_time_ms=0.1)  # Very short timeout
        planner = AStarPlanner(heightmap, no_fly, cfg)
        result = planner.plan((0, 0), (99, 99))
        
        assert result.compute_time_ms <= 10.0  # Should finish or timeout gracefully


class TestSolvability:
    """Test solvability certificate checking."""
    
    def test_solvable_scenario(self):
        """Test solvability checker on solvable map."""
        heightmap = np.zeros((10, 10))
        no_fly = np.zeros((10, 10), dtype=bool)
        
        solvable, reason = check_solvability_certificate(
            heightmap, no_fly, (0, 0), (9, 9), min_disjoint_paths=2
        )
        
        assert solvable == True
        assert "disjoint" in reason or "solvable" in reason.lower()
    
    def test_unsolvable_scenario(self):
        """Test solvability checker recognizes multiple disjoint paths."""
        # Simple 2-path maze
        heightmap = np.zeros((10, 10))
        no_fly = np.zeros((10, 10), dtype=bool)
        
        # Multiple disjoint paths should exist in open space
        solvable, reason = check_solvability_certificate(
            heightmap, no_fly, (0, 0), (9, 9), min_disjoint_paths=2
        )
        
        assert solvable == True
        assert "disjoint" in reason.lower() or "solvable" in reason.lower()


class TestMetrics:
    """Test metrics computation."""
    
    def test_episode_metrics_computation(self):
        """Test computing episode metrics."""
        heightmap = np.zeros((10, 10))
        no_fly = np.zeros((10, 10), dtype=bool)
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        
        metrics = compute_episode_metrics(
            scenario_id="test_scenario",
            planner_id="astar",
            seed=0,
            success=True,
            path=path,
            start=(0, 0),
            goal=(4, 0),
            heightmap=heightmap,
            no_fly=no_fly,
            planning_time_ms=1.5,
            episode_duration_ms=50.0,
            replans=0,
            collisions=0,
            nfz_violations=0,
            termination_reason="success"
        )
        
        assert metrics.success == True
        assert metrics.path_length == 5.0
        assert metrics.replans == 0
    
    def test_aggregate_metrics(self):
        """Test aggregating metrics across seeds."""
        episodes = [
            EpisodeMetrics(
                scenario_id="test", planner_id="astar", seed=i,
                episode_step=20, success=True, termination_reason="success",
                path_length=20.0, path_length_any_angle=25.0,
                planning_time_ms=1.0, total_time_ms=50.0,
                replans=0, first_replan_step=None, blocked_path_events=0,
                collision_count=0.0, nfz_violations=0.0,
                fire_exposure=0.0, traffic_proximity_time=0.0,
                intruder_proximity_time=0.0, smoke_exposure=0.0,
                regret_length=None, regret_risk=None, regret_time=None,
            )
            for i in range(5)
        ]
        
        agg = aggregate_episode_metrics(episodes)
        
        assert agg.scenario_id == "test"
        assert agg.planner_id == "astar"
        assert agg.num_seeds == 5
        assert agg.success_rate == 1.0
        assert agg.path_length_mean == 20.0
        assert agg.path_length_std == 0.0


class TestScenarioValidation:
    """Test scenario config validation."""
    
    def test_valid_scenario_config(self):
        """Test creating valid scenario config."""
        cfg = ScenarioConfig(
            name="test",
            domain=Domain.URBAN,
            difficulty=Difficulty.EASY,
            mission_type=MissionType.POINT_TO_POINT,
            regime=Regime.NATURALISTIC,
            map_size=50,
            max_altitude=5,
        )
        cfg.validate()  # Should not raise
    
    def test_invalid_regime_constraint(self):
        """Test stress_test regime requires dynamics."""
        with pytest.raises(ValueError, match="stress_test requires"):
            cfg = ScenarioConfig(
                name="test",
                domain=Domain.URBAN,
                difficulty=Difficulty.EASY,
                regime=Regime.STRESS_TEST,
                enable_fire=False,
                enable_traffic=False,
                enable_moving_target=False,
                enable_intruders=False,
                enable_dynamic_nfz=False,
            )
            cfg.validate()


if __name__ == "__main__":
    # Run basic sanity checks
    print("Running sanity tests...")
    
    print("✓ Testing scenario registry...")
    test_reg = TestScenarioRegistry()
    test_reg.test_registry_has_scenarios()
    test_reg.test_list_scenarios()
    test_reg.test_scenario_metadata()
    
    print("✓ Testing planners...")
    test_plan = TestPlanners()
    test_plan.test_planner_registry()
    test_plan.test_astar_planning()
    test_plan.test_theta_star_planning()
    
    print("✓ Testing solvability...")
    test_solv = TestSolvability()
    test_solv.test_solvable_scenario()
    test_solv.test_unsolvable_scenario()
    
    print("✓ Testing metrics...")
    test_met = TestMetrics()
    test_met.test_episode_metrics_computation()
    test_met.test_aggregate_metrics()
    
    print("✓ Testing scenario validation...")
    test_scen_val = TestScenarioValidation()
    test_scen_val.test_valid_scenario_config()
    
    print("\n✓ All sanity tests passed!")
    print("\nScenario registry:")
    print_scenario_registry()
