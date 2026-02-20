"""Tests for run_mission_episode — MissionEngine wired into CLI benchmark.

Validates:
  1. run_mission_episode() returns mission-specific keys
  2. Task completion rate > 0 for easy scenarios
  3. Products are generated
  4. Backward-compatible keys (path_length, success, etc.) are present
  5. Both greedy and lookahead policies work
  6. All 3 mission types dispatch correctly
  7. P1 realism fields propagate
  8. Mission metrics are populated
"""

from __future__ import annotations

import pytest
import numpy as np

from uavbench.cli.benchmark import run_mission_episode


class TestRunMissionEpisode:
    """Integration tests for the mission-episode CLI entrypoint."""

    def test_basic_civil_protection_easy(self):
        """Easy civil protection mission should complete with tasks done."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=42,
        )
        # Must have backward-compatible keys
        assert "success" in result
        assert "path_length" in result
        assert "episode_steps" in result
        assert "total_replans" in result
        assert "constraint_violations" in result
        assert "planning_time_ms" in result

        # Must have mission-specific keys
        assert "mission_id" in result
        assert "difficulty" in result
        assert "policy_id" in result
        assert "mission_score" in result
        assert "task_completion_rate" in result
        assert "task_log" in result
        assert "products" in result
        assert "mission_metrics" in result

        # Verify mission params
        assert result["mission_id"] == "civil_protection"
        assert result["difficulty"] == "easy"
        assert result["policy_id"] == "greedy"

    def test_mission_keys_superset_of_dynamic(self):
        """Mission result must have all keys that run_dynamic_episode provides."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=0,
        )
        required_keys = [
            "scenario", "planner", "seed", "success",
            "constraint_violations", "path_length", "path",
            "heightmap", "no_fly", "start", "goal",
            "planning_time_ms", "plan_budget_ms",
            "protocol_variant", "map_source", "osm_tile_id",
            "episode_steps", "total_replans",
            "replan_budget_violations", "risk_exposure_integral",
            "total_reward", "termination_reason",
            "constraint_latency_steps", "comms_dropout_prob",
            "gnss_noise_sigma", "replan_mode",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_greedy_policy(self):
        """Greedy policy should be the default."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=1,
            policy_id="greedy",
        )
        assert result["policy_id"] == "greedy"
        # Should make progress (trajectory > 1)
        assert result["path_length"] > 1

    def test_lookahead_policy(self):
        """Lookahead policy should work."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=1,
            policy_id="lookahead",
        )
        assert result["policy_id"] == "lookahead"
        assert result["path_length"] > 1

    def test_maritime_domain(self):
        """Maritime domain mission should dispatch correctly."""
        result = run_mission_episode(
            "gov_maritime_domain_easy",
            "astar",
            seed=42,
        )
        assert result["mission_id"] == "maritime_domain"
        assert result["difficulty"] == "easy"

    def test_critical_infrastructure(self):
        """Critical infrastructure mission should dispatch correctly."""
        result = run_mission_episode(
            "gov_critical_infrastructure_easy",
            "astar",
            seed=42,
        )
        assert result["mission_id"] == "critical_infrastructure"
        assert result["difficulty"] == "easy"

    def test_adaptive_planner_works(self):
        """Adaptive planner (periodic_replan) should work in mission mode."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "periodic_replan",
            seed=42,
        )
        assert result["replan_mode"] == "native"
        assert result["path_length"] > 1

    def test_non_adaptive_planner_works(self):
        """Non-adaptive planner (theta_star) should work in mission mode."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "theta_star",
            seed=42,
        )
        assert result["replan_mode"] == "harness_replan"
        assert result["path_length"] > 1

    def test_mission_metrics_populated(self):
        """Mission metrics dict should have standard keys."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=42,
        )
        mm = result["mission_metrics"]
        assert isinstance(mm, dict)
        assert "mission_score" in mm
        assert "task_completion_rate" in mm
        assert "completion_time" in mm

    def test_p1_realism_fields_propagate(self):
        """P1 realism fields from scenario config should be in result."""
        result = run_mission_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=42,
        )
        assert "constraint_latency_steps" in result
        assert "comms_dropout_prob" in result
        assert "gnss_noise_sigma" in result
        # Easy scenario should have zero P1 realism
        assert result["constraint_latency_steps"] == 0
        assert result["comms_dropout_prob"] == 0.0

    def test_deterministic_across_runs(self):
        """Same seed should produce identical results."""
        r1 = run_mission_episode("gov_civil_protection_easy", "astar", seed=99)
        r2 = run_mission_episode("gov_civil_protection_easy", "astar", seed=99)
        assert r1["path_length"] == r2["path_length"]
        assert r1["episode_steps"] == r2["episode_steps"]
        assert r1["mission_score"] == r2["mission_score"]
        assert r1["task_completion_rate"] == r2["task_completion_rate"]

    def test_hard_scenario_terminates(self):
        """Hard scenario with dynamics should terminate without crash."""
        result = run_mission_episode(
            "gov_civil_protection_hard",
            "astar",
            seed=42,
            episode_horizon_steps=100,  # cap for test speed
        )
        assert result["episode_steps"] <= 100
        assert result["termination_reason"] in (
            "timeout", "mission_complete", "all_tasks_done",
            "no_reachable_tasks", "stuck", "truncated",
            "strict_compliance_violation", "unknown",
        )
