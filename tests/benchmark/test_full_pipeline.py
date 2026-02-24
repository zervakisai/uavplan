"""Benchmark tests: full 500×500 OSM pipeline.

ALL tests here are marked @pytest.mark.slow and SKIPPED by default.
Run with:  pytest tests/benchmark --run-slow

Merges & deduplicates:
  - test_benchmark_cli (36 tests → deduplicated to ~25)
  - test_mission_episode (12 tests)
  - test_urban_env_basic (2 tests)
  - test_fair_protocol (2 tests)
  - test_forced_replan_scheduler (1 test, 500×500 variant)
  - test_dynamic_feasibility_guardrail (1 test)
  - test_p1_realism.TestP1FieldsInBenchmarkResults (1 test)

Total: ~44 unique tests.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uavbench.cli.benchmark import (
    run_planner_once,
    aggregate,
    scenario_path,
    _waypoint_action,
)
from uavbench.metrics.operational import compute_all_metrics
from uavbench.scenarios.registry import (
    SCENARIO_REGISTRY,
    list_scenarios_by_mission,
    list_scenarios_by_regime,
)
from uavbench.scenarios.schema import MissionType, Regime, InterdictionReferencePlanner
from uavbench.planners import PAPER_PLANNERS

pytestmark = pytest.mark.slow


# ═══════════════════════════════════════════════════════════════
# Waypoint → Action (fast, but logically belongs w/ benchmark CLI)
# ═══════════════════════════════════════════════════════════════

class TestWaypointAction:
    def test_move_right(self):
        assert _waypoint_action((5, 5), (6, 5)) == 3

    def test_move_left(self):
        assert _waypoint_action((5, 5), (4, 5)) == 2

    def test_move_down(self):
        assert _waypoint_action((5, 5), (5, 6)) == 1

    def test_move_up(self):
        assert _waypoint_action((5, 5), (5, 4)) == 0


# ═══════════════════════════════════════════════════════════════
# Scenario path + registry (fast, but benchmark-specific)
# ═══════════════════════════════════════════════════════════════

class TestScenarioPath:
    def test_scenario_path_exists(self):
        path = scenario_path("gov_civil_protection_easy")
        assert path.exists() and path.suffix == ".yaml"

    def test_all_gov_scenarios_resolve(self):
        for sid in SCENARIO_REGISTRY:
            assert scenario_path(sid).exists()


class TestScenarioRegistryBenchmark:
    def test_nine_scenarios(self):
        assert len(SCENARIO_REGISTRY) == 9

    def test_all_have_required_fields(self):
        for sid, meta in SCENARIO_REGISTRY.items():
            assert hasattr(meta, "scenario_id")
            assert hasattr(meta, "mission_type")
            assert hasattr(meta, "regime")

    def test_three_mission_types(self):
        types = {m.mission_type for m in SCENARIO_REGISTRY.values()}
        assert len(types) == 3

    def test_difficulties_coverage(self):
        diffs = {m.difficulty for m in SCENARIO_REGISTRY.values()}
        assert {"EASY", "MEDIUM", "HARD"}.issubset(diffs)

    def test_both_regimes(self):
        regimes = {m.regime for m in SCENARIO_REGISTRY.values()}
        assert {Regime.NATURALISTIC, Regime.STRESS_TEST}.issubset(regimes)

    def test_filter_civil_protection(self):
        assert len(list_scenarios_by_mission(MissionType.CIVIL_PROTECTION)) == 3

    def test_filter_maritime(self):
        assert len(list_scenarios_by_mission(MissionType.MARITIME_DOMAIN)) == 3


# ═══════════════════════════════════════════════════════════════
# run_planner_once (500×500 OSM)
# ═══════════════════════════════════════════════════════════════

class TestRunPlannerOnce:
    def test_astar_gov_easy(self):
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        assert isinstance(result, dict)
        for k in ("scenario_id", "planner_id", "seed", "success", "path_length", "planning_time"):
            assert k in result
        assert result["scenario_id"] == "gov_civil_protection_easy"
        assert result["planner_id"] == "astar"

    def test_theta_star_gov_easy(self):
        result = run_planner_once("gov_civil_protection_easy", "theta_star", seed=0)
        assert result["planner_id"] == "theta_star"
        assert "success" in result

    def test_deterministic_seed(self):
        r1 = run_planner_once("gov_civil_protection_easy", "astar", seed=42)
        r2 = run_planner_once("gov_civil_protection_easy", "astar", seed=42)
        assert r1["path_length"] == r2["path_length"]

    def test_result_serializable(self):
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        serializable = {k: v for k, v in result.items() if k not in ("heightmap", "no_fly", "config")}
        parsed = json.loads(json.dumps(serializable))
        assert parsed["scenario_id"] == result["scenario_id"]


# ═══════════════════════════════════════════════════════════════
# Metrics & Aggregation
# ═══════════════════════════════════════════════════════════════

class TestMetricsAggregation:
    def test_compute_all_metrics(self):
        result = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        metrics = compute_all_metrics(result)
        assert isinstance(metrics, dict) and len(metrics) > 0

    def test_aggregate_multiple(self):
        results = [
            run_planner_once("gov_civil_protection_easy", "astar", seed=i)
            for i in range(2)
        ]
        agg = aggregate(results)
        assert isinstance(agg, dict) and len(agg) > 0

    def test_aggregate_empty(self):
        assert isinstance(aggregate([]), dict)

    def test_result_keys_consistent(self):
        ra = run_planner_once("gov_civil_protection_easy", "astar", seed=0)
        rt = run_planner_once("gov_civil_protection_easy", "theta_star", seed=0)
        common = {"scenario_id", "planner_id", "seed", "success", "path_length", "planning_time"}
        assert common.issubset(set(ra.keys()))
        assert common.issubset(set(rt.keys()))


# ═══════════════════════════════════════════════════════════════
# Mini benchmark (cross-planner, cross-scenario)
# ═══════════════════════════════════════════════════════════════

class TestMiniBenchmark:
    def test_2_scenarios_2_planners(self):
        results = []
        for sid in ("gov_civil_protection_easy", "gov_maritime_domain_easy"):
            for pid in ("astar", "theta_star"):
                results.append(run_planner_once(sid, pid, seed=0))
        assert len(results) == 4
        assert all("success" in r for r in results)

    def test_first_3_paper_planners(self):
        for pid in list(PAPER_PLANNERS)[:3]:
            r = run_planner_once("gov_civil_protection_easy", pid, seed=0)
            assert r["planner_id"] == pid


# ═══════════════════════════════════════════════════════════════
# Urban Environment (500×500 OSM load)
# ═══════════════════════════════════════════════════════════════

class TestUrbanEnvOSM:
    def test_reset_and_step(self):
        from uavbench.scenarios.loader import load_scenario
        from uavbench.envs.urban import UrbanEnv

        cfg = load_scenario(scenario_path("gov_civil_protection_easy"))
        env = UrbanEnv(cfg)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(5):
            obs, *_ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)

    def test_trajectory_log(self):
        from uavbench.scenarios.loader import load_scenario
        from uavbench.envs.urban import UrbanEnv

        cfg = load_scenario(scenario_path("gov_civil_protection_easy"))
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        for _ in range(3):
            env.step(env.action_space.sample())
        traj = env.trajectory
        assert len(traj) == 3 and traj[0]["step"] == 1


# ═══════════════════════════════════════════════════════════════
# Fair Protocol (500×500 OSM)
# ═══════════════════════════════════════════════════════════════

class TestFairProtocol:
    def test_protocol_defaults_present(self):
        from uavbench.scenarios.loader import load_scenario

        cfg = load_scenario(scenario_path("gov_civil_protection_easy"))
        assert cfg.interdiction_reference_planner == InterdictionReferencePlanner.ASTAR
        assert cfg.plan_budget_static_ms > 0.0
        assert cfg.replan_every_steps >= 1

    def test_reference_planner_logged_as_bfs(self):
        from uavbench.scenarios.loader import load_scenario
        from uavbench.envs.urban import UrbanEnv

        cfg = load_scenario(scenario_path("gov_civil_protection_hard"))
        cfg = replace(
            cfg,
            interdiction_reference_planner=InterdictionReferencePlanner.ASTAR,
            force_replan_count=1,
            event_t1=8,
            event_t2=None,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        env._maybe_trigger_interdictions(8)
        ev = next(e for e in env.events if e["type"] == "path_interdiction_1")
        assert ev["payload"]["reference_planner"] == "bfs_shortest_path"


# ═══════════════════════════════════════════════════════════════
# Forced Replan Scheduler (500×500 OSM)
# ═══════════════════════════════════════════════════════════════

class TestForcedInterdictionsOSM:
    def test_interdictions_trigger_events(self):
        from uavbench.scenarios.loader import load_scenario
        from uavbench.envs.urban import UrbanEnv

        cfg = load_scenario(scenario_path("gov_civil_protection_hard"))
        cfg = replace(cfg, force_replan_count=2, event_t1=12, event_t2=28)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        env._maybe_trigger_interdictions(12)
        env._maybe_trigger_interdictions(28)
        types = [e["type"] for e in env.events]
        assert "path_interdiction_1" in types
        assert "path_interdiction_2" in types
        assert "forced_replan_triggered" in types


# ═══════════════════════════════════════════════════════════════
# Dynamic Feasibility Guardrail (500×500 OSM)
# ═══════════════════════════════════════════════════════════════

class TestFeasibilityGuardrail:
    def test_guardrail_relaxes_disconnected_state(self):
        from uavbench.scenarios.loader import load_scenario
        from uavbench.envs.urban import UrbanEnv

        cfg = load_scenario(scenario_path("gov_civil_protection_hard"))
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        env._forced_block_mask[:] = True
        current = (int(env._agent_pos[0]), int(env._agent_pos[1]))
        goal = (int(env._goal_pos[0]), int(env._goal_pos[1]))
        relaxed = env._enforce_feasibility_guardrail(current, goal)
        assert relaxed
        status = env._last_guardrail_status
        assert status["reachability_failed_before_relax"] is True
        assert status["feasible_after_guardrail"] is True


# ═══════════════════════════════════════════════════════════════
# P1 Realism in Benchmark Results (500×500 OSM)
# ═══════════════════════════════════════════════════════════════

class TestP1RealismBenchmark:
    def test_realism_fields_in_run_dynamic_episode(self):
        from uavbench.cli.benchmark import run_dynamic_episode

        result = run_dynamic_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=0,
            config_overrides={
                "constraint_latency_steps": 2,
                "comms_dropout_prob": 0.1,
                "gnss_noise_sigma": 0.5,
            },
        )
        assert result["constraint_latency_steps"] == 2
        assert result["comms_dropout_prob"] == 0.1
        assert result["gnss_noise_sigma"] == 0.5
        assert result["replan_mode"] in ("native", "harness_replan")


# ═══════════════════════════════════════════════════════════════
# Mission Episode (500×500 OSM)
# ═══════════════════════════════════════════════════════════════

class TestMissionEpisode:
    def test_basic_civil_protection(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode("gov_civil_protection_easy", "astar", seed=42)
        for k in ("success", "path_length", "episode_steps", "mission_id",
                   "mission_score", "task_completion_rate", "mission_metrics"):
            assert k in result
        assert result["mission_id"] == "civil_protection"

    def test_keys_superset_of_dynamic(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode("gov_civil_protection_easy", "astar", seed=0)
        required = [
            "scenario", "planner", "seed", "success", "path_length",
            "planning_time_ms", "episode_steps", "total_replans",
            "constraint_latency_steps", "comms_dropout_prob",
            "gnss_noise_sigma", "replan_mode",
        ]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_maritime_domain(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode("gov_maritime_domain_easy", "astar", seed=42)
        assert result["mission_id"] == "maritime_domain"

    def test_critical_infrastructure(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode("gov_critical_infrastructure_easy", "astar", seed=42)
        assert result["mission_id"] == "critical_infrastructure"

    def test_greedy_policy(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode(
            "gov_civil_protection_easy", "astar", seed=1, policy_id="greedy",
        )
        assert result["policy_id"] == "greedy" and result["path_length"] > 1

    def test_lookahead_policy(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode(
            "gov_civil_protection_easy", "astar", seed=1, policy_id="lookahead",
        )
        assert result["policy_id"] == "lookahead"

    def test_adaptive_planner(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode(
            "gov_civil_protection_easy", "periodic_replan", seed=42,
        )
        assert result["replan_mode"] == "native"

    def test_mission_metrics_populated(self):
        from uavbench.cli.benchmark import run_mission_episode

        mm = run_mission_episode(
            "gov_civil_protection_easy", "astar", seed=42,
        )["mission_metrics"]
        assert "mission_score" in mm and "task_completion_rate" in mm

    def test_p1_realism_fields_propagate(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode("gov_civil_protection_easy", "astar", seed=42)
        assert result["constraint_latency_steps"] == 0
        assert result["comms_dropout_prob"] == 0.0

    def test_deterministic(self):
        from uavbench.cli.benchmark import run_mission_episode

        r1 = run_mission_episode("gov_civil_protection_easy", "astar", seed=99)
        r2 = run_mission_episode("gov_civil_protection_easy", "astar", seed=99)
        assert r1["path_length"] == r2["path_length"]
        assert r1["mission_score"] == r2["mission_score"]

    def test_hard_terminates(self):
        from uavbench.cli.benchmark import run_mission_episode

        result = run_mission_episode(
            "gov_civil_protection_hard", "astar", seed=42,
            episode_horizon_steps=100,
        )
        assert result["episode_steps"] <= 100
