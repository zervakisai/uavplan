"""Unit tests: Scenario registry, config validation, P1 realism schema, track partition.

Merges & deduplicates:
  - test_sanity.TestScenarioRegistry (3 tests)
  - test_sanity.TestScenarioValidation (2 tests)
  - test_sanity.TestMetrics (2 tests)
  - test_sanity.TestSolvability (2 tests)
  - test_p1_realism (9 fast schema tests — excludes 1 slow run_dynamic_episode)
  - test_scenario_basic (3 config-load tests)
  - test_track_partition (1 test)

Total: ~22 unique tests.  Runtime: < 1 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uavbench.scenarios.registry import (
    SCENARIO_REGISTRY,
    list_scenarios,
    list_scenarios_by_regime,
    list_scenarios_by_track,
    list_scenarios_with_dynamics,
    print_scenario_registry,
)
from uavbench.scenarios.schema import (
    Domain,
    Difficulty,
    MissionType,
    Regime,
    ScenarioConfig,
)
from uavbench.scenarios.loader import load_scenario
from uavbench.metrics.comprehensive import (
    EpisodeMetrics,
    compute_episode_metrics,
    aggregate_episode_metrics,
)
from uavbench.benchmark.solvability import check_solvability_certificate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── helpers ───────────────────────────────────────────────────

def _base_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        name="test_schema",
        domain=Domain.URBAN,
        difficulty=Difficulty.EASY,
        map_size=10,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


# ── Scenario Registry ────────────────────────────────────────

class TestScenarioRegistry:
    def test_registry_has_six_scenarios(self):
        assert len(SCENARIO_REGISTRY) == 6

    def test_list_scenarios(self):
        all_scen = list_scenarios()
        assert len(all_scen) == 6
        gov = [s for s in all_scen if s.startswith("gov_")]
        assert len(gov) == 6

    def test_naturalistic_plus_stress_equals_all(self):
        n = list_scenarios_by_regime(Regime.NATURALISTIC)
        s = list_scenarios_by_regime(Regime.STRESS_TEST)
        assert len(n) + len(s) == len(list_scenarios())

    def test_dynamic_scenarios_list(self):
        dynamic = list_scenarios_with_dynamics()
        assert isinstance(dynamic, list)

    def test_scenario_metadata_civil_protection_medium(self):
        sid = "gov_civil_protection_medium"
        assert sid in SCENARIO_REGISTRY
        meta = SCENARIO_REGISTRY[sid]
        assert meta.mission_type == MissionType.CIVIL_PROTECTION
        assert meta.regime == Regime.STRESS_TEST


# ── Track Partition ───────────────────────────────────────────

class TestTrackPartition:
    def test_all_scenarios_are_dynamic(self):
        """All 6 gov scenarios are dynamic (stress_test regime)."""
        all_ids = set(list_scenarios())
        static_ids = set(list_scenarios_by_track("static"))
        dynamic_ids = set(list_scenarios_by_track("dynamic"))
        assert len(static_ids) == 0, "No static scenarios after easy removal"
        assert dynamic_ids == all_ids


# ── Scenario Config Load ─────────────────────────────────────

class TestScenarioConfigLoad:
    _CONFIGS = _PROJECT_ROOT / "src" / "uavbench" / "scenarios" / "configs"

    def test_gov_civil_protection_medium(self):
        cfg = load_scenario(self._CONFIGS / "gov_civil_protection_medium.yaml")
        assert cfg.domain.value == "urban"
        assert cfg.difficulty.value == "medium"

    def test_gov_civil_protection_hard(self):
        cfg = load_scenario(self._CONFIGS / "gov_civil_protection_hard.yaml")
        assert cfg.difficulty.value == "hard"
        assert cfg.domain.value == "urban"

    def test_gov_maritime_domain_medium(self):
        cfg = load_scenario(self._CONFIGS / "gov_maritime_domain_medium.yaml")
        assert cfg.difficulty.value == "medium"
        assert cfg.no_fly_radius >= 0


# ── Scenario Config Validation ────────────────────────────────

class TestScenarioValidation:
    def test_valid_config(self):
        cfg = ScenarioConfig(
            name="test",
            domain=Domain.URBAN,
            difficulty=Difficulty.EASY,
            mission_type=MissionType.POINT_TO_POINT,
            regime=Regime.NATURALISTIC,
            map_size=50,
            max_altitude=5,
        )
        cfg.validate()

    def test_stress_test_without_dynamics_raises(self):
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


# ── P1 Realism Schema ────────────────────────────────────────

class TestConstraintLatencySchema:
    def test_default_is_zero(self):
        assert _base_config().constraint_latency_steps == 0

    def test_valid_positive(self):
        cfg = _base_config(constraint_latency_steps=3)
        cfg.validate()
        assert cfg.constraint_latency_steps == 3

    def test_negative_raises(self):
        cfg = _base_config(constraint_latency_steps=-1)
        with pytest.raises(ValueError, match="constraint_latency_steps"):
            cfg.validate()


class TestCommsDropoutSchema:
    def test_default_is_zero(self):
        assert _base_config().comms_dropout_prob == 0.0

    def test_valid_range(self):
        for p in (0.0, 0.5, 1.0):
            _base_config(comms_dropout_prob=p).validate()

    def test_out_of_range_raises(self):
        for p in (-0.1, 1.1):
            cfg = _base_config(comms_dropout_prob=p)
            with pytest.raises(ValueError, match="comms_dropout_prob"):
                cfg.validate()


class TestGNSSNoiseSchema:
    def test_default_is_zero(self):
        assert _base_config().gnss_noise_sigma == 0.0

    def test_valid_positive(self):
        cfg = _base_config(gnss_noise_sigma=1.5)
        cfg.validate()
        assert cfg.gnss_noise_sigma == 1.5

    def test_negative_raises(self):
        cfg = _base_config(gnss_noise_sigma=-0.5)
        with pytest.raises(ValueError, match="gnss_noise_sigma"):
            cfg.validate()


# ── Fairness Fields ───────────────────────────────────────────

class TestFairnessFields:
    def test_fairness_defaults_present(self):
        cfg = ScenarioConfig(
            name="ff",
            domain=Domain.URBAN,
            difficulty=Difficulty.EASY,
            mission_type=MissionType.CIVIL_PROTECTION,
            regime=Regime.NATURALISTIC,
            map_size=32,
            map_source="synthetic",
        )
        assert cfg.plan_budget_static_ms > 0
        assert cfg.plan_budget_dynamic_ms > 0
        assert cfg.replan_every_steps >= 1
        assert cfg.max_replans_per_episode >= 1


# ── Solvability ──────────────────────────────────────────────

class TestSolvability:
    def test_solvable_on_open_grid(self):
        h = np.zeros((10, 10))
        nfz = np.zeros((10, 10), dtype=bool)
        ok, reason = check_solvability_certificate(h, nfz, (0, 0), (9, 9), min_disjoint_paths=2)
        assert ok
        assert "disjoint" in reason.lower() or "solvable" in reason.lower()

    def test_multiple_disjoint_paths(self):
        h = np.zeros((10, 10))
        nfz = np.zeros((10, 10), dtype=bool)
        ok, reason = check_solvability_certificate(h, nfz, (0, 0), (9, 9), min_disjoint_paths=2)
        assert ok


# ── Metrics ──────────────────────────────────────────────────

class TestMetrics:
    def test_episode_metrics_computation(self):
        h = np.zeros((10, 10))
        nfz = np.zeros((10, 10), dtype=bool)
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        m = compute_episode_metrics(
            scenario_id="test", planner_id="astar", seed=0,
            success=True, path=path, start=(0, 0), goal=(4, 0),
            heightmap=h, no_fly=nfz, planning_time_ms=1.5,
            episode_duration_ms=50.0, replans=0, collisions=0,
            nfz_violations=0, termination_reason="success",
        )
        assert m.success
        assert m.path_length == 5.0
        assert m.replans == 0

    def test_aggregate_metrics(self):
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
        assert agg.num_seeds == 5
        assert agg.success_rate == 1.0
        assert agg.path_length_mean == 20.0
        assert agg.path_length_std == 0.0


class TestRiskWeightValidation:
    """Verify M6 fix: risk weights must sum to 1.0."""

    def test_default_weights_valid(self):
        sc = ScenarioConfig(name="rw", domain=Domain.URBAN, difficulty=Difficulty.EASY)
        sc.validate()  # should not raise

    def test_weights_not_summing_to_one_rejected(self):
        sc = ScenarioConfig(
            name="rw", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            risk_weight_population=0.5,
            risk_weight_adversarial=0.3,
            risk_weight_smoke=0.3,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            sc.validate()

    def test_zero_weights_rejected(self):
        sc = ScenarioConfig(
            name="rw", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            risk_weight_population=0.0,
            risk_weight_adversarial=0.0,
            risk_weight_smoke=0.0,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            sc.validate()

    def test_custom_valid_weights_accepted(self):
        sc = ScenarioConfig(
            name="rw", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            risk_weight_population=0.4,
            risk_weight_adversarial=0.4,
            risk_weight_smoke=0.2,
        )
        sc.validate()  # should not raise


class TestDowntownWindowGuard:
    """Verify M7 fix: downtown_window >= map_size is rejected."""

    def test_downtown_window_equals_map_size_rejected(self):
        sc = ScenarioConfig(
            name="dw", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            map_size=25, downtown_window=25,
        )
        with pytest.raises(ValueError, match="downtown_window"):
            sc.validate()

    def test_downtown_window_larger_than_map_size_rejected(self):
        sc = ScenarioConfig(
            name="dw", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            map_size=25, downtown_window=31,
        )
        with pytest.raises(ValueError, match="downtown_window"):
            sc.validate()

    def test_small_downtown_window_accepted(self):
        sc = ScenarioConfig(
            name="dw", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            map_size=25, downtown_window=7,
        )
        sc.validate()  # should not raise


# ── Physics Constants (V&V Fix: cell_size_m / dt_s) ─────────────

class TestPhysicsConstants:
    """Verify V&V physics constants are present and correct."""

    def test_default_cell_size_m(self):
        cfg = _base_config()
        assert cfg.cell_size_m == 5.0

    def test_default_dt_s(self):
        cfg = _base_config()
        assert cfg.dt_s == 1.0

    def test_custom_cell_size_accepted(self):
        cfg = _base_config(cell_size_m=2.5)
        cfg.validate()
        assert cfg.cell_size_m == 2.5

    def test_custom_dt_s_accepted(self):
        cfg = _base_config(dt_s=0.5)
        cfg.validate()
        assert cfg.dt_s == 0.5

    def test_physics_constants_loaded_from_yaml(self):
        """Loader preserves cell_size_m / dt_s from YAML (defaults if absent)."""
        from pathlib import Path
        configs_dir = Path(__file__).resolve().parents[2] / "src" / "uavbench" / "scenarios" / "configs"
        from uavbench.scenarios.loader import load_scenario
        cfg = load_scenario(configs_dir / "gov_civil_protection_medium.yaml")
        # Defaults applied: 5.0 m / cell, 1.0 s / step
        assert cfg.cell_size_m == 5.0
        assert cfg.dt_s == 1.0
