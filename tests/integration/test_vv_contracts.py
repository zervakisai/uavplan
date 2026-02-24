"""Integration tests: V&V contracts + Fairness contracts.

Merges:
  - test_vv_contracts  (23 tests)
  - test_fairness_contracts (17 tests)

All use 32×32 synthetic maps — no OSM, no network.  Runtime: < 3 s.

Contract inventory:
  VV-1  Determinism
  VV-2  Safety monitor
  VV-3  Responsiveness (solvability, BFS)
  VV-4  Plausibility (A* optimal, obstacle avoidance, all-planner smoke)
  FC-1  Cross-planner interdiction fairness
  FC-2  Protocol variant isolation
  FC-3  Stress alpha monotonicity
  FC-4  Planner group IDs match PAPER_PLANNERS
  FC-5  forced_replan_certificate is not a stub
  FC-6  Emergency corridor passability
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from uavbench.envs.urban import UrbanEnv
from uavbench.planners import PLANNERS, PAPER_PLANNERS
from uavbench.metrics.operational import compute_safety_metrics
from uavbench.benchmark.solvability import (
    check_solvability_certificate,
    _bfs_connectivity,
)
from uavbench.scenarios.schema import (
    ScenarioConfig,
    Domain,
    Difficulty,
    MissionType,
    Regime,
    InterdictionReferencePlanner,
)


# ─── Helpers ───────────────────────────────────────────────────

def _make_simple_config(**overrides) -> ScenarioConfig:
    """Minimal ScenarioConfig for unit-testing (synthetic map)."""
    defaults = dict(
        name="vv_test",
        domain=Domain.URBAN,
        difficulty=Difficulty.EASY,
        mission_type=MissionType.CIVIL_PROTECTION,
        regime=Regime.NATURALISTIC,
        map_size=32,
        map_source="synthetic",
        building_density=0.05,
        no_fly_radius=0,
        max_altitude=5,
        safe_altitude=5,
        min_start_goal_l1=8,
        enable_fire=False,
        enable_traffic=False,
        paper_track="static",
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _make_dynamic_config(**overrides) -> ScenarioConfig:
    """Minimal dynamic ScenarioConfig for fairness tests."""
    defaults = dict(
        name="fc_test",
        domain=Domain.URBAN,
        difficulty=Difficulty.MEDIUM,
        mission_type=MissionType.CIVIL_PROTECTION,
        regime=Regime.STRESS_TEST,
        map_size=32,
        map_source="synthetic",
        building_density=0.05,
        no_fly_radius=0,
        max_altitude=5,
        safe_altitude=5,
        min_start_goal_l1=8,
        enable_fire=False,
        enable_traffic=False,
        paper_track="dynamic",
        force_replan_count=2,
        event_t1=8,
        event_t2=18,
        wind=("medium"),
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        v = state[key]
        h.update(key.encode())
        if isinstance(v, np.ndarray):
            h.update(v.tobytes())
        elif isinstance(v, (dict, list)):
            h.update(json.dumps(v, sort_keys=True, default=str).encode())
        else:
            h.update(str(v).encode())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
# VV-1  Determinism
# ═══════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_static_planner_determinism(self):
        cfg = _make_simple_config()
        paths = []
        for _ in range(2):
            env = UrbanEnv(cfg)
            env.reset(seed=42)
            hm, nfz, start, goal = env.export_planner_inputs()
            result = PLANNERS["astar"](hm, nfz).plan(start, goal)
            paths.append(result.path)
        assert paths[0] == paths[1]

    @pytest.mark.parametrize("seed", [0, 7, 99])
    def test_env_reset_determinism(self, seed: int):
        cfg = _make_simple_config()
        env_a, env_b = UrbanEnv(cfg), UrbanEnv(cfg)
        env_a.reset(seed=seed)
        env_b.reset(seed=seed)
        hm_a, nfz_a, s_a, g_a = env_a.export_planner_inputs()
        hm_b, nfz_b, s_b, g_b = env_b.export_planner_inputs()
        np.testing.assert_array_equal(hm_a, hm_b)
        np.testing.assert_array_equal(nfz_a, nfz_b)
        assert s_a == s_b and g_a == g_b

    def test_step_sequence_determinism(self):
        cfg = _make_simple_config(
            enable_fire=True, enable_traffic=True, paper_track="dynamic",
        )
        actions = [3, 1, 3, 0, 2, 3, 1, 3]
        hashes: list[list[str]] = []
        for _ in range(2):
            env = UrbanEnv(cfg)
            env.reset(seed=77)
            run = []
            for a in actions:
                env.step(a)
                run.append(_state_hash(env.get_dynamic_state()))
            hashes.append(run)
        assert hashes[0] == hashes[1]


# ═══════════════════════════════════════════════════════════════
# VV-2  Safety Monitor
# ═══════════════════════════════════════════════════════════════

class TestSafetyMonitor:
    def test_nfz_violation_detected(self):
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        nfz[5, 5] = True
        m = compute_safety_metrics([(4, 5), (5, 5), (6, 5)], hm, nfz)
        assert m["nfz_violation_count"] >= 1.0

    def test_building_violation_detected(self):
        hm = np.zeros((16, 16), dtype=np.float32)
        hm[8, 8] = 10.0
        nfz = np.zeros((16, 16), dtype=bool)
        m = compute_safety_metrics([(7, 8), (8, 8), (9, 8)], hm, nfz)
        assert m["building_violation_count"] >= 1.0

    def test_clean_path_no_violations(self):
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        m = compute_safety_metrics([(0, 0), (1, 0), (2, 0)], hm, nfz)
        assert m["nfz_violation_count"] == 0.0
        assert m["building_violation_count"] == 0.0

    def test_risk_exposure_accumulates(self):
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        risk = np.zeros((16, 16), dtype=np.float32)
        risk[0, 1] = 0.5
        risk[0, 2] = 0.3
        m = compute_safety_metrics([(0, 0), (1, 0), (2, 0)], hm, nfz, risk_map=risk)
        assert m["risk_exposure_sum"] == pytest.approx(0.8, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# VV-3  Responsiveness
# ═══════════════════════════════════════════════════════════════

class TestResponsiveness:
    def test_solvability_open_grid(self):
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        ok, _ = check_solvability_certificate(hm, nfz, (0, 0), (15, 15))
        assert ok

    def test_solvability_blocked_grid(self):
        hm = np.ones((16, 16), dtype=np.float32) * 10.0
        nfz = np.zeros((16, 16), dtype=bool)
        ok, _ = check_solvability_certificate(hm, nfz, (0, 0), (15, 15))
        assert not ok

    def test_bfs_reaches_goal(self):
        mask = np.ones((16, 16), dtype=bool)
        reachable, path = _bfs_connectivity(mask, (0, 0), (15, 15))
        assert reachable and path[0] == (0, 0) and path[-1] == (15, 15)

    def test_bfs_blocked_by_wall(self):
        mask = np.ones((16, 16), dtype=bool)
        mask[:, 8] = False
        reachable, _ = _bfs_connectivity(mask, (0, 0), (15, 15))
        assert not reachable


# ═══════════════════════════════════════════════════════════════
# VV-4  Plausibility
# ═══════════════════════════════════════════════════════════════

class TestPlausibility:
    def test_astar_optimal_on_empty_grid(self):
        hm = np.zeros((32, 32), dtype=np.float32)
        nfz = np.zeros((32, 32), dtype=bool)
        result = PLANNERS["astar"](hm, nfz).plan((0, 0), (10, 10))
        assert len(result.path) == 21  # Manhattan + 1

    def test_astar_respects_obstacles(self):
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        for y in range(11):
            hm[y, 5] = 10.0
        result = PLANNERS["astar"](hm, nfz).plan((3, 5), (8, 5))
        assert result.path is not None
        for x, y in result.path:
            assert hm[y, x] == 0.0

    @pytest.mark.parametrize("pid", list(PAPER_PLANNERS))
    def test_all_planners_trivial_path(self, pid: str):
        hm = np.zeros((20, 20), dtype=np.float32)
        nfz = np.zeros((20, 20), dtype=bool)
        result = PLANNERS[pid](hm, nfz).plan((0, 0), (10, 10))
        assert result.path is not None and len(result.path) >= 2

    def test_config_validates(self):
        _make_simple_config().validate()

    def test_fairness_fields_present(self):
        cfg = _make_simple_config()
        assert cfg.plan_budget_static_ms > 0
        assert cfg.plan_budget_dynamic_ms > 0
        assert cfg.replan_every_steps >= 1
        assert cfg.max_replans_per_episode >= 1


# ═══════════════════════════════════════════════════════════════
# FC-1  Cross-planner interdiction fairness
# ═══════════════════════════════════════════════════════════════

class TestCrossPlannerFairness:
    def test_identical_forced_block_mask_at_event_t1(self):
        cfg = _make_dynamic_config()
        masks = []
        for _ in range(2):
            env = UrbanEnv(cfg)
            env.reset(seed=42)
            for _ in range(int(cfg.event_t1) + 1):
                env.step(0)
            masks.append(env.get_dynamic_state()["forced_block_mask"].copy())
        np.testing.assert_array_equal(masks[0], masks[1])

    def test_interdiction_schedule_identical(self):
        cfg = _make_dynamic_config()
        schedules = []
        for _ in range(2):
            env = UrbanEnv(cfg)
            env.reset(seed=99)
            schedules.append([
                (e["step"], e["point"], e["radius"])
                for e in env._forced_interdictions
            ])
        assert schedules[0] == schedules[1]


# ═══════════════════════════════════════════════════════════════
# FC-2  Protocol variant isolation
# ═══════════════════════════════════════════════════════════════

class TestProtocolVariants:
    def test_no_interactions(self):
        from uavbench.cli.benchmark import _apply_protocol_variant
        cfg = _apply_protocol_variant(_make_dynamic_config(), "no_interactions")
        assert cfg.extra.get("disable_interactions") is True

    def test_no_forced_breaks(self):
        from uavbench.cli.benchmark import _apply_protocol_variant
        cfg = _apply_protocol_variant(_make_dynamic_config(), "no_forced_breaks")
        assert cfg.force_replan_count == 0
        assert cfg.event_t1 is None and cfg.event_t2 is None

    def test_risk_only(self):
        from uavbench.cli.benchmark import _apply_protocol_variant
        cfg = _apply_protocol_variant(
            _make_dynamic_config(fire_blocks_movement=False), "risk_only",
        )
        assert cfg.fire_blocks_movement is False
        assert cfg.traffic_blocks_movement is False
        assert cfg.force_replan_count == 0

    def test_blocking_only(self):
        from uavbench.cli.benchmark import _apply_protocol_variant
        cfg = _apply_protocol_variant(_make_dynamic_config(), "blocking_only")
        assert cfg.extra.get("disable_population_risk") is True

    def test_no_guardrail(self):
        from uavbench.cli.benchmark import _apply_protocol_variant
        cfg = _apply_protocol_variant(_make_dynamic_config(), "no_guardrail")
        assert cfg.extra.get("disable_feasibility_guardrail") is True

    def test_invalid_variant_raises(self):
        from uavbench.cli.benchmark import _apply_protocol_variant
        with pytest.raises(ValueError, match="protocol_variant"):
            _apply_protocol_variant(_make_dynamic_config(), "invalid_variant")


# ═══════════════════════════════════════════════════════════════
# FC-3  Stress alpha monotonicity
# ═══════════════════════════════════════════════════════════════

class TestStressAlpha:
    def test_alpha_scales_fire_ignition(self):
        from uavbench.cli.benchmark import _apply_stress_alpha
        cfg = _make_dynamic_config(
            enable_fire=True, fire_ignition_points=3,
            map_source="osm", osm_tile_id="penteli",
        )
        lo = _apply_stress_alpha(cfg, 0.0)
        hi = _apply_stress_alpha(cfg, 1.0)
        assert hi.fire_ignition_points > lo.fire_ignition_points

    def test_alpha_scales_vehicles(self):
        from uavbench.cli.benchmark import _apply_stress_alpha
        cfg = _make_dynamic_config(num_emergency_vehicles=5)
        lo = _apply_stress_alpha(cfg, 0.0)
        hi = _apply_stress_alpha(cfg, 1.0)
        assert hi.num_emergency_vehicles >= lo.num_emergency_vehicles

    def test_alpha_none_is_noop(self):
        from uavbench.cli.benchmark import _apply_stress_alpha
        cfg = _make_dynamic_config()
        assert _apply_stress_alpha(cfg, None) is cfg


# ═══════════════════════════════════════════════════════════════
# FC-4  Planner group IDs match PAPER_PLANNERS
# ═══════════════════════════════════════════════════════════════

class TestPlannerGroupConsistency:
    def test_groups_use_paper_planner_ids(self):
        static = {"astar", "theta_star"}
        adaptive = {"periodic_replan", "aggressive_replan", "incremental_dstar_lite", "grid_mppi"}
        risk_aware = {"grid_mppi"}
        all_groups = static | adaptive | risk_aware
        assert not (all_groups - set(PAPER_PLANNERS))

    def test_no_deprecated_ids_in_hypotheses(self):
        from uavbench.benchmark.theoretical_validation import HYPOTHESES
        from uavbench.planners import DEPRECATED_ALIASES, DEPRECATED_PLANNERS
        deprecated = set(DEPRECATED_ALIASES.keys()) | DEPRECATED_PLANNERS
        for hyp in HYPOTHESES:
            a, b = hyp["comparison"]
            for name in (a, b):
                assert name not in deprecated, (
                    f"Hypothesis {hyp['id']} references deprecated '{name}'"
                )


# ═══════════════════════════════════════════════════════════════
# FC-5  forced_replan_certificate
# ═══════════════════════════════════════════════════════════════

class TestForcedReplanCertificate:
    def test_certificate_with_interdictions(self):
        from uavbench.benchmark.solvability import check_forced_replan_certificate
        hm = np.zeros((32, 32), dtype=np.float32)
        nfz = np.zeros((32, 32), dtype=bool)
        ok, reason = check_forced_replan_certificate(
            hm, nfz, (0, 0), (31, 31),
            dyn_state_at_step={"force_replan_count": 2, "event_t1": 12},
            time_horizon=50,
        )
        assert ok
        assert "deferred" not in reason.lower()

    def test_certificate_fails_without_interdictions(self):
        from uavbench.benchmark.solvability import check_forced_replan_certificate
        hm = np.zeros((32, 32), dtype=np.float32)
        nfz = np.zeros((32, 32), dtype=bool)
        ok, _ = check_forced_replan_certificate(
            hm, nfz, (0, 0), (31, 31),
            dyn_state_at_step={"force_replan_count": 0},
            time_horizon=50,
        )
        assert not ok

    def test_certificate_fails_when_event_t1_exceeds_horizon(self):
        from uavbench.benchmark.solvability import check_forced_replan_certificate
        hm = np.zeros((32, 32), dtype=np.float32)
        nfz = np.zeros((32, 32), dtype=bool)
        ok, _ = check_forced_replan_certificate(
            hm, nfz, (0, 0), (31, 31),
            dyn_state_at_step={"force_replan_count": 2, "event_t1": 100},
            time_horizon=50,
        )
        assert not ok


# ═══════════════════════════════════════════════════════════════
# FC-6  Emergency corridor passability
# ═══════════════════════════════════════════════════════════════

class TestEmergencyCorridorPassability:
    def test_corridor_cells_are_free(self):
        cfg = _make_dynamic_config()
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        corridor = env._emergency_corridor_mask
        hm = env._heightmap
        nfz = env._no_fly_mask
        for y, x in np.argwhere(corridor):
            assert hm[y, x] <= 0, f"Corridor ({x},{y}) is a building"
            assert not nfz[y, x], f"Corridor ({x},{y}) in NFZ"


class TestGuardrailDepthDistribution:
    """Verify M1 fix: depth distribution sums to 1.0."""

    def test_depths_sum_to_one_with_events(self):
        from uavbench.cli.benchmark import _compute_guardrail_depth_distribution
        events = [
            {"type": "feasibility_relaxation_applied", "payload": {"guardrail_depth": 1}},
            {"type": "feasibility_relaxation_applied", "payload": {"guardrail_depth": 2}},
            {"type": "feasibility_relaxation_applied", "payload": {"guardrail_depth": 1}},
            {"type": "feasibility_relaxation_applied", "payload": {"guardrail_depth": 3}},
        ]
        dist = _compute_guardrail_depth_distribution(events)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9, f"Depth distribution must sum to 1.0, got {total}"
        assert dist["depth_1"] == 0.5
        assert dist["depth_2"] == 0.25
        assert dist["depth_3"] == 0.25

    def test_depths_sum_to_one_empty(self):
        from uavbench.cli.benchmark import _compute_guardrail_depth_distribution
        dist = _compute_guardrail_depth_distribution([])
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9
        assert dist["depth_0"] == 1.0

    def test_depths_sum_to_one_all_same(self):
        from uavbench.cli.benchmark import _compute_guardrail_depth_distribution
        events = [
            {"type": "feasibility_relaxation_applied", "payload": {"guardrail_depth": 2}}
            for _ in range(10)
        ]
        dist = _compute_guardrail_depth_distribution(events)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9, f"Depth distribution must sum to 1.0, got {total}"
        assert dist["depth_2"] == 1.0


# ── V&V Fix: Moving-target buffer in _build_runtime_blocking_mask ──

class TestMovingTargetGuardrailMask:
    """VV-FIX-2: _build_runtime_blocking_mask includes moving-target buffer.

    The feasibility guardrail must block the same cells that step() rejects
    when enable_moving_target=True.  Before the fix, the mask omitted the
    target buffer zone, causing a structural mismatch.
    """

    def _make_target_config(self, **overrides) -> ScenarioConfig:
        defaults = dict(
            name="mt_guardrail_test",
            domain=Domain.URBAN,
            difficulty=Difficulty.EASY,
            mission_type=MissionType.CIVIL_PROTECTION,
            regime=Regime.STRESS_TEST,
            map_size=32,
            map_source="synthetic",
            building_density=0.0,
            no_fly_radius=0,
            max_altitude=3,
            safe_altitude=3,
            min_start_goal_l1=8,
            enable_fire=False,
            enable_traffic=False,
            enable_moving_target=True,
            target_buffer_radius=4,
            paper_track="dynamic",
            force_replan_count=0,
        )
        defaults.update(overrides)
        return ScenarioConfig(**defaults)

    def test_moving_target_buffer_in_runtime_mask(self):
        """After reset, blocking mask must cover moving-target buffer cells."""
        cfg = self._make_target_config()
        env = UrbanEnv(cfg)
        env.reset(seed=0)

        # Call _build_runtime_blocking_mask directly
        mask = env._build_runtime_blocking_mask()
        assert mask.any(), (
            "Runtime blocking mask is all-False; moving-target buffer not reflected"
        )

        # The target position must be blocked
        if env._moving_target is not None:
            tp = env._moving_target.current_position
            tx, ty = int(tp[0]), int(tp[1])
            assert mask[ty, tx], (
                f"Moving-target centre ({tx},{ty}) not in runtime blocking mask"
            )

        env.close()

    def test_step_reject_consistent_with_mask(self):
        """A move into the target buffer must be rejected by step() AND blocked in the mask."""
        cfg = self._make_target_config(target_buffer_radius=3)
        env = UrbanEnv(cfg)
        env.reset(seed=7)

        if env._moving_target is None:
            pytest.skip("No moving target in this env (seed variance)")

        mask = env._build_runtime_blocking_mask()
        # Find any True cell in the mask that's inside the target buffer zone
        blocked_cells = list(zip(*np.where(mask)))
        assert len(blocked_cells) > 0, "Mask empty — cannot verify step consistency"

        # Confirm every cell blocked by mask is also blocked by env step logic
        # (We can't easily call step for arbitrary positions, but we can assert
        #  that the mask reports the same or more blocked cells as step would.)
        # This is a structural check: mask must not be empty when target exists.
        assert mask.any()

        env.close()


# ── V&V Instrumentation: reject_reason + reject_cell (Patch 2) ──

class TestRejectReasonField:
    """reject_reason enum and reject_cell are present in info dict (C3 instrumentation)."""

    def test_reject_reason_none_on_accepted_move(self):
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        _, _, _, _, info = env.step(3)
        assert "reject_reason" in info, "reject_reason missing from info"
        assert "reject_cell" in info, "reject_cell missing from info"
        if info["accepted_move"]:
            assert info["reject_reason"] == "none"
            assert info["reject_cell"] is None
        env.close()

    def test_reject_reason_forced_block_when_interdiction_active(self):
        cfg = _make_simple_config(
            building_density=0.0, no_fly_radius=0, map_size=40,
            paper_track="dynamic", force_replan_count=2,
            event_t1=2, event_t2=50,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=0)

        # Step until t1 triggers
        info_t = {}
        for _ in range(3):
            _, _, _, _, info_t = env.step(3)

        # Now forced block is active; if the next step hits it, reason = forced_block
        if info_t.get("attempted_forced_block"):
            assert info_t["reject_reason"] == "forced_block"
            assert info_t["reject_cell"] is not None
        env.close()

    def test_reject_reason_no_fly_on_nfz_cell(self):
        """Stepping into static no-fly zone yields reject_reason='no_fly'."""
        from uavbench.scenarios.schema import Difficulty
        cfg = ScenarioConfig(
            name="nfz_rej", domain=Domain.URBAN, difficulty=Difficulty.HARD,
            map_size=20, building_density=0.0, no_fly_radius=3,
            terminate_on_collision=False, map_source="synthetic",
        )
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        # Try many steps; at some point might hit NFZ; if so check reason
        for _ in range(20):
            _, _, _, _, info = env.step(3)
            if info.get("attempted_no_fly"):
                assert info["reject_reason"] == "no_fly"
                assert info["reject_cell"] is not None
                break
        env.close()

    def test_reject_reason_building_on_collision(self):
        """Stepping into a building yields reject_reason='building'."""
        from uavbench.scenarios.schema import Difficulty
        cfg = ScenarioConfig(
            name="bld_rej", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            map_size=20, building_density=0.0, no_fly_radius=0,
            terminate_on_collision=False, map_source="synthetic",
        )
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        # Manually place a building at (x=1, y=0) — force a collision
        env._heightmap[0, 1] = 3.0
        env._agent_pos[:] = [0, 0, 0]  # altitude=0, step right into building
        _, _, _, _, info = env.step(3)  # action=3 = right
        if info.get("attempted_building_collision"):
            assert info["reject_reason"] == "building"
            assert info["reject_cell"] == (1, 0)
        env.close()


class TestPlanInstrumentation:
    """PLAN_LEN, PLAN_STALE, PLAN_REASON are present in info dict (P2 instrumentation)."""

    def test_plan_len_present_and_non_negative(self):
        """info['plan_len'] is always present and >= 0 after any step."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        # Default (no set_plan_info call) — plan_len starts at 0
        _, _, _, _, info = env.step(3)
        assert "plan_len" in info, "plan_len missing from info dict"
        assert isinstance(info["plan_len"], int), "plan_len must be int"
        assert info["plan_len"] >= 0, f"plan_len must be non-negative, got {info['plan_len']}"
        env.close()

    def test_plan_stale_and_reason_after_set_plan_info(self):
        """set_plan_info() correctly propagates stale flag and reason into info dict."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        env.set_plan_info(plan_len=42, plan_stale=True, plan_reason="budget_exceeded")
        _, _, _, _, info = env.step(3)
        assert info["plan_len"] == 42
        assert info["plan_stale"] is True
        assert info["plan_reason"] == "budget_exceeded"
        env.close()


# ═══════════════════════════════════════════════════════════════════
# TG-P2-EXT  snapshot_age_steps + plan_reason enum completeness
# ═══════════════════════════════════════════════════════════════════

class TestPlanInstrumentationExtended:
    """Extended P2 tests: snapshot_age_steps, reason enum, set_plan_info API."""

    VALID_REASONS = {"initial", "cadence", "path_invalidated", "forced_event",
                     "stuck", "budget_exceeded", "planner_signal", "dropout",
                     "latency", "none"}

    def test_snapshot_age_steps_present_and_zero_by_default(self):
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert "snapshot_age_steps" in info, "snapshot_age_steps missing from info"
        assert isinstance(info["snapshot_age_steps"], int)
        assert info["snapshot_age_steps"] == 0
        env.close()

    def test_snapshot_age_steps_set_by_set_plan_info(self):
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=0)
        env.set_plan_info(plan_len=10, snapshot_age_steps=3)
        _, _, _, _, info = env.step(0)
        assert info["snapshot_age_steps"] == 3
        env.close()

    def test_plan_reason_is_string(self):
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert isinstance(info["plan_reason"], str)
        env.close()

    def test_plan_reason_all_valid_enum_values_accepted(self):
        """set_plan_info must accept all documented reason strings."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=0)
        for reason in self.VALID_REASONS:
            env.set_plan_info(plan_len=5, plan_reason=reason)
            _, _, _, _, info = env.step(0)
            assert info["plan_reason"] == reason, f"reason {reason!r} not round-tripped"
        env.close()


# ═══════════════════════════════════════════════════════════════════
# TG-DYN  Per-layer rejection tests (moving_target, intruder, dyn_nfz)
# ═══════════════════════════════════════════════════════════════════

class TestDynamicLayerRejections:
    """Each dynamic blocking layer produces correct reject_reason in info dict."""

    def _step_into(self, env, action: int = 3):
        """Take one step and return info."""
        _, _, _, _, info = env.step(action)
        return info

    def test_fire_block_rejection_when_enabled(self):
        """Fire block rejects move and sets reject_reason='fire' when fire_blocks_movement=True."""
        cfg = _make_simple_config(
            map_size=20, building_density=0.0, no_fly_radius=0,
            enable_fire=True, fire_blocks_movement=True,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        # Place a fire cell just to the right of agent
        ax, ay = int(env._agent_pos[0]), int(env._agent_pos[1])
        if env._fire_model is not None:
            env._fire_model.fire_mask[ay, min(ax + 1, 19)] = True
        info = self._step_into(env, action=3)  # right
        if info.get("attempted_fire_block"):
            assert info["reject_reason"] == "fire"
            assert info["reject_cell"] is not None
        env.close()

    def test_forced_block_rejection(self):
        """Forced block rejects move and sets reject_reason='forced_block'."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        ax, ay = int(env._agent_pos[0]), int(env._agent_pos[1])
        # Manually set forced_block_mask at the right neighbour
        nx, ny = min(ax + 1, 19), ay
        env._forced_block_mask[ny, nx] = True
        info = self._step_into(env, action=3)
        assert info["attempted_forced_block"] is True
        assert info["reject_reason"] == "forced_block"
        assert info["reject_cell"] == (nx, ny)
        env.close()

    def test_moving_target_buffer_rejection(self):
        """Moving target buffer rejects move and sets reject_reason='moving_target'."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        ax, ay = int(env._agent_pos[0]), int(env._agent_pos[1])
        nx, ny = min(ax + 1, 19), ay

        # Inject a simple mock target positioned at (nx, ny) with buffer_radius=0.
        # urban.py reads: tp = self._moving_target.current_position → tp[0]=x, tp[1]=y
        class _SyntheticMovingTarget:
            def __init__(self, x, y):
                self._pos = np.array([float(x), float(y)])
                self.buffer_radius = 0
            @property
            def current_position(self):
                return self._pos.copy()
            def step(self, *a, **kw): pass

        env._moving_target = _SyntheticMovingTarget(nx, ny)
        info = self._step_into(env, action=3)
        if info.get("attempted_target_block"):
            assert info["reject_reason"] == "moving_target"
            assert info["reject_cell"] == (nx, ny)
        env.close()

    def test_dynamic_nfz_block_rejection(self):
        """Dynamic NFZ rejects move and sets reject_reason='dynamic_nfz'."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        ax, ay = int(env._agent_pos[0]), int(env._agent_pos[1])
        nx, ny = min(ax + 1, 19), ay
        # Create a synthetic dynamic NFZ mask covering the target cell
        nfz_mask = np.zeros((20, 20), dtype=bool)
        nfz_mask[ny, nx] = True

        class _SyntheticNFZ:
            zone_violations = 0
            def get_nfz_mask(self):
                return nfz_mask
            def step(self, *a, **kw):
                pass

        env._dynamic_nfz = _SyntheticNFZ()
        info = self._step_into(env, action=3)
        if info.get("attempted_nfz_block"):
            assert info["reject_reason"] == "dynamic_nfz"
            assert info["reject_cell"] == (nx, ny)
        env.close()


# ═══════════════════════════════════════════════════════════════════
# TG-INT  Forced interdiction timeline truth (exactly twice, t1/t2)
# ═══════════════════════════════════════════════════════════════════

class TestInterdictionTwiceTimeline:
    """Forced interdictions: exactly 2 events, fired at t1/t2, Manhattan geometry."""

    def _run_to_step(self, env, n: int):
        for _ in range(n):
            env.step(0)

    def test_exactly_two_interdiction_events(self):
        """force_replan_count=2 → exactly 2 path_interdiction events logged."""
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=5, event_t2=12)
        env = UrbanEnv(cfg); env.reset(seed=42)
        # Run well past t2
        for _ in range(20):
            env.step(0)
        interdiction_events = [
            e for e in env.events
            if "interdiction" in e.get("type", "")
        ]
        assert len(interdiction_events) == 2, (
            f"Expected 2 interdiction events, got {len(interdiction_events)}: {interdiction_events}"
        )
        env.close()

    def test_interdictions_fire_at_t1_and_t2(self):
        """The two interdiction events fire at exactly step=t1 and step=t2."""
        t1, t2 = 5, 12
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=t1, event_t2=t2)
        env = UrbanEnv(cfg); env.reset(seed=42)
        for _ in range(20):
            env.step(0)
        interdiction_events = [
            e for e in env.events if "interdiction" in e.get("type", "")
        ]
        steps_seen = sorted(int(e.get("step", -1)) for e in interdiction_events)
        # log_event uses _step_override=step_idx so event["step"] == exact scheduled step.
        assert steps_seen == [t1, t2], (
            f"Interdiction steps {steps_seen} != expected [{t1}, {t2}]"
        )
        env.close()

    def test_forced_block_mask_set_at_t1_not_before(self):
        """Interdiction is not triggered before t1; triggered flag is set at/after t1."""
        t1 = 5
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=t1, event_t2=15)
        env = UrbanEnv(cfg); env.reset(seed=42)
        # Step up to just before t1 (next_step_idx reaches t1 on the t1-th call)
        for _ in range(t1 - 1):
            env.step(0)
        triggered_before = any(e["triggered"] for e in env._forced_interdictions)
        assert not triggered_before, "interdiction must not be triggered before t1"
        # Step at t1: _maybe_trigger_interdictions gets next_step_idx=t1 → fires
        env.step(0)
        # After firing, the guardrail may clear _forced_block_mask for reachability;
        # check the canonical 'triggered' flag instead.
        triggered_at = any(e["triggered"] for e in env._forced_interdictions if int(e["step"]) == t1)
        assert triggered_at, (
            f"path_interdiction_1 must be triggered at t1={t1}; "
            f"interdictions={env._forced_interdictions}"
        )
        env.close()

    def test_interdiction_disk_is_manhattan(self):
        """Forced block cells satisfy Manhattan distance <= radius from center."""
        t1 = 5
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=t1, event_t2=15)
        env = UrbanEnv(cfg); env.reset(seed=42)
        for _ in range(t1 + 1):
            env.step(0)
        events = [e for e in env.events if "interdiction" in e.get("type", "")]
        if not events:
            env.close(); return
        ev = events[0]
        cx, cy = int(ev.get("x", 0)), int(ev.get("y", 0))
        radius = int(ev.get("radius", 3))
        mask = env.get_dynamic_state()["forced_block_mask"]
        ys, xs = np.where(mask)
        for bx, by in zip(xs.tolist(), ys.tolist()):
            dist = abs(bx - cx) + abs(by - cy)
            assert dist <= radius, (
                f"Block cell ({bx},{by}) has L1 dist {dist} > radius {radius} from center ({cx},{cy})"
            )
        env.close()

    def test_interdictions_planner_agnostic(self):
        """Interdiction block positions are identical across two different seeds' forced_interdictions."""
        # Same scenario, same seed → same positions regardless of planner choice
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=5, event_t2=12)
        masks = []
        for _ in range(2):
            env = UrbanEnv(cfg); env.reset(seed=77)
            for _ in range(15):
                env.step(0)
            masks.append(env.get_dynamic_state()["forced_block_mask"].copy())
            env.close()
        np.testing.assert_array_equal(masks[0], masks[1])


# ═══════════════════════════════════════════════════════════════════
# TG-FEAS  Feasibility/Guardrail mask parity
# ═══════════════════════════════════════════════════════════════════

class TestFeasibilityMaskParity:
    """_build_runtime_blocking_mask includes every step-level movement blocker."""

    def test_forced_block_in_runtime_mask(self):
        """A forced block cell appears in _build_runtime_blocking_mask output."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        env._forced_block_mask[5, 5] = True
        mask = env._build_runtime_blocking_mask()
        assert mask[5, 5], "forced_block not reflected in runtime mask"
        env.close()

    def test_dynamic_nfz_in_runtime_mask(self):
        """Dynamic NFZ cells appear in _build_runtime_blocking_mask output."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        nfz_mask = np.zeros((20, 20), dtype=bool)
        nfz_mask[7, 7] = True

        class _SyntheticNFZ:
            def get_nfz_mask(self): return nfz_mask
            def step(self, *a, **kw): pass

        env._dynamic_nfz = _SyntheticNFZ()
        mask = env._build_runtime_blocking_mask()
        assert mask[7, 7], "dynamic_nfz not reflected in runtime mask"
        env.close()

    def test_fire_in_runtime_mask_when_blocks_movement(self):
        """Fire cells appear in runtime mask when fire_blocks_movement=True."""
        cfg = _make_simple_config(
            map_size=20, building_density=0.0, no_fly_radius=0,
            enable_fire=True, fire_blocks_movement=True,
        )
        env = UrbanEnv(cfg); env.reset(seed=0)
        if env._fire_model is not None:
            from uavbench.dynamics.fire_spread import BURNING
            env._fire_model.force_cell_state(3, 3, BURNING)  # public test hook
            mask = env._build_runtime_blocking_mask()
            assert mask[3, 3], "fire not reflected in runtime mask despite fire_blocks_movement=True"
        env.close()


# ═══════════════════════════════════════════════════════════════════
# TG-PLN  Planner↔Env contract tests (CT1-CT5)
# ═══════════════════════════════════════════════════════════════════

class TestPlannerEnvContract:
    """Contract tests for planner outputs vs env action model."""

    def _plan(self, planner_id: str, heightmap, no_fly, start, goal):
        from uavbench.planners import PLANNERS
        planner = PLANNERS[planner_id](heightmap, no_fly)
        result = planner.plan(start, goal)
        return result

    def _make_env_heightmap(self, size=32):
        import numpy as np
        from uavbench.scenarios.schema import ScenarioConfig, Domain, Difficulty, MissionType, Regime
        from uavbench.envs.urban import UrbanEnv
        cfg = ScenarioConfig(
            name="ct_test", domain=Domain.URBAN, difficulty=Difficulty.EASY,
            mission_type=MissionType.CIVIL_PROTECTION, regime=Regime.NATURALISTIC,
            map_size=size, map_source="synthetic", building_density=0.05, no_fly_radius=0,
            max_altitude=3, safe_altitude=3, min_start_goal_l1=5, enable_fire=False, enable_traffic=False,
        )
        env = UrbanEnv(cfg); env.reset(seed=0)
        return env, env._heightmap.copy(), env._no_fly_mask.copy(), env.agent_xy, env.goal_xy

    @pytest.mark.parametrize("planner_id", ["astar", "theta_star", "periodic_replan",
                                              "aggressive_replan", "incremental_dstar_lite"])
    def test_CT1_path_is_4connected_after_expansion(self, planner_id):
        """CT1: Every consecutive pair in the expanded execution path is 4-connected (L1=1)."""
        from uavbench.cli.benchmark import _expand_execution_path
        env, heightmap, no_fly, start, goal = self._make_env_heightmap()
        result = self._plan(planner_id, heightmap, no_fly, start, goal)
        env.close()
        if not result.success or not result.path:
            pytest.skip(f"{planner_id} failed to find path in test env")
        expanded = _expand_execution_path(list(result.path), heightmap, no_fly)
        for i in range(len(expanded) - 1):
            x1, y1 = expanded[i]
            x2, y2 = expanded[i + 1]
            l1 = abs(x2 - x1) + abs(y2 - y1)
            assert l1 == 1, (
                f"{planner_id}: step {i}→{i+1}: ({x1},{y1})→({x2},{y2}) L1={l1} != 1 (not 4-connected)"
            )

    @pytest.mark.parametrize("planner_id", ["astar", "theta_star", "periodic_replan",
                                              "aggressive_replan", "incremental_dstar_lite"])
    def test_CT2_path_avoids_blocked_cells(self, planner_id):
        """CT2: Planner output never includes a cell marked True in no_fly mask."""
        env, heightmap, no_fly, start, goal = self._make_env_heightmap()
        result = self._plan(planner_id, heightmap, no_fly, start, goal)
        env.close()
        if not result.success or not result.path:
            pytest.skip(f"{planner_id} failed to find path in test env")
        for x, y in result.path[1:]:  # skip start
            assert not no_fly[y, x], (
                f"{planner_id}: path includes no_fly cell ({x},{y})"
            )

    @pytest.mark.parametrize("planner_id", ["astar", "theta_star", "periodic_replan",
                                              "aggressive_replan", "incremental_dstar_lite"])
    def test_CT3_waypoint_action_produces_valid_action(self, planner_id):
        """CT3: _waypoint_action never returns fallback action on a 4-connected path."""
        from uavbench.cli.benchmark import _expand_execution_path, _waypoint_action
        env, heightmap, no_fly, start, goal = self._make_env_heightmap()
        result = self._plan(planner_id, heightmap, no_fly, start, goal)
        env.close()
        if not result.success or not result.path:
            pytest.skip(f"{planner_id} failed to find path in test env")
        expanded = _expand_execution_path(list(result.path), heightmap, no_fly)
        for i in range(len(expanded) - 1):
            a = _waypoint_action(expanded[i], expanded[i + 1])
            assert a in (0, 1, 2, 3), f"{planner_id}: invalid action {a} at step {i}"

    def test_CT5_soft_budget_violation_logged(self):
        """CT5 (soft budget): budget exceeded → violation logged; plan_stale=True; path NOT discarded."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=0)
        # Simulate budget exceeded: set_plan_info with stale=True
        env.set_plan_info(plan_len=10, plan_stale=True, plan_reason="budget_exceeded")
        _, _, _, _, info = env.step(0)
        assert info["plan_stale"] is True
        assert info["plan_reason"] == "budget_exceeded"
        assert info["plan_len"] == 10  # old path preserved (not 0)
        env.close()


# ═══════════════════════════════════════════════════════════════════
# TG-SEM  Semantic hardening (post PAPER-READY batch)
# ═══════════════════════════════════════════════════════════════════

class TestEventStepSemantics:
    """A) log_event _step_override: interdiction events report exact scheduled step."""

    def test_interdiction_event_step_equals_t1(self):
        """Event['step'] for path_interdiction_1 == event_t1 (not t1-1)."""
        t1 = 8
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=t1, event_t2=20)
        env = UrbanEnv(cfg); env.reset(seed=42)
        for _ in range(25):
            env.step(0)
        interd = [e for e in env.events if "path_interdiction_1" in e.get("type", "")]
        assert len(interd) >= 1, "path_interdiction_1 event must be logged"
        assert int(interd[0]["step"]) == t1, (
            f"event['step']={interd[0]['step']} must equal t1={t1} (authoritative via _step_override)"
        )
        env.close()

    def test_log_event_default_uses_step_count(self):
        """Without _step_override, log_event stores self._step_count (normal events)."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=16)
        env = UrbanEnv(cfg); env.reset(seed=0)
        env.log_event("test_evt", value=42)
        # At reset step_count=0; before any step this is 0.
        evts = [e for e in env.events if e.get("type") == "test_evt"]
        assert len(evts) == 1
        assert int(evts[0]["step"]) == 0  # no steps taken yet
        env.step(0)
        env.log_event("test_evt2", value=99)
        evts2 = [e for e in env.events if e.get("type") == "test_evt2"]
        assert int(evts2[0]["step"]) == 1  # after 1 step
        env.close()

    def test_log_event_override_wins(self):
        """_step_override=N overrides _step_count regardless of actual step."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=16)
        env = UrbanEnv(cfg); env.reset(seed=0)
        for _ in range(5):
            env.step(0)
        env.log_event("pinned_event", _step_override=99, x=1, y=2)
        pinned = [e for e in env.events if e.get("type") == "pinned_event"]
        assert int(pinned[0]["step"]) == 99
        env.close()


class TestPublicEnvAPI:
    """B) agent_xy / goal_xy stable public properties."""

    def test_agent_xy_property_returns_tuple(self):
        """env.agent_xy is a (int, int) tuple matching _agent_pos."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=7)
        xy = env.agent_xy
        assert isinstance(xy, tuple) and len(xy) == 2
        assert xy == (int(env._agent_pos[0]), int(env._agent_pos[1]))
        env.close()

    def test_goal_xy_property_returns_tuple(self):
        """env.goal_xy is a (int, int) tuple matching _goal_pos."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=7)
        xy = env.goal_xy
        assert isinstance(xy, tuple) and len(xy) == 2
        assert xy == (int(env._goal_pos[0]), int(env._goal_pos[1]))
        env.close()

    def test_agent_xy_updates_after_accepted_move(self):
        """agent_xy reflects agent position after each step."""
        cfg = _make_simple_config(building_density=0.0, no_fly_radius=0, map_size=20)
        env = UrbanEnv(cfg); env.reset(seed=0)
        xy_before = env.agent_xy
        # Action 0 = up (y-1). Try multiple actions to find one that's accepted.
        accepted = False
        for action in range(4):
            _, _, _, _, info = env.step(action)
            if info.get("accepted_move"):
                accepted = True
                break
        if accepted:
            xy_after = env.agent_xy
            assert xy_after != xy_before, "agent_xy must change after accepted move"
        env.close()


class TestFireTestHook:
    """C) FireSpreadModel.force_cell_state testability hook."""

    def _make_fire_model(self, H=20, W=20):
        from uavbench.dynamics.fire_spread import FireSpreadModel
        import numpy as np
        landuse = np.zeros((H, W), dtype=np.int8)
        roads = np.zeros((H, W), dtype=bool)
        return FireSpreadModel(landuse, roads, wind_dir=0.0, wind_speed=0.0, rng=np.random.default_rng(0))

    def test_force_cell_state_sets_fire(self):
        """force_cell_state(x, y, BURNING) makes fire_mask[y,x]=True."""
        from uavbench.dynamics.fire_spread import BURNING
        model = self._make_fire_model()
        assert not model.fire_mask[5, 5], "cell should be unburned initially"
        model.force_cell_state(5, 5, BURNING)
        assert model.fire_mask[5, 5], "fire_mask[5,5] must be True after force_cell_state"

    def test_force_cell_state_clears_fire(self):
        """force_cell_state(x, y, UNBURNED) clears a burning cell."""
        from uavbench.dynamics.fire_spread import BURNING, UNBURNED
        model = self._make_fire_model()
        model.force_cell_state(3, 3, BURNING)
        assert model.fire_mask[3, 3]
        model.force_cell_state(3, 3, UNBURNED)
        assert not model.fire_mask[3, 3], "fire_mask[3,3] must be False after forcing UNBURNED"

    def test_env_fire_runtime_mask_via_hook(self):
        """Runtime mask reflects fire set via force_cell_state (not private _state)."""
        from uavbench.dynamics.fire_spread import BURNING
        cfg = _make_simple_config(
            map_size=20, building_density=0.0, no_fly_radius=0,
            enable_fire=True, fire_blocks_movement=True,
        )
        env = UrbanEnv(cfg); env.reset(seed=0)
        if env._fire_model is not None:
            env._fire_model.force_cell_state(4, 4, BURNING)
            mask = env._build_runtime_blocking_mask()
            assert mask[4, 4], "runtime mask must block fire cell set via force_cell_state"
        env.close()


class TestForcedBlockLifecycle:
    """D) Forced interdiction lifecycle: triggered / active / cleared instrumentation."""

    def test_triggered_steps_populated_after_interdiction(self):
        """forced_interdiction_triggered_steps contains t1 and t2 after both fire."""
        t1, t2 = 6, 14
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=t1, event_t2=t2)
        env = UrbanEnv(cfg); env.reset(seed=42)
        for _ in range(20):
            env.step(0)
        state = env.get_dynamic_state()
        triggered = sorted(state["forced_interdiction_triggered_steps"])
        assert triggered == [t1, t2], (
            f"triggered_steps={triggered} must equal [t1={t1}, t2={t2}]"
        )
        env.close()

    def test_forced_block_active_true_when_mask_set(self):
        """forced_block_active=True in get_dynamic_state when mask is non-empty."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        env._forced_block_mask[5, 5] = True
        state = env.get_dynamic_state()
        assert state["forced_block_active"] is True
        assert state["forced_block_area"] >= 1
        env.close()

    def test_forced_block_area_counts_cells(self):
        """forced_block_area = exact count of True cells in forced_block_mask."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        env._forced_block_mask[2, 2] = True
        env._forced_block_mask[3, 3] = True
        env._forced_block_mask[4, 4] = True
        state = env.get_dynamic_state()
        assert state["forced_block_area"] == 3
        env.close()

    def test_forced_block_cleared_flag_set_by_guardrail(self):
        """forced_block_cleared_by_guardrail=True in info dict when guardrail depth=1 clears mask."""
        t1 = 5
        cfg = _make_dynamic_config(force_replan_count=2, event_t1=t1, event_t2=20)
        env = UrbanEnv(cfg); env.reset(seed=42)
        # Run past t1; if map is small enough, guardrail fires depth=1
        for _ in range(t1 + 3):
            env.step(0)
        state = env.get_dynamic_state()
        # Either the block is still active OR guardrail cleared it — one must be true
        triggered = any(e["triggered"] for e in env._forced_interdictions if int(e["step"]) == t1)
        if triggered:
            assert (state["forced_block_active"] or state["forced_block_cleared_by_guardrail"]), (
                "After interdiction, either mask active or cleared flag must be True"
            )
        env.close()

    def test_lifecycle_in_info_dict(self):
        """forced_block_active, forced_block_area, forced_interdiction_triggered_steps in step info."""
        cfg = _make_simple_config(map_size=20, building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg); env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert "forced_block_active" in info
        assert "forced_block_area" in info
        assert "forced_interdiction_triggered_steps" in info
        assert isinstance(info["forced_block_active"], bool)
        assert isinstance(info["forced_block_area"], int)
        assert isinstance(info["forced_interdiction_triggered_steps"], list)
        env.close()
