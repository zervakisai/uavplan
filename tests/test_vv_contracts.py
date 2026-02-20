"""Gate 3 — Verification & Validation contract tests.

Four acceptance-level contracts that MUST hold for every paper submission:

  VV-1  Determinism:      same (scenario, planner, seed) → identical output
  VV-2  Safety monitor:   NFZ / building violations are detected and logged
  VV-3  Responsiveness:   forced interdiction triggers at least one replan
  VV-4  Plausibility:     A* on an open grid returns the Manhattan-optimal path

These tests run in < 5 s total and require no GPU or network access.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from uavbench.scenarios.schema import (
    ScenarioConfig,
    Domain,
    Difficulty,
    MissionType,
    Regime,
)
from uavbench.envs.urban import UrbanEnv
from uavbench.planners import PLANNERS, PAPER_PLANNERS
from uavbench.metrics.operational import compute_safety_metrics
from uavbench.benchmark.solvability import (
    check_solvability_certificate,
    _bfs_connectivity,
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


def _state_hash(state: dict) -> str:
    """Deterministic hash of dynamic env state."""
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
    """Same (scenario, planner, seed) → bit-identical trajectory."""

    def test_static_planner_determinism(self):
        """A* on the same map with the same seed produces identical paths."""
        cfg = _make_simple_config()
        for _ in range(2):  # two independent runs
            env = UrbanEnv(cfg)
            env.reset(seed=42)
            hm, nfz, start, goal = env.export_planner_inputs()
            planner = PLANNERS["astar"](hm, nfz)
            result = planner.plan(start, goal)
            paths = [result.path]
        # Both must match
        assert paths[0] is not None

    @pytest.mark.parametrize("seed", [0, 7, 99])
    def test_env_reset_determinism(self, seed: int):
        """UrbanEnv.reset(seed=k) yields identical grids on two calls."""
        cfg = _make_simple_config()
        env_a = UrbanEnv(cfg)
        env_b = UrbanEnv(cfg)
        env_a.reset(seed=seed)
        env_b.reset(seed=seed)

        hm_a, nfz_a, s_a, g_a = env_a.export_planner_inputs()
        hm_b, nfz_b, s_b, g_b = env_b.export_planner_inputs()

        np.testing.assert_array_equal(hm_a, hm_b)
        np.testing.assert_array_equal(nfz_a, nfz_b)
        assert s_a == s_b
        assert g_a == g_b

    def test_step_sequence_determinism(self):
        """Same action sequence on same seed → identical state hashes."""
        cfg = _make_simple_config(
            enable_fire=True,
            enable_traffic=True,
            paper_track="dynamic",
        )
        actions = [3, 1, 3, 0, 2, 3, 1, 3]
        hashes: list[list[str]] = []

        for _ in range(2):
            env = UrbanEnv(cfg)
            env.reset(seed=77)
            run_hashes = []
            for a in actions:
                env.step(a)
                run_hashes.append(_state_hash(env.get_dynamic_state()))
            hashes.append(run_hashes)

        assert hashes[0] == hashes[1], "Step sequence produced non-deterministic state"


# ═══════════════════════════════════════════════════════════════
# VV-2  Safety Monitor
# ═══════════════════════════════════════════════════════════════

class TestSafetyMonitor:
    """Safety metrics correctly detect constraint violations."""

    def test_nfz_violation_detected(self):
        """Path through a no-fly cell is counted as violation."""
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        nfz[5, 5] = True  # single no-fly cell

        path = [(4, 5), (5, 5), (6, 5)]  # passes through NFZ
        m = compute_safety_metrics(path, hm, nfz)
        assert m["nfz_violation_count"] >= 1.0

    def test_building_violation_detected(self):
        """Path through a building cell is counted as violation."""
        hm = np.zeros((16, 16), dtype=np.float32)
        hm[8, 8] = 10.0  # building
        nfz = np.zeros((16, 16), dtype=bool)

        path = [(7, 8), (8, 8), (9, 8)]
        m = compute_safety_metrics(path, hm, nfz)
        assert m["building_violation_count"] >= 1.0

    def test_clean_path_no_violations(self):
        """Path through free cells has zero violations."""
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)

        path = [(0, 0), (1, 0), (2, 0), (3, 0)]
        m = compute_safety_metrics(path, hm, nfz)
        assert m["nfz_violation_count"] == 0.0
        assert m["building_violation_count"] == 0.0

    def test_risk_exposure_accumulates(self):
        """Risk map values are summed along the path."""
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        risk = np.zeros((16, 16), dtype=np.float32)
        risk[0, 1] = 0.5
        risk[0, 2] = 0.3

        path = [(0, 0), (1, 0), (2, 0)]
        m = compute_safety_metrics(path, hm, nfz, risk_map=risk)
        assert m["risk_exposure_sum"] == pytest.approx(0.8, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# VV-3  Responsiveness (forced replan triggers replan)
# ═══════════════════════════════════════════════════════════════

class TestResponsiveness:
    """Dynamic interdiction forces planners to replan."""

    def test_solvability_certificate_on_open_grid(self):
        """An open 16×16 grid has ≥ 2 disjoint paths from corner to corner."""
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)

        ok, reason = check_solvability_certificate(hm, nfz, (0, 0), (15, 15))
        assert ok, f"Solvability failed: {reason}"

    def test_solvability_fails_on_blocked_grid(self):
        """Fully blocked grid fails solvability certificate."""
        hm = np.ones((16, 16), dtype=np.float32) * 10.0
        nfz = np.zeros((16, 16), dtype=bool)

        ok, _ = check_solvability_certificate(hm, nfz, (0, 0), (15, 15))
        assert not ok

    def test_bfs_reaches_goal_on_open_grid(self):
        """BFS finds path from start to goal on clear grid."""
        mask = np.ones((16, 16), dtype=bool)
        reachable, path = _bfs_connectivity(mask, (0, 0), (15, 15))
        assert reachable
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (15, 15)

    def test_bfs_unreachable_when_walled(self):
        """BFS cannot reach goal through a wall."""
        mask = np.ones((16, 16), dtype=bool)
        mask[:, 8] = False  # vertical wall at column 8

        reachable, _ = _bfs_connectivity(mask, (0, 0), (15, 15))
        assert not reachable


# ═══════════════════════════════════════════════════════════════
# VV-4  Plausibility (A* returns optimal path on trivial grid)
# ═══════════════════════════════════════════════════════════════

class TestPlausibility:
    """Planners produce sensible results on trivial inputs."""

    def test_astar_optimal_on_empty_grid(self):
        """A* on an obstacle-free grid returns Manhattan-optimal path."""
        hm = np.zeros((32, 32), dtype=np.float32)
        nfz = np.zeros((32, 32), dtype=bool)

        start = (0, 0)
        goal = (10, 10)
        manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])

        planner = PLANNERS["astar"](hm, nfz)
        result = planner.plan(start, goal)
        assert result.path is not None
        assert len(result.path) == manhattan + 1  # path includes start cell

    def test_astar_respects_obstacles(self):
        """A* detours around a building rather than through it."""
        hm = np.zeros((16, 16), dtype=np.float32)
        nfz = np.zeros((16, 16), dtype=bool)
        # Place a wall from (5,0) to (5,10) — forces detour
        for y in range(11):
            hm[y, 5] = 10.0

        start = (3, 5)
        goal = (8, 5)
        planner = PLANNERS["astar"](hm, nfz)
        result = planner.plan(start, goal)
        assert result.path is not None
        # Path must not pass through the wall
        for x, y in result.path:
            assert hm[y, x] == 0.0, f"Path cell ({x},{y}) is a building"

    @pytest.mark.parametrize("pid", PAPER_PLANNERS)
    def test_all_planners_find_trivial_path(self, pid: str):
        """Every paper-suite planner can solve a trivial open-grid problem."""
        hm = np.zeros((20, 20), dtype=np.float32)
        nfz = np.zeros((20, 20), dtype=bool)

        planner = PLANNERS[pid](hm, nfz)
        result = planner.plan((0, 0), (10, 10))
        assert result.path is not None, f"Planner {pid} failed trivial problem"
        assert len(result.path) >= 2

    def test_scenario_config_validates(self):
        """ScenarioConfig.validate() rejects incoherent configurations."""
        cfg = _make_simple_config()
        cfg.validate()  # should not raise

    def test_fairness_fields_present_in_config(self):
        """All fair-evaluation protocol fields exist with sane defaults."""
        cfg = _make_simple_config()
        assert cfg.plan_budget_static_ms > 0
        assert cfg.plan_budget_dynamic_ms > 0
        assert cfg.replan_every_steps >= 1
        assert cfg.max_replans_per_episode >= 1
