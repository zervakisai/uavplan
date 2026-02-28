"""Contract tests for feasibility guardrail: GC-1, GC-2.

GC-1: Relaxation D1→D2→D3→D4 is logged when applied.
GC-2: D4 (nuclear deconfliction) is the last resort.
      If still infeasible after D4 → episode flagged infeasible.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.schema import (
    ScenarioConfig, Domain, Difficulty, MissionType, Regime,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(**overrides) -> ScenarioConfig:
    """Minimal config for guardrail tests."""
    defaults = dict(
        name="gc_test",
        domain=Domain.URBAN,
        difficulty=Difficulty.MEDIUM,
        mission_type=MissionType.CIVIL_PROTECTION,
        regime=Regime.NATURALISTIC,
        map_size=20,
        map_source="synthetic",
        building_density=0.0,
        no_fly_radius=0,
        max_altitude=5,
        safe_altitude=5,
        min_start_goal_l1=4,
        enable_fire=False,
        enable_traffic=False,
        paper_track="dynamic",
        force_replan_count=2,
        event_t1=3,
        event_t2=6,
        terminate_on_collision=False,
        emergency_corridor_enabled=True,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _step_env(env: UrbanEnv, action: int = 3, n: int = 1):
    """Take n steps, return last info dict."""
    info = {}
    for _ in range(n):
        _, _, _, _, info = env.step(action)
    return info


# ── GC-1: Relaxation chain is logged ────────────────────────────────────


class TestGC1_RelaxationLogged:
    """GC-1: When the guardrail applies relaxation, it is recorded in the
    guardrail status dict with the depth reached."""

    def test_depth_0_no_relaxation_by_default(self):
        """On a clean map with no dynamics, guardrail stays at depth 0."""
        cfg = _make_config(
            force_replan_count=0,
            paper_track="static",
        )
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy
        depth = env._enforce_feasibility_guardrail(start, goal)
        assert depth == 0

    def test_forced_block_triggers_depth_1(self):
        """Blocking the path with forced blocks → guardrail clears them (D1)."""
        cfg = _make_config(force_replan_count=0, paper_track="static")
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy

        # Manually block a corridor between start and goal
        mid_x = (start[0] + goal[0]) // 2
        for y in range(env.map_size):
            env._forced_block_mask[y, mid_x] = True
        env._topology_change_counter += 1

        depth = env._enforce_feasibility_guardrail(start, goal)
        status = env._last_guardrail_status

        assert depth >= 1, "Expected at least depth 1 for forced block clearance"
        assert status["feasible_after_guardrail"] is True
        assert "forced_blocks_cleared" in status.get("relaxation_applied", {})

    def test_guardrail_status_has_required_fields(self):
        """The status dict always contains the required fields."""
        cfg = _make_config(force_replan_count=0, paper_track="static")
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy
        env._enforce_feasibility_guardrail(start, goal)
        status = env._last_guardrail_status

        required = [
            "reachability_failed_before_relax",
            "relaxation_applied",
            "corridor_fallback_used",
            "feasible_after_guardrail",
            "guardrail_depth",
        ]
        for key in required:
            assert key in status, f"Missing guardrail status key: {key}"

    def test_depth_returns_integer(self):
        """Guardrail depth is always an integer 0-4."""
        cfg = _make_config(force_replan_count=0, paper_track="static")
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy
        depth = env._enforce_feasibility_guardrail(start, goal)
        assert isinstance(depth, int)
        assert 0 <= depth <= 4


# ── GC-2: D4 nuclear deconfliction and infeasibility flag ───────────────


class TestGC2_D4NuclearDeconfliction:
    """GC-2: D4 is the last resort. If still infeasible → flagged."""

    def test_d4_clears_all_forced_blocks(self):
        """After D4, no forced blocks remain."""
        cfg = _make_config(force_replan_count=0, paper_track="static")
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy

        # Block the entire map except start/goal with forced blocks
        # This ensures D1-D3 alone may not suffice
        env._forced_block_mask[:] = True
        env._forced_block_mask[start[1], start[0]] = False
        env._forced_block_mask[goal[1], goal[0]] = False
        env._topology_change_counter += 1

        depth = env._enforce_feasibility_guardrail(start, goal)
        # D1 should clear forced blocks and succeed
        assert depth >= 1
        assert not np.any(env._forced_block_mask), (
            "After guardrail, forced blocks should be cleared"
        )

    def test_infeasible_flagged_when_truly_blocked(self):
        """If start and goal are separated by buildings (permanent obstacles),
        guardrail reaches max depth and flags infeasible."""
        cfg = _make_config(
            force_replan_count=0,
            paper_track="static",
            building_density=0.0,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy

        # Build a permanent wall of buildings separating start from goal
        wall_x = (start[0] + goal[0]) // 2
        for y in range(env.map_size):
            env._heightmap[y, wall_x] = 10  # building

        env._topology_change_counter += 1
        depth = env._enforce_feasibility_guardrail(start, goal)
        status = env._last_guardrail_status

        assert status["feasible_after_guardrail"] is False, (
            "GC-2 violated: permanent obstacle should yield infeasible"
        )
        assert depth >= 3, "Should have tried all depths before flagging infeasible"

    def test_d4_status_has_nuclear_deconfliction_flag(self):
        """When D4 is reached, the relaxation dict includes nuclear_deconfliction."""
        cfg = _make_config(
            force_replan_count=0,
            paper_track="static",
            building_density=0.0,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy

        # Build a permanent wall to force full depth escalation
        wall_x = (start[0] + goal[0]) // 2
        for y in range(env.map_size):
            env._heightmap[y, wall_x] = 10

        env._topology_change_counter += 1
        depth = env._enforce_feasibility_guardrail(start, goal)

        if depth == 4:
            status = env._last_guardrail_status
            relaxation = status.get("relaxation_applied", {})
            assert relaxation.get("nuclear_deconfliction") is True

    def test_guardrail_max_depth_is_4(self):
        """The guardrail depth should never exceed 4."""
        cfg = _make_config(force_replan_count=0, paper_track="static")
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy

        # Worst case: wall of buildings
        wall_x = (start[0] + goal[0]) // 2
        for y in range(env.map_size):
            env._heightmap[y, wall_x] = 10
        env._topology_change_counter += 1

        depth = env._enforce_feasibility_guardrail(start, goal)
        assert depth <= 4, f"Guardrail depth {depth} exceeds max of 4"

    def test_feasible_after_guardrail_is_bool(self):
        """The feasible_after_guardrail field is always a boolean."""
        cfg = _make_config(force_replan_count=0, paper_track="static")
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy
        env._enforce_feasibility_guardrail(start, goal)
        status = env._last_guardrail_status

        assert isinstance(status["feasible_after_guardrail"], bool)
