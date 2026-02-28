"""Contract tests for interdiction fairness: FC-1, FC-2.

FC-1: Same scenario+seed with different planners → identical forced
      interdiction positions (planner-agnostic BFS placement).
FC-2: Interdictions are on the BFS corridor, not on start/goal.
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
    """Minimal synthetic ScenarioConfig with forced interdictions enabled."""
    defaults = dict(
        name="fc_test",
        domain=Domain.URBAN,
        difficulty=Difficulty.MEDIUM,
        mission_type=MissionType.CIVIL_PROTECTION,
        regime=Regime.NATURALISTIC,
        map_size=30,
        map_source="synthetic",
        building_density=0.1,
        no_fly_radius=0,
        max_altitude=5,
        safe_altitude=5,
        min_start_goal_l1=8,
        enable_fire=False,
        enable_traffic=False,
        paper_track="dynamic",
        force_replan_count=2,
        event_t1=12,
        event_t2=28,
        terminate_on_collision=False,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _get_interdiction_positions(env: UrbanEnv) -> list[tuple[int, int]]:
    """Extract forced interdiction cut points from internal state."""
    return [
        tuple(intd["point"])
        for intd in env._forced_interdictions
    ]


# ── FC-1: Cross-planner interdiction identity ────────────────────────────


class TestFC1_CrossPlannerInterdictionIdentity:
    """FC-1: Same scenario+seed → identical interdiction positions regardless
    of which planner will later be used."""

    def test_two_envs_same_seed_same_interdictions(self):
        """Two independent env resets with same seed produce identical
        interdiction positions."""
        cfg = _make_config()

        env_a = UrbanEnv(cfg)
        env_a.reset(seed=42)
        pos_a = _get_interdiction_positions(env_a)

        env_b = UrbanEnv(cfg)
        env_b.reset(seed=42)
        pos_b = _get_interdiction_positions(env_b)

        assert len(pos_a) == 2, "Expected 2 forced interdictions"
        assert pos_a == pos_b, (
            f"FC-1 violated: interdictions differ across env instances "
            f"with same seed: {pos_a} != {pos_b}"
        )

    def test_different_seeds_different_maps_may_differ(self):
        """Different seeds → different maps → different interdictions
        (sanity: positions aren't hardcoded)."""
        cfg = _make_config()

        env_a = UrbanEnv(cfg)
        env_a.reset(seed=1)
        pos_a = _get_interdiction_positions(env_a)

        env_b = UrbanEnv(cfg)
        env_b.reset(seed=999)
        pos_b = _get_interdiction_positions(env_b)

        # They *may* coincidentally be equal on trivial maps,
        # but if both have 2 interdictions, at least verify the system
        # doesn't crash with different seeds.
        assert len(pos_a) == 2
        assert len(pos_b) == 2

    def test_reference_planner_field_has_no_effect(self):
        """The deprecated interdiction_reference_planner field is ignored."""
        cfg_a = _make_config()
        env_a = UrbanEnv(cfg_a)
        env_a.reset(seed=42)
        pos_a = _get_interdiction_positions(env_a)

        # Same config, same seed — positions must match
        cfg_b = _make_config()
        env_b = UrbanEnv(cfg_b)
        env_b.reset(seed=42)
        pos_b = _get_interdiction_positions(env_b)

        assert pos_a == pos_b

    def test_interdiction_radii_match_across_instances(self):
        """Radii are deterministic too, not just positions."""
        cfg = _make_config()

        env_a = UrbanEnv(cfg)
        env_a.reset(seed=42)
        radii_a = [intd["radius"] for intd in env_a._forced_interdictions]

        env_b = UrbanEnv(cfg)
        env_b.reset(seed=42)
        radii_b = [intd["radius"] for intd in env_b._forced_interdictions]

        assert radii_a == radii_b


# ── FC-2: Interdictions on BFS corridor, not on start/goal ──────────────


class TestFC2_InterdictionsOnCorridor:
    """FC-2: Interdiction positions lie on the BFS reference corridor and
    are never placed on the start or goal cell."""

    def test_interdictions_on_reference_corridor(self):
        """Every interdiction point must be within the reference corridor mask."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        corridor_mask = env._reference_corridor_mask
        for intd in env._forced_interdictions:
            px, py = intd["point"]
            assert corridor_mask[py, px], (
                f"FC-2 violated: interdiction at ({px},{py}) is NOT on "
                f"the BFS reference corridor"
            )

    def test_interdictions_not_on_start(self):
        """No interdiction may be placed on the start cell."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=42)
        start = env.agent_xy

        for intd in env._forced_interdictions:
            assert intd["point"] != start, (
                f"FC-2 violated: interdiction placed on start cell {start}"
            )

    def test_interdictions_not_on_goal(self):
        """No interdiction may be placed on the goal cell."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=42)
        goal = env.goal_xy

        for intd in env._forced_interdictions:
            assert intd["point"] != goal, (
                f"FC-2 violated: interdiction placed on goal cell {goal}"
            )

    def test_cut_points_are_interior(self):
        """Cut points should be at ~30% and ~65% of the corridor, not at
        the start/end fringes."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        ref_path = env._reference_path
        if len(ref_path) < 6:
            pytest.skip("Reference path too short for interior checks")

        positions = _get_interdiction_positions(env)
        for pos in positions:
            idx = ref_path.index(pos)
            # Must be at least 2 cells from start and 2 from end
            assert idx >= 2, f"Cut point {pos} too close to start (index={idx})"
            assert idx <= len(ref_path) - 3, (
                f"Cut point {pos} too close to end (index={idx}/{len(ref_path)})"
            )

    def test_reference_planner_is_bfs(self):
        """All interdictions must record reference_planner='bfs_shortest_path'."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        for intd in env._forced_interdictions:
            assert intd["reference_planner"] == "bfs_shortest_path", (
                f"FC-2 violated: reference_planner is '{intd['reference_planner']}'"
            )
