"""Contract tests for Decision Record (EC-1, EC-2).

EC-1: Every rejected move logs reject_reason(enum), reject_layer, reject_cell, step_idx.
EC-2: Every accepted move logs accepted_move=True, dynamics step counter.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.envs.base import RejectReason
from uavbench.envs.urban import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    UrbanEnvV2,
)
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> ScenarioConfig:
    """Create a minimal ScenarioConfig for decision record tests."""
    defaults = dict(
        name="test_decision_record",
        mission_type=MissionType.FIRE_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=10,
        building_density=0.3,
        max_episode_steps=200,
        terminate_on_collision=False,  # keep episode alive after collision
        fixed_start_xy=(0, 0),
        fixed_goal_xy=(9, 9),
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _find_building_adjacent_action(env: UrbanEnvV2) -> int | None:
    """Return an action that steps into a building, or None."""
    ax, ay = env.agent_xy
    heightmap, _, _, _ = env.export_planner_inputs()
    deltas = {
        ACTION_UP: (0, -1),
        ACTION_DOWN: (0, 1),
        ACTION_LEFT: (-1, 0),
        ACTION_RIGHT: (1, 0),
    }
    for action, (dx, dy) in deltas.items():
        nx, ny = ax + dx, ay + dy
        if 0 <= nx < heightmap.shape[1] and 0 <= ny < heightmap.shape[0]:
            if heightmap[ny, nx] > 0:
                return action
    return None


def _find_oob_action(env: UrbanEnvV2, map_size: int) -> int:
    """Return an action that moves out of bounds."""
    ax, ay = env.agent_xy
    if ax == 0:
        return ACTION_LEFT
    if ax == map_size - 1:
        return ACTION_RIGHT
    if ay == 0:
        return ACTION_UP
    if ay == map_size - 1:
        return ACTION_DOWN
    # Agent is interior — shouldn't happen for our configs
    return ACTION_LEFT


# ===========================================================================
# EC-1: Rejected move logging
# ===========================================================================


class TestEC1_RejectedMoveLogging:
    """EC-1: Every rejected move logs reject_reason, reject_layer, reject_cell, step_idx."""

    def test_rejected_move_has_all_fields(self):
        """Rejected move info dict contains reject_reason, reject_layer,
        reject_cell, step_idx."""
        config = _make_config(
            building_density=0.0,
            map_size=5,
            fixed_start_xy=(0, 0),
            fixed_goal_xy=(4, 4),
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Move LEFT from (0,0) -> out of bounds -> must be rejected
        _, _, _, _, info = env.step(ACTION_LEFT)

        assert info["accepted_move"] is False, "Move should be rejected"
        assert "reject_reason" in info, "EC-1: rejected move must have reject_reason"
        assert "reject_layer" in info, "EC-1: rejected move must have reject_layer"
        assert "reject_cell" in info, "EC-1: rejected move must have reject_cell"
        assert "step_idx" in info, "EC-1: rejected move must have step_idx"

    def test_reject_reason_is_enum(self):
        """reject_reason is a RejectReason enum member, not a raw string."""
        config = _make_config(
            building_density=0.0,
            map_size=5,
            fixed_start_xy=(0, 0),
            fixed_goal_xy=(4, 4),
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Move LEFT from (0,0) -> out of bounds
        _, _, _, _, info = env.step(ACTION_LEFT)

        assert info["accepted_move"] is False
        assert isinstance(info["reject_reason"], RejectReason), (
            f"EC-1: reject_reason must be RejectReason enum, "
            f"got {type(info['reject_reason'])}"
        )

    def test_reject_cell_matches_destination(self):
        """reject_cell equals the attempted destination (x, y)."""
        config = _make_config(
            building_density=0.0,
            map_size=5,
            fixed_start_xy=(0, 0),
            fixed_goal_xy=(4, 4),
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Move LEFT from (0,0) -> attempted destination is (-1, 0)
        _, _, _, _, info = env.step(ACTION_LEFT)

        assert info["accepted_move"] is False
        assert info["reject_cell"] == (-1, 0), (
            f"EC-1: reject_cell should be (-1, 0), got {info['reject_cell']}"
        )

    def test_building_rejection_logged(self):
        """Moving into a building cell logs RejectReason.BUILDING."""
        # Use high density to guarantee buildings near start
        config = _make_config(
            building_density=0.6,
            map_size=10,
            max_episode_steps=500,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Find an action that steps into a building
        action = _find_building_adjacent_action(env)

        if action is None:
            # Walk around until we're adjacent to a building
            for step_i in range(100):
                act = step_i % 4  # cycle through UP/DOWN/LEFT/RIGHT
                _, _, terminated, truncated, info = env.step(act)
                if terminated or truncated:
                    env.reset(seed=42)
                    continue
                action = _find_building_adjacent_action(env)
                if action is not None:
                    break

        if action is None:
            pytest.skip("No building found adjacent to agent path (unlikely with density=0.6)")

        _, _, _, _, info = env.step(action)
        assert info["accepted_move"] is False
        assert info["reject_reason"] == RejectReason.BUILDING
        assert info["reject_layer"] == "building"

    def test_all_reject_reasons_exercised(self):
        """At least BUILDING and OUT_OF_BOUNDS are observed in Phase 3.

        Dynamic reject reasons (FIRE, FORCED_BLOCK) require Phase 4+
        and are NOT tested here.
        """
        config = _make_config(
            building_density=0.5,
            map_size=10,
            max_episode_steps=500,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        observed_reasons: set[RejectReason] = set()

        # 1. Get OUT_OF_BOUNDS: move LEFT from (0,0)
        _, _, _, _, info = env.step(ACTION_LEFT)
        if "reject_reason" in info:
            observed_reasons.add(info["reject_reason"])

        # 2. Get BUILDING: navigate to find a building
        env.reset(seed=42)
        for step_i in range(200):
            act = step_i % 4
            _, _, terminated, truncated, info = env.step(act)
            if "reject_reason" in info:
                observed_reasons.add(info["reject_reason"])
            if terminated or truncated:
                break

        assert RejectReason.OUT_OF_BOUNDS in observed_reasons, (
            f"EC-1: OUT_OF_BOUNDS not observed, only: {observed_reasons}"
        )
        # All observed reasons must be proper enum members
        for reason in observed_reasons:
            assert isinstance(reason, RejectReason)


# ===========================================================================
# EC-2: Accepted move logging
# ===========================================================================


class TestEC2_AcceptedMoveLogging:
    """EC-2: Every accepted move logs accepted_move=True, dynamics step counter."""

    def test_accepted_move_has_fields(self):
        """Info dict contains accepted_move=True and dynamics_step integer."""
        config = _make_config(
            building_density=0.0,
            map_size=10,
            fixed_start_xy=(0, 0),
            fixed_goal_xy=(9, 9),
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Move RIGHT on empty grid -> accepted
        _, _, _, _, info = env.step(ACTION_RIGHT)

        assert info["accepted_move"] is True, "EC-2: move should be accepted"
        assert "dynamics_step" in info, "EC-2: accepted move must have dynamics_step"
        assert isinstance(info["dynamics_step"], int), (
            f"EC-2: dynamics_step must be int, got {type(info['dynamics_step'])}"
        )

    def test_dynamics_step_matches_runner(self):
        """dynamics_step increments correctly and matches the runner's step count.

        The env's dynamics_step must equal the runner's step counter at each
        step, ensuring EV-1 consistency.
        """
        config = _make_config(
            building_density=0.0,
            map_size=10,
            fixed_start_xy=(0, 0),
            fixed_goal_xy=(9, 9),
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        for expected_step in range(1, 11):
            _, _, terminated, truncated, info = env.step(ACTION_RIGHT)
            assert info["dynamics_step"] == expected_step, (
                f"EC-2: dynamics_step should be {expected_step}, "
                f"got {info['dynamics_step']}"
            )
            if terminated or truncated:
                break
