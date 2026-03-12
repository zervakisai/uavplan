"""Tests for fire/debris termination — agent caught by hazard is destroyed.

BUG: Fire or debris reaching the agent's cell after dynamics advance didn't
terminate the episode. The agent would sit in fire indefinitely, getting
move rejections but never failing.

Fix: Post-dynamics survival check in env.step() emits FIRE_CAUGHT or
DEBRIS_CAUGHT termination.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.envs.base import TerminationReason
from uavbench.envs.urban import ACTION_STAY, UrbanEnvV2
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


def _make_config(**overrides) -> ScenarioConfig:
    """Minimal config with fire enabled on a small grid."""
    defaults = dict(
        name="test_fire_termination",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=10,
        building_density=0.0,
        max_episode_steps=500,
        fixed_start_xy=(5, 5),
        fixed_goal_xy=(9, 9),
        enable_fire=True,
        fire_ignition_points=1,  # need ≥1 so FireSpreadModel is created
        fire_buffer_radius=0,    # no buffer so we test direct fire contact
        terminate_on_collision=True,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


class TestFireTermination:
    """Agent caught by fire is destroyed (FIRE_CAUGHT)."""

    def test_fire_on_agent_cell_terminates(self):
        """If fire spreads to the agent's cell, episode terminates with FIRE_CAUGHT."""
        config = _make_config()
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        # Verify agent is at (5, 5)
        assert env._agent_xy == (5, 5)

        # Manually place fire directly on the agent's cell.
        # After dynamics advance, the agent should be caught.
        # We set fire on agent cell BEFORE the step so that after
        # _step_dynamics runs, fire is still there.
        if env._fire is not None:
            env._fire._state[5, 5] = 1  # BURNING state
            env._fire._burn_timer[5, 5] = 0

        # Take a STAY action — fire is already on our cell, dynamics run,
        # survival check should catch us
        obs, reward, terminated, truncated, info = env.step(ACTION_STAY)

        assert terminated, "Agent in fire cell should be terminated"
        assert env._termination_reason == TerminationReason.FIRE_CAUGHT

    def test_fire_caught_event_emitted(self):
        """FIRE_CAUGHT termination emits a fire_caught event."""
        config = _make_config()
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        if env._fire is not None:
            env._fire._state[5, 5] = 1  # BURNING
            env._fire._burn_timer[5, 5] = 0

        env.step(ACTION_STAY)

        fire_events = [e for e in env.events if e["type"] == "fire_caught"]
        assert len(fire_events) == 1
        assert fire_events[0]["step_idx"] == 1
        assert fire_events[0]["cell"] == (5, 5)

    def test_fire_caught_gives_negative_reward(self):
        """Being caught by fire gives a terminal penalty."""
        config = _make_config()
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        if env._fire is not None:
            env._fire._state[5, 5] = 1
            env._fire._burn_timer[5, 5] = 0

        _, reward, terminated, _, _ = env.step(ACTION_STAY)
        assert terminated
        assert reward < -10.0, "Fire caught should have a significant penalty"


class TestDebrisTermination:
    """Agent caught by debris is destroyed (DEBRIS_CAUGHT)."""

    def test_debris_on_agent_cell_terminates(self):
        """If debris appears on agent's cell, episode terminates with DEBRIS_CAUGHT."""
        config = _make_config(enable_collapse=True, collapse_delay=1, debris_prob=1.0)
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        assert env._agent_xy == (5, 5)

        # Manually place debris on agent cell
        if env._collapse is not None:
            env._collapse._debris[5, 5] = True

        obs, reward, terminated, truncated, info = env.step(ACTION_STAY)

        assert terminated, "Agent on debris cell should be terminated"
        assert env._termination_reason == TerminationReason.DEBRIS_CAUGHT

    def test_debris_caught_event_emitted(self):
        """DEBRIS_CAUGHT termination emits a debris_caught event."""
        config = _make_config(enable_collapse=True, collapse_delay=1, debris_prob=1.0)
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        if env._collapse is not None:
            env._collapse._debris[5, 5] = True

        env.step(ACTION_STAY)

        debris_events = [e for e in env.events if e["type"] == "debris_caught"]
        assert len(debris_events) == 1
        assert debris_events[0]["cell"] == (5, 5)


class TestNoFalseTermination:
    """Survival check doesn't trigger false positives."""

    def test_no_fire_no_termination(self):
        """Without fire on agent cell, no FIRE_CAUGHT termination."""
        config = _make_config()
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Fire NOT on agent cell
        obs, reward, terminated, truncated, info = env.step(ACTION_STAY)
        assert not terminated or env._termination_reason not in (
            TerminationReason.FIRE_CAUGHT,
            TerminationReason.DEBRIS_CAUGHT,
        )

    def test_fire_far_away_no_termination(self):
        """Fire far from agent doesn't trigger FIRE_CAUGHT."""
        config = _make_config()
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Clear all fire, place fire far from agent at (0,0)
        if env._fire is not None:
            env._fire._state[:] = 0  # clear everything
            env._fire._burn_timer[:] = 0
            env._fire._state[0, 0] = 1  # BURNING at (0,0), far from (5,5)
            env._fire._burn_timer[0, 0] = 0

        obs, reward, terminated, truncated, info = env.step(ACTION_STAY)
        # Should NOT be terminated for fire_caught (fire too far away)
        if terminated:
            assert env._termination_reason != TerminationReason.FIRE_CAUGHT
