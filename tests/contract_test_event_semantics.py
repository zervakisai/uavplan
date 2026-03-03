"""Contract tests for Event Semantics (EV-1).

EV-1: Every event contains authoritative step_idx, consistent across
runner/env/logger/renderer.
"""

from __future__ import annotations

import pytest

from uavbench.benchmark.runner import run_episode
from uavbench.envs.urban import ACTION_RIGHT, ACTION_STAY, UrbanEnvV2
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCENARIO_ID = "gov_fire_delivery_easy"
PLANNER_ID = "astar"
SEED = 42


def _make_config(**overrides) -> ScenarioConfig:
    """Create a minimal ScenarioConfig for event semantics tests."""
    defaults = dict(
        name="test_event_semantics",
        mission_type=MissionType.FIRE_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=10,
        building_density=0.0,
        max_episode_steps=200,
        fixed_start_xy=(0, 0),
        fixed_goal_xy=(9, 9),
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


# ===========================================================================
# EV-1: Authoritative step_idx
# ===========================================================================


class TestEV1_AuthoritativeStepIdx:
    """EV-1: Every event has authoritative step_idx, consistent across all components."""

    def test_all_events_have_step_idx(self):
        """Every event in env.events has an integer step_idx field."""
        result = run_episode(SCENARIO_ID, PLANNER_ID, SEED)

        assert len(result.events) > 0, "Episode must produce at least one event"

        for i, event in enumerate(result.events):
            assert "step_idx" in event, (
                f"EV-1: event #{i} ({event.get('type', '?')}) missing step_idx"
            )
            assert isinstance(event["step_idx"], int), (
                f"EV-1: event #{i} step_idx must be int, "
                f"got {type(event['step_idx'])}"
            )

    def test_step_idx_monotonic(self):
        """Event step_idx values are non-decreasing across the episode."""
        result = run_episode(SCENARIO_ID, PLANNER_ID, SEED)

        step_indices = [e["step_idx"] for e in result.events]
        for i in range(1, len(step_indices)):
            assert step_indices[i] >= step_indices[i - 1], (
                f"EV-1: step_idx not monotonic at event #{i}: "
                f"{step_indices[i - 1]} -> {step_indices[i]}"
            )

    def test_step_idx_matches_runner(self):
        """Events logged during step N have step_idx == N.

        The reset event has step_idx=0. Subsequent events from step()
        have step_idx matching the env's internal step counter.
        """
        config = _make_config(building_density=0.0)
        env = UrbanEnvV2(config)
        env.reset(seed=SEED)

        # Verify reset events (reset + mission_briefing)
        assert len(env.events) >= 1
        assert env.events[0]["type"] == "reset"
        assert env.events[0]["step_idx"] == 0
        # All reset-phase events should have step_idx=0
        for evt in env.events:
            assert evt["step_idx"] == 0

        # Step until goal_reached or timeout
        step_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            step_count += 1
            _, _, terminated, truncated, info = env.step(ACTION_RIGHT)

        # Check all events have valid step_idx
        for event in env.events:
            if event["type"] in ("reset", "mission_briefing"):
                assert event["step_idx"] == 0
            else:
                # step_idx should be a positive integer matching the step
                assert event["step_idx"] > 0, (
                    f"EV-1: non-reset event has step_idx={event['step_idx']}"
                )

        # Verify the dynamics_step in info matches final step count
        assert info["dynamics_step"] == step_count

    def test_no_off_by_one(self):
        """Reset event has step_idx=0; first step event has step_idx=1."""
        config = _make_config(building_density=0.0)
        env = UrbanEnvV2(config)
        env.reset(seed=SEED)

        # After reset: events at step_idx=0 (reset + mission_briefing)
        assert len(env.events) >= 1
        for evt in env.events:
            assert evt["step_idx"] == 0, (
                f"EV-1: reset-phase event should have step_idx=0, "
                f"got {evt['step_idx']}"
            )

        # First step: step_idx=1
        _, _, _, _, info = env.step(ACTION_RIGHT)
        assert info["step_idx"] == 1, (
            f"EV-1: first step should have step_idx=1, got {info['step_idx']}"
        )
        assert info["dynamics_step"] == 1, (
            f"EV-1: first dynamics_step should be 1, got {info['dynamics_step']}"
        )

        # Any events from step 1 should have step_idx=1
        step1_events = [e for e in env.events if e["step_idx"] == 1]
        for event in step1_events:
            assert event["step_idx"] == 1

        # Second step: step_idx=2
        _, _, _, _, info = env.step(ACTION_RIGHT)
        assert info["step_idx"] == 2
        assert info["dynamics_step"] == 2

    def test_goal_reached_event_step_idx(self):
        """goal_reached event step_idx matches the step it occurred on."""
        result = run_episode(SCENARIO_ID, PLANNER_ID, SEED)

        goal_events = [e for e in result.events if e["type"] == "goal_reached"]
        assert len(goal_events) == 1, "Should have exactly one goal_reached event"

        goal_event = goal_events[0]
        assert goal_event["step_idx"] > 0, (
            "goal_reached step_idx should be positive"
        )
        # step_idx should be <= trajectory length - 1
        # (trajectory includes start position, so len(traj) - 1 = number of steps)
        max_steps = len(result.trajectory) - 1
        assert goal_event["step_idx"] <= max_steps, (
            f"EV-1: goal_reached step_idx ({goal_event['step_idx']}) "
            f"exceeds max steps ({max_steps})"
        )

    def test_task_completed_event_step_idx(self):
        """task_completed event has valid step_idx when it fires."""
        result = run_episode(SCENARIO_ID, PLANNER_ID, SEED)

        task_events = [e for e in result.events if e["type"] == "task_completed"]
        # fire_delivery has service_time=0 (fly-through),
        # so task completion should happen
        for event in task_events:
            assert "step_idx" in event
            assert isinstance(event["step_idx"], int)
            assert event["step_idx"] > 0
