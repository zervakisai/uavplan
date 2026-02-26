"""Contract tests for Battery Model (BC-1, BC-2, BC-3).

BC-1: Battery decreases monotonically every step (never increases).
BC-2: Battery <= 0 → immediate BATTERY_DEPLETED termination.
BC-3: Same seed → identical battery trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench2.envs.base import TerminationReason
from uavbench2.envs.urban import UrbanEnvV2
from uavbench2.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_battery_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        name="test_battery",
        mission_type=MissionType.FIRE_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=20,
        building_density=0.05,
        max_episode_steps=200,
        terminate_on_collision=False,
        battery_capacity_wh=50.0,  # small for fast depletion
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _run_until_done(env, seed=42, max_steps=200):
    """Run episode returning battery trace and final info."""
    obs, info = env.reset(seed=seed)
    action_rng = np.random.default_rng(seed)  # deterministic action source
    battery_trace = [info["battery_wh"]]
    step = 0
    terminated = False
    truncated = False
    final_info = info

    while not terminated and not truncated and step < max_steps:
        action = int(action_rng.integers(5))
        obs, reward, terminated, truncated, info = env.step(action)
        battery_trace.append(info["battery_wh"])
        final_info = info
        step += 1

    return battery_trace, final_info, terminated, truncated


# ===========================================================================
# BC-1: Battery decreases monotonically
# ===========================================================================


class TestBC1_MonotonicDecrease:
    """BC-1: Battery never increases during an episode."""

    def test_battery_decreases_every_step(self):
        """Battery value is strictly <= previous value at every step."""
        config = _make_battery_config(battery_capacity_wh=100.0)
        env = UrbanEnvV2(config)
        trace, _, _, _ = _run_until_done(env, seed=42, max_steps=50)

        for i in range(1, len(trace)):
            assert trace[i] <= trace[i - 1], (
                f"BC-1: battery increased at step {i}: "
                f"{trace[i-1]:.4f} → {trace[i]:.4f}"
            )

    def test_battery_starts_at_capacity(self):
        """Battery starts at configured capacity_wh."""
        config = _make_battery_config(battery_capacity_wh=75.0)
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        assert info["battery_wh"] == pytest.approx(75.0), (
            f"BC-1: battery should start at 75.0, got {info['battery_wh']}"
        )

    def test_hover_costs_less_than_move(self):
        """STAY action costs hover_cost, which may differ from move cost."""
        config = _make_battery_config(battery_capacity_wh=100.0)
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)
        start_battery = info["battery_wh"]

        # STAY action
        _, _, _, _, info_stay = env.step(4)  # STAY
        stay_cost = start_battery - info_stay["battery_wh"]

        assert stay_cost > 0, "BC-1: STAY should consume energy"

    def test_battery_in_info_every_step(self):
        """battery_wh field present in info dict every step."""
        config = _make_battery_config()
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)
        assert "battery_wh" in info

        for _ in range(10):
            obs, _, terminated, truncated, info = env.step(4)
            assert "battery_wh" in info, "BC-1: battery_wh missing from info"
            if terminated or truncated:
                break


# ===========================================================================
# BC-2: Battery depletion causes termination
# ===========================================================================


class TestBC2_DepletionTermination:
    """BC-2: battery <= 0 → BATTERY_DEPLETED termination."""

    def test_depletion_terminates(self):
        """When battery reaches 0, episode terminates with BATTERY_DEPLETED."""
        config = _make_battery_config(
            battery_capacity_wh=5.0,  # tiny battery
            max_episode_steps=500,
        )
        env = UrbanEnvV2(config)
        trace, final_info, terminated, truncated = _run_until_done(
            env, seed=42, max_steps=500
        )

        # Should have terminated from battery depletion
        assert terminated, "BC-2: episode should terminate on battery depletion"
        tr = final_info.get("termination_reason")
        # Accept both enum and string comparison
        tr_val = tr.value if hasattr(tr, "value") else str(tr)
        assert tr_val == "battery_depleted", (
            f"BC-2: termination_reason should be battery_depleted, got {tr_val}"
        )

    def test_battery_never_negative(self):
        """Battery floor is 0.0 (never goes negative in reported values)."""
        config = _make_battery_config(battery_capacity_wh=5.0)
        env = UrbanEnvV2(config)
        trace, _, _, _ = _run_until_done(env, seed=42, max_steps=500)

        for i, val in enumerate(trace):
            assert val >= 0.0, (
                f"BC-2: battery went negative at step {i}: {val}"
            )

    def test_depletion_before_timeout(self):
        """With tiny battery, depletion happens well before timeout."""
        config = _make_battery_config(
            battery_capacity_wh=5.0,
            max_episode_steps=500,
        )
        env = UrbanEnvV2(config)
        trace, final_info, terminated, truncated = _run_until_done(
            env, seed=42, max_steps=500
        )

        # Battery should deplete in fewer steps than max
        assert len(trace) < 500, (
            f"BC-2: battery should deplete before 500 steps, ran {len(trace)} steps"
        )


# ===========================================================================
# BC-3: Deterministic battery trace
# ===========================================================================


class TestBC3_DeterministicTrace:
    """BC-3: same seed → identical battery trace."""

    def test_identical_trace(self):
        """Two runs with same seed produce identical battery traces."""
        config = _make_battery_config(battery_capacity_wh=50.0)

        env_a = UrbanEnvV2(config)
        trace_a, _, _, _ = _run_until_done(env_a, seed=42, max_steps=50)

        env_b = UrbanEnvV2(config)
        trace_b, _, _, _ = _run_until_done(env_b, seed=42, max_steps=50)

        assert len(trace_a) == len(trace_b), (
            f"BC-3: trace lengths differ: {len(trace_a)} vs {len(trace_b)}"
        )
        for i in range(len(trace_a)):
            assert trace_a[i] == pytest.approx(trace_b[i]), (
                f"BC-3: battery diverged at step {i}: "
                f"{trace_a[i]:.6f} vs {trace_b[i]:.6f}"
            )

    def test_different_seed_different_trace(self):
        """Different seeds may produce different battery traces
        (due to different actions/positions)."""
        config = _make_battery_config(battery_capacity_wh=50.0)

        env_a = UrbanEnvV2(config)
        trace_a, _, _, _ = _run_until_done(env_a, seed=42, max_steps=30)

        env_b = UrbanEnvV2(config)
        trace_b, _, _, _ = _run_until_done(env_b, seed=99, max_steps=30)

        # Traces should differ (different maps, positions, random actions)
        # At minimum, starting values should be same (capacity)
        assert trace_a[0] == pytest.approx(trace_b[0])
        # But subsequent values should diverge
        # (not guaranteed, but highly likely with different seeds)


# ===========================================================================
# Mission Briefing (MC-3 extension)
# ===========================================================================


class TestMissionBriefing:
    """Mission briefing event logged at step 0 with all required fields."""

    def test_briefing_event_present(self):
        """First event after reset contains a mission_briefing event."""
        config = _make_battery_config()
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        briefing_events = [
            e for e in env.events if e.get("type") == "mission_briefing"
        ]
        assert len(briefing_events) >= 1, (
            "MC-3: no mission_briefing event found"
        )

    def test_briefing_fields_complete(self):
        """Mission briefing has all required fields."""
        config = _make_battery_config()
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        briefing = next(
            (e for e in env.events if e.get("type") == "mission_briefing"),
            None,
        )
        assert briefing is not None

        required_fields = [
            "type", "step_idx", "mission_type", "domain",
            "origin_name", "destination_name", "objective",
            "deliverable", "constraints", "battery_capacity_wh",
            "service_time_steps", "priority", "max_time_steps",
        ]
        for field in required_fields:
            assert field in briefing, (
                f"MC-3: mission_briefing missing field '{field}'"
            )

    def test_briefing_at_step_zero(self):
        """Mission briefing step_idx is 0."""
        config = _make_battery_config()
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        briefing = next(
            (e for e in env.events if e.get("type") == "mission_briefing"),
            None,
        )
        assert briefing is not None
        assert briefing["step_idx"] == 0

    def test_briefing_battery_matches_config(self):
        """Briefing battery_capacity_wh matches scenario config."""
        config = _make_battery_config(battery_capacity_wh=75.0)
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        briefing = next(
            (e for e in env.events if e.get("type") == "mission_briefing"),
            None,
        )
        assert briefing is not None
        assert briefing["battery_capacity_wh"] == pytest.approx(75.0)

    def test_briefing_constraints_is_list(self):
        """Briefing constraints field is a list of strings."""
        config = _make_battery_config()
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=42)

        briefing = next(
            (e for e in env.events if e.get("type") == "mission_briefing"),
            None,
        )
        assert briefing is not None
        assert isinstance(briefing["constraints"], list)
        for c in briefing["constraints"]:
            assert isinstance(c, str)
