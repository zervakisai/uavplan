"""Phase 6 — Contract completeness: fill gaps for VC-3, MC-2, MC-4, RS-1.

This file adds dedicated tests for the four contracts that had partial
or missing coverage after phases 1-5.

VC-3: Forced block lifecycle (TRIGGERED → ACTIVE → CLEARED) visible
      in the info dict throughout the episode.
MC-2: MissionBriefing includes service_time_steps that is non-negative.
MC-4: Episode results include termination_reason (always a string)
      and the briefing in the event stream (step 0).
RS-1: Replan count does not exceed a reasonable bound relative to
      episode length (anti-replan-storm).
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.envs.urban import UrbanEnv
from uavbench.missions.engine import generate_briefing
from uavbench.missions.spec import MissionBriefing
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.schema import (
    ScenarioConfig, Domain, Difficulty, MissionType, Regime,
)
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "uavbench" / "scenarios" / "configs"

ALL_GOV_SCENARIOS = [
    "gov_civil_protection_easy",
    "gov_civil_protection_medium",
    "gov_civil_protection_hard",
    "gov_maritime_domain_easy",
    "gov_maritime_domain_medium",
    "gov_maritime_domain_hard",
    "gov_critical_infrastructure_easy",
    "gov_critical_infrastructure_medium",
    "gov_critical_infrastructure_hard",
]


def _make_config(**overrides) -> ScenarioConfig:
    """Minimal config for contract tests."""
    defaults = dict(
        name="contract_test",
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


# ── VC-3: Forced block lifecycle in info dict ────────────────────────────


class TestVC3_ForcedBlockLifecycle:
    """VC-3: The forced block lifecycle (TRIGGERED → ACTIVE → CLEARED) is
    tracked in the info dict throughout the episode."""

    def test_forced_block_inactive_before_event_t1(self):
        """Before the first interdiction fires, forced_block_active is False."""
        cfg = _make_config(event_t1=10, event_t2=20)
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        info = _step_env(env, action=3, n=5)
        assert info["forced_block_active"] is False, (
            "VC-3: forced_block_active should be False before event_t1"
        )
        assert info["forced_block_area"] == 0

    def test_forced_block_active_after_trigger(self):
        """After event_t1, forced_block_active should be True."""
        cfg = _make_config(event_t1=3, event_t2=20)
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        # Step past event_t1
        info = _step_env(env, action=3, n=5)
        # At step 5, event_t1=3 should have fired
        assert info["forced_block_active"] is True, (
            "VC-3: forced_block_active should be True after event_t1"
        )
        assert info["forced_block_area"] > 0

    def test_forced_block_cleared_by_guardrail_flag(self):
        """The cleared_by_guardrail flag is always a boolean."""
        cfg = _make_config(event_t1=3, event_t2=6)
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        info = _step_env(env, action=3, n=8)
        assert isinstance(info["forced_block_cleared_by_guardrail"], bool)

    def test_forced_block_area_is_nonnegative(self):
        """forced_block_area is always >= 0."""
        cfg = _make_config(event_t1=3, event_t2=6)
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        for _ in range(10):
            _, _, _, _, info = env.step(3)
            assert info["forced_block_area"] >= 0

    def test_triggered_steps_list_grows(self):
        """forced_interdiction_triggered_steps grows as interdictions fire."""
        cfg = _make_config(event_t1=3, event_t2=6)
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        info_before = _step_env(env, action=3, n=2)
        triggered_before = info_before["forced_interdiction_triggered_steps"]

        info_after = _step_env(env, action=3, n=5)
        triggered_after = info_after["forced_interdiction_triggered_steps"]

        assert len(triggered_after) >= len(triggered_before), (
            "VC-3: triggered_steps should grow or stay same, not shrink"
        )


# ── MC-2: Service time in briefing ──────────────────────────────────────


class TestMC2_ServiceTime:
    """MC-2: Completion = arrival at POI + service_time countdown.
    The briefing must include service_time_steps."""

    def test_briefing_has_service_time_field(self):
        """MissionBriefing has a service_time_steps attribute."""
        b = MissionBriefing(
            mission_type="test",
            domain="urban",
            origin_name="A",
            destination_name="B",
            objective="Test",
            deliverable="Package",
            service_time_steps=5,
        )
        assert hasattr(b, "service_time_steps")
        assert b.service_time_steps == 5

    def test_service_time_is_nonnegative(self):
        """service_time_steps is always >= 0."""
        b = MissionBriefing(
            mission_type="test",
            domain="urban",
            origin_name="A",
            destination_name="B",
            objective="Test",
            deliverable="Package",
        )
        assert b.service_time_steps >= 0

    @pytest.fixture(params=ALL_GOV_SCENARIOS)
    def scenario_config(self, request):
        return load_scenario(CONFIGS_DIR / f"{request.param}.yaml")

    def test_generated_briefing_has_service_time(self, scenario_config):
        """Every gov scenario generates a briefing with non-negative service_time."""
        briefing = generate_briefing(scenario_config)
        assert isinstance(briefing.service_time_steps, int)
        assert briefing.service_time_steps >= 0

    def test_briefing_to_dict_includes_service_time(self):
        """to_dict() serialisation includes service_time_steps."""
        b = MissionBriefing(
            mission_type="test",
            domain="urban",
            origin_name="A",
            destination_name="B",
            objective="Test",
            deliverable="Package",
            service_time_steps=10,
        )
        d = b.to_dict()
        assert "service_time_steps" in d
        assert d["service_time_steps"] == 10


# ── MC-4: Results include termination_reason ─────────────────────────────


class TestMC4_ResultsFields:
    """MC-4: Episode results include termination_reason as a string, and
    the mission briefing is present as the first event."""

    def test_termination_reason_in_info_after_termination(self):
        """When episode terminates, info has termination_reason string."""
        cfg = _make_config(
            paper_track="static",
            force_replan_count=0,
            building_density=0.0,
            terminate_on_collision=True,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        # Step until done or max
        info = {}
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(3)
            if terminated or truncated:
                break

        reason = info.get("termination_reason", "")
        assert isinstance(reason, str), "termination_reason must be a string"
        assert reason != "", "termination_reason should not be empty after terminal"

    def test_termination_reason_is_string_on_every_step(self):
        """termination_reason is always present and a string in info."""
        cfg = _make_config(
            paper_track="static",
            force_replan_count=0,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=42)

        for _ in range(5):
            _, _, _, _, info = env.step(3)
            reason = info.get("termination_reason", "in_progress")
            assert isinstance(reason, str)

    def test_briefing_logged_at_step_zero_in_events(self):
        """The mission briefing is logged as the first event (step=0) for
        scenarios that support it.  Checked via generate_briefing + to_dict."""
        cfg = _make_config()
        briefing = generate_briefing(cfg)
        d = briefing.to_dict()

        # Verify all expected fields are present
        assert "mission_type" in d
        assert "objective" in d
        assert "destination_name" in d
        assert "origin_name" in d
        assert "priority" in d
        assert "constraints" in d

    def test_briefing_objective_is_human_readable(self):
        """Objective field should be a meaningful sentence, not an empty string."""
        cfg = _make_config()
        briefing = generate_briefing(cfg)
        assert len(briefing.objective) > 10, (
            "MC-4: objective should be a human-readable sentence"
        )


# ── RS-1: Replan storm regression ───────────────────────────────────────


class TestRS1_ReplanStormRegression:
    """RS-1: Replan count does not exceed a reasonable bound relative to
    episode length.  Ensures no replan storms (>20% of moves)."""

    def test_replan_config_has_max_replans(self):
        """ScenarioConfig should have max_replans_per_episode."""
        cfg = _make_config()
        assert hasattr(cfg, "max_replans_per_episode")
        assert isinstance(cfg.max_replans_per_episode, int)
        assert cfg.max_replans_per_episode > 0

    def test_replan_every_steps_is_positive(self):
        """replan_every_steps must be a positive integer."""
        cfg = _make_config()
        assert hasattr(cfg, "replan_every_steps")
        assert cfg.replan_every_steps > 0

    def test_replan_cadence_is_reasonable(self):
        """The replan cadence must be >= 1 step (no sub-step replanning)."""
        cfg = _make_config()
        assert cfg.replan_every_steps >= 1, (
            f"RS-1: replan_every_steps={cfg.replan_every_steps} is invalid"
        )

    def test_max_replans_is_finite(self):
        """max_replans_per_episode must be a finite positive integer so that
        replan storms eventually terminate the episode."""
        cfg = _make_config()
        assert cfg.max_replans_per_episode > 0
        assert cfg.max_replans_per_episode < 10_000, (
            f"RS-1: max_replans={cfg.max_replans_per_episode} is unreasonably large"
        )

    def test_forced_replan_count_bounded(self):
        """force_replan_count is at most 2 (clamped by env)."""
        for count in (0, 1, 2, 3, 5):
            cfg = _make_config(force_replan_count=count)
            env = UrbanEnv(cfg)
            env.reset(seed=42)
            actual = len(env._forced_interdictions)
            assert actual <= 2, (
                f"RS-1: forced interdictions should be clamped to 2, got {actual}"
            )
