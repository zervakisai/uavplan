"""Contract tests for Mission HUD visual truth.

Contracts
---------
VC-1: If a plan exists, the HUD shows PLAN: ACTIVE in the rendered text.
VC-2: If no plan exists, the HUD shows NO_PLAN or STALE — never blank.

These tests validate the HUD *logic* (format_lines, derive_plan_status)
without requiring matplotlib rendering, keeping them fast and CI-safe.
"""

from __future__ import annotations

import pytest

from uavbench.visualization.hud import (
    MissionHUD,
    MissionHUDState,
    derive_plan_status,
    derive_mission_status,
    MISSION_STATUS_EN_ROUTE,
    MISSION_STATUS_SERVICING,
    MISSION_STATUS_COMPLETED,
    MISSION_STATUS_FAILED,
    PLAN_STATUS_ACTIVE,
    PLAN_STATUS_NO_PLAN,
    PLAN_STATUS_STALE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def hud_with_briefing() -> MissionHUD:
    return MissionHUD(has_briefing=True)


@pytest.fixture
def hud_no_briefing() -> MissionHUD:
    return MissionHUD(has_briefing=False)


@pytest.fixture
def default_state() -> MissionHUDState:
    return MissionHUDState(
        objective="Emergency medical delivery during wildfire crisis",
        destination_name="Evacuation Zone Alpha",
        mission_status=MISSION_STATUS_EN_ROUTE,
        distance_to_goal=42,
        plan_status=PLAN_STATUS_ACTIVE,
        step=10,
        max_steps=200,
    )


# ── VC-1: Plan visible in frame ─────────────────────────────────────────

class TestVC1_PlanVisibleInFrame:
    """VC-1: If a plan exists, it appears in the rendered frame."""

    def test_plan_active_shows_in_hud(
        self, hud_with_briefing: MissionHUD, default_state: MissionHUDState
    ) -> None:
        text = hud_with_briefing.format_lines(default_state)
        assert "PLAN: ACTIVE" in text, "VC-1: PLAN: ACTIVE must appear when plan exists"

    def test_plan_active_derived_from_plan_len(self) -> None:
        status = derive_plan_status(plan_len=15, plan_stale=False, plan_reason="initial")
        assert status == PLAN_STATUS_ACTIVE

    def test_plan_active_for_single_cell_plan(self) -> None:
        """A plan_len=1 (at-goal) is still a valid plan."""
        status = derive_plan_status(plan_len=1, plan_stale=False, plan_reason="none")
        assert status == PLAN_STATUS_ACTIVE

    def test_mission_line_contains_objective_and_destination(
        self, hud_with_briefing: MissionHUD, default_state: MissionHUDState
    ) -> None:
        text = hud_with_briefing.format_lines(default_state)
        assert "Emergency medical delivery" in text
        assert "Evacuation Zone Alpha" in text

    def test_step_line_shows_current_and_max(
        self, hud_with_briefing: MissionHUD, default_state: MissionHUDState
    ) -> None:
        text = hud_with_briefing.format_lines(default_state)
        assert "STEP: 10 / 200" in text

    def test_distance_shown_in_status(
        self, hud_with_briefing: MissionHUD, default_state: MissionHUDState
    ) -> None:
        text = hud_with_briefing.format_lines(default_state)
        assert "Dist: 42 cells" in text

    def test_four_lines_when_briefing_present(
        self, hud_with_briefing: MissionHUD, default_state: MissionHUDState
    ) -> None:
        text = hud_with_briefing.format_lines(default_state)
        lines = text.strip().split("\n")
        assert len(lines) == 4, f"Expected 4 HUD lines, got {len(lines)}"


# ── VC-2: NO_PLAN / STALE badge ─────────────────────────────────────────

class TestVC2_NoPlanBadge:
    """VC-2: If no plan exists, HUD shows NO_PLAN or STALE — never blank."""

    def test_stale_plan_shows_stale(self) -> None:
        status = derive_plan_status(plan_len=10, plan_stale=True, plan_reason="fire")
        assert status == PLAN_STATUS_STALE

    def test_no_plan_when_plan_len_zero(self) -> None:
        status = derive_plan_status(plan_len=0, plan_stale=False, plan_reason="none")
        assert status == PLAN_STATUS_NO_PLAN

    def test_no_plan_when_plan_len_low_with_reason(self) -> None:
        status = derive_plan_status(plan_len=0, plan_stale=False, plan_reason="blocked")
        assert status == PLAN_STATUS_NO_PLAN

    def test_stale_overrides_plan_len(self) -> None:
        """Even with plan_len > 1, stale flag takes precedence."""
        status = derive_plan_status(plan_len=50, plan_stale=True, plan_reason="nfz")
        assert status == PLAN_STATUS_STALE

    def test_hud_text_shows_no_plan(self, hud_with_briefing: MissionHUD) -> None:
        state = MissionHUDState(
            objective="Test mission",
            destination_name="Destination",
            plan_status=PLAN_STATUS_NO_PLAN,
            step=5,
            max_steps=100,
        )
        text = hud_with_briefing.format_lines(state)
        assert "PLAN: NO_PLAN" in text, "VC-2: NO_PLAN must appear when no plan"

    def test_hud_text_shows_stale(self, hud_with_briefing: MissionHUD) -> None:
        state = MissionHUDState(
            objective="Test mission",
            destination_name="Destination",
            plan_status=PLAN_STATUS_STALE,
            step=5,
            max_steps=100,
        )
        text = hud_with_briefing.format_lines(state)
        assert "PLAN: STALE" in text, "VC-2: STALE must appear when plan is stale"

    def test_plan_line_never_blank(self, hud_with_briefing: MissionHUD) -> None:
        """The PLAN line must always have a badge, regardless of input."""
        for plan_status in [PLAN_STATUS_ACTIVE, PLAN_STATUS_NO_PLAN, PLAN_STATUS_STALE]:
            state = MissionHUDState(
                objective="X", destination_name="Y",
                plan_status=plan_status,
            )
            text = hud_with_briefing.format_lines(state)
            assert "PLAN:" in text, f"PLAN line missing for status={plan_status}"
            plan_line = [l for l in text.split("\n") if l.startswith("PLAN:")][0]
            badge = plan_line.split("PLAN:")[1].strip()
            assert badge, f"VC-2: PLAN badge is blank for status={plan_status}"


# ── Legacy / degradation ─────────────────────────────────────────────────

class TestVC_LegacyDegradation:
    """HUD degrades gracefully when no briefing is available."""

    def test_no_briefing_shows_no_mission(self, hud_no_briefing: MissionHUD) -> None:
        text = hud_no_briefing.format_lines(MissionHUDState())
        assert text == "NO MISSION"

    def test_no_briefing_single_line(self, hud_no_briefing: MissionHUD) -> None:
        text = hud_no_briefing.format_lines(MissionHUDState())
        assert "\n" not in text, "NO MISSION should be a single line"


# ── derive_mission_status ────────────────────────────────────────────────

class TestDeriveMissionStatus:
    """Unit tests for derive_mission_status helper."""

    def test_en_route_default(self) -> None:
        assert derive_mission_status() == MISSION_STATUS_EN_ROUTE

    def test_completed_on_goal(self) -> None:
        status = derive_mission_status(terminated=True, reached_goal=True)
        assert status == MISSION_STATUS_COMPLETED

    def test_failed_on_terminate_no_goal(self) -> None:
        status = derive_mission_status(terminated=True, reached_goal=False)
        assert status == MISSION_STATUS_FAILED

    def test_servicing_at_goal(self) -> None:
        status = derive_mission_status(at_goal=True)
        assert status == MISSION_STATUS_SERVICING

    def test_en_route_mid_flight(self) -> None:
        status = derive_mission_status(terminated=False, at_goal=False)
        assert status == MISSION_STATUS_EN_ROUTE
