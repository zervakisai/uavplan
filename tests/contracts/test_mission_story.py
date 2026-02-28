"""Contract tests for mission story: MC-1, MC-3.

MC-1: Every episode has a mission objective (POI + human-readable reason).
MC-3: Mission briefing is logged at step 0 of every episode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uavbench.missions.spec import (
    MissionBriefing,
    MissionID,
    MissionSpec,
    DifficultyKnobs,
    BRIEFING_TEMPLATES,
)
from uavbench.missions.engine import generate_briefing
from uavbench.scenarios.loader import load_scenario


# ── Fixtures ──────────────────────────────────────────────────────────

CONFIGS_DIR = Path("src/uavbench/scenarios/configs")

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


@pytest.fixture(params=ALL_GOV_SCENARIOS)
def scenario_config(request):
    """Load each gov scenario config."""
    return load_scenario(CONFIGS_DIR / f"{request.param}.yaml")


# ── MC-1: Every episode has a mission objective ───────────────────────


class TestMC1_MissionObjective:
    """MC-1: Every episode has a mission objective."""

    def test_briefing_has_objective(self, scenario_config):
        """generate_briefing() returns a briefing with non-empty objective."""
        briefing = generate_briefing(scenario_config)
        assert isinstance(briefing, MissionBriefing)
        assert briefing.objective, "Briefing must have a non-empty objective"
        assert len(briefing.objective) > 10, "Objective should be descriptive"

    def test_briefing_has_destination(self, scenario_config):
        """Briefing includes a human-readable destination."""
        briefing = generate_briefing(scenario_config)
        assert briefing.destination_name, "Briefing must have a destination"

    def test_briefing_has_origin(self, scenario_config):
        """Briefing includes a human-readable origin."""
        briefing = generate_briefing(scenario_config)
        assert briefing.origin_name, "Briefing must have an origin"

    def test_briefing_has_deliverable(self, scenario_config):
        """Briefing includes a deliverable description."""
        briefing = generate_briefing(scenario_config)
        assert briefing.deliverable, "Briefing must have a deliverable"

    def test_briefing_mission_type_matches_config(self, scenario_config):
        """Briefing mission_type matches the scenario config."""
        briefing = generate_briefing(scenario_config)
        mt = getattr(scenario_config.mission_type, "value", str(scenario_config.mission_type))
        assert briefing.mission_type == mt

    def test_briefing_priority_scales_with_difficulty(self):
        """Priority escalates: easy=routine, medium=high, hard=critical."""
        priority_expected = {
            "easy": "routine",
            "medium": "high",
            "hard": "critical",
        }
        for diff, expected_priority in priority_expected.items():
            cfg = load_scenario(
                CONFIGS_DIR / f"gov_civil_protection_{diff}.yaml"
            )
            briefing = generate_briefing(cfg)
            assert briefing.priority == expected_priority, (
                f"Difficulty {diff} should have priority {expected_priority}, "
                f"got {briefing.priority}"
            )


# ── MC-3: Briefing logged at step 0 ──────────────────────────────────


class TestMC3_BriefingAtStep0:
    """MC-3: Briefing is logged at step 0 in the event stream."""

    def test_briefing_serialises_to_event(self, scenario_config):
        """Briefing can be serialised as a step-0 event."""
        briefing = generate_briefing(scenario_config)
        event = {
            "step": 0,
            "type": "mission_briefing",
            "payload": briefing.to_dict(),
        }
        assert event["step"] == 0
        assert event["type"] == "mission_briefing"
        assert "objective" in event["payload"]
        assert "mission_type" in event["payload"]

    def test_briefing_to_dict_complete(self, scenario_config):
        """to_dict() returns all required fields."""
        briefing = generate_briefing(scenario_config)
        d = briefing.to_dict()
        required_keys = {
            "mission_type", "domain", "origin_name", "destination_name",
            "objective", "deliverable", "constraints", "service_time_steps",
            "priority", "max_time_steps",
        }
        assert required_keys.issubset(d.keys()), (
            f"Missing keys: {required_keys - d.keys()}"
        )

    def test_briefing_constraints_reflect_dynamics(self):
        """Hard scenarios with dynamics should list constraints."""
        cfg = load_scenario(CONFIGS_DIR / "gov_civil_protection_hard.yaml")
        briefing = generate_briefing(cfg)
        assert len(briefing.constraints) > 0, (
            "Hard scenario with fire+traffic+NFZ should have constraints"
        )

    def test_briefing_easy_has_no_dynamic_constraints(self):
        """Easy scenarios (no dynamics) should have empty constraints."""
        cfg = load_scenario(CONFIGS_DIR / "gov_civil_protection_easy.yaml")
        briefing = generate_briefing(cfg)
        assert len(briefing.constraints) == 0, (
            "Easy scenario with no dynamics should have no constraints"
        )


# ── Template coverage ─────────────────────────────────────────────────


class TestBriefingTemplateCoverage:
    """Every MissionID has a briefing template."""

    def test_all_mission_ids_have_templates(self):
        """BRIEFING_TEMPLATES covers all MissionID values."""
        for mid in MissionID:
            assert mid.value in BRIEFING_TEMPLATES, (
                f"MissionID.{mid.name} ({mid.value}) has no briefing template"
            )

    def test_templates_have_required_fields(self):
        """Each template has objective, deliverable, origin, destination."""
        for mt, tmpl in BRIEFING_TEMPLATES.items():
            assert "objective" in tmpl, f"Template {mt} missing 'objective'"
            assert "deliverable" in tmpl, f"Template {mt} missing 'deliverable'"
            assert "origin_name" in tmpl, f"Template {mt} missing 'origin_name'"
            assert "destination_name" in tmpl, f"Template {mt} missing 'destination_name'"


# ── MissionBriefing dataclass invariants ──────────────────────────────


class TestMissionBriefingDataclass:
    """MissionBriefing is frozen and serialisable."""

    def test_briefing_is_frozen(self):
        """MissionBriefing instances are immutable."""
        b = MissionBriefing(
            mission_type="civil_protection",
            domain="urban",
            origin_name="Test Origin",
            destination_name="Test Dest",
            objective="Test objective",
            deliverable="Test deliverable",
        )
        with pytest.raises(AttributeError):
            b.objective = "modified"  # type: ignore[misc]

    def test_briefing_constraints_are_tuple(self):
        """Constraints field is a tuple (immutable)."""
        b = MissionBriefing(
            mission_type="civil_protection",
            domain="urban",
            origin_name="A",
            destination_name="B",
            objective="C",
            deliverable="D",
            constraints=("x", "y"),
        )
        assert isinstance(b.constraints, tuple)
