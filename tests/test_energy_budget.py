"""Tests for energy budget constraint (Upgrade 10)."""
from __future__ import annotations

import numpy as np
import pytest

from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty


def test_energy_budget_default_zero():
    """energy_budget=0 means unlimited (backward compat)."""
    config = ScenarioConfig(
        name="test", mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY,
    )
    assert config.energy_budget == 0.0


def test_energy_budget_field_exists():
    """energy_budget field can be set."""
    config = ScenarioConfig(
        name="test", mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY, energy_budget=100.0,
    )
    assert config.energy_budget == 100.0


def test_energy_depleted_termination_reason():
    """ENERGY_DEPLETED is a valid termination reason."""
    from uavbench.envs.base import TerminationReason
    assert TerminationReason.ENERGY_DEPLETED == "energy_depleted"
    assert TerminationReason.ENERGY_DEPLETED.value == "energy_depleted"
