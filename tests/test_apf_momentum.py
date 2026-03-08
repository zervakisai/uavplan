"""Tests for APF directional repulsion + momentum (Upgrade 12)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from uavbench.planners.apf import APFPlanner
from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty


def _make_config(wind_speed=0.0, wind_dir_deg=0.0):
    return ScenarioConfig(
        name="test_apf",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=50,
        wind_speed=wind_speed,
        wind_direction_deg=wind_dir_deg,
    )


def test_apf_no_wind_backward_compat():
    """APF with wind_speed=0 produces same behavior as before."""
    config = _make_config(wind_speed=0.0)
    heightmap = np.zeros((50, 50), dtype=np.float32)
    no_fly = np.zeros((50, 50), dtype=bool)
    planner = APFPlanner(heightmap, no_fly, config)
    result = planner.plan((5, 5), (45, 45))
    assert result.success


def test_apf_wind_aware_fields_exist():
    """APF planner stores wind parameters."""
    config = _make_config(wind_speed=2.0, wind_dir_deg=90.0)
    heightmap = np.zeros((50, 50), dtype=np.float32)
    no_fly = np.zeros((50, 50), dtype=bool)
    planner = APFPlanner(heightmap, no_fly, config)
    assert planner._wind_speed == 2.0
    assert abs(planner._wind_dir - math.radians(90.0)) < 1e-6


def test_apf_momentum_alpha():
    """Momentum alpha parameter exists."""
    config = _make_config()
    heightmap = np.zeros((50, 50), dtype=np.float32)
    no_fly = np.zeros((50, 50), dtype=bool)
    planner = APFPlanner(heightmap, no_fly, config)
    assert hasattr(planner, '_momentum_alpha')
    assert planner._momentum_alpha == 0.7


def test_apf_with_wind_still_finds_path():
    """APF with wind parameters still successfully plans."""
    config = _make_config(wind_speed=2.0, wind_dir_deg=45.0)
    heightmap = np.zeros((50, 50), dtype=np.float32)
    no_fly = np.zeros((50, 50), dtype=bool)
    planner = APFPlanner(heightmap, no_fly, config)
    result = planner.plan((5, 5), (45, 45))
    assert result.success
