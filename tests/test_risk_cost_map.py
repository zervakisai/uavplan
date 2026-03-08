"""Tests for compute_risk_cost_map (MP-1)."""

import numpy as np
import pytest

from uavbench.blocking import compute_risk_cost_map
from uavbench.scenarios.schema import (
    Difficulty, MissionType, ScenarioConfig,
)


def _default_config(**overrides):
    defaults = dict(
        name="test",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.MEDIUM,
        map_size=50,
        fire_blocks_movement=True,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


class TestRiskCostMap:
    """Verify risk cost map properties."""

    def test_shape_matches_input(self):
        h = np.zeros((50, 50), dtype=np.float32)
        nf = np.zeros((50, 50), dtype=bool)
        config = _default_config()
        risk = compute_risk_cost_map(h, nf, config)
        assert risk.shape == (50, 50)
        assert risk.dtype == np.float32

    def test_range_zero_to_one(self):
        h = np.zeros((50, 50), dtype=np.float32)
        nf = np.zeros((50, 50), dtype=bool)
        config = _default_config()
        fire = np.zeros((50, 50), dtype=bool)
        fire[25, 25] = True
        dyn = {"fire_mask": fire, "smoke_mask": None,
               "traffic_occupancy_mask": None, "dynamic_nfz_mask": None}
        risk = compute_risk_cost_map(h, nf, config, dyn)
        assert risk.min() >= 0.0
        assert risk.max() <= 1.0

    def test_blocked_cells_risk_one(self):
        h = np.zeros((50, 50), dtype=np.float32)
        h[10, 10] = 5.0  # building
        nf = np.zeros((50, 50), dtype=bool)
        nf[20, 20] = True  # no-fly
        config = _default_config()
        risk = compute_risk_cost_map(h, nf, config)
        assert risk[10, 10] == 1.0
        assert risk[20, 20] == 1.0

    def test_fire_proximity_gradient(self):
        """Risk should decrease with distance from fire."""
        h = np.zeros((100, 100), dtype=np.float32)
        nf = np.zeros((100, 100), dtype=bool)
        config = _default_config(map_size=100)
        fire = np.zeros((100, 100), dtype=bool)
        fire[50, 50] = True
        dyn = {"fire_mask": fire, "smoke_mask": None,
               "traffic_occupancy_mask": None, "dynamic_nfz_mask": None}
        risk = compute_risk_cost_map(h, nf, config, dyn)
        # Risk at fire = 1.0, decreases with distance
        assert risk[50, 50] == 1.0
        assert risk[50, 55] > risk[50, 60]
        assert risk[50, 60] > risk[50, 70]

    def test_no_dynamic_state(self):
        """Without dynamic state, only static obstacles contribute."""
        h = np.zeros((50, 50), dtype=np.float32)
        nf = np.zeros((50, 50), dtype=bool)
        config = _default_config()
        risk = compute_risk_cost_map(h, nf, config)
        # All free cells should be 0
        assert risk.sum() == 0.0

    def test_smoke_contributes(self):
        h = np.zeros((50, 50), dtype=np.float32)
        nf = np.zeros((50, 50), dtype=bool)
        config = _default_config()
        smoke = np.zeros((50, 50), dtype=np.float32)
        smoke[25, 25] = 0.8
        dyn = {"fire_mask": None, "smoke_mask": smoke,
               "traffic_occupancy_mask": None, "dynamic_nfz_mask": None}
        risk = compute_risk_cost_map(h, nf, config, dyn)
        assert risk[25, 25] == pytest.approx(0.4, abs=0.01)  # 0.8 * 0.5
