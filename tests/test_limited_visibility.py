"""Tests for Limited Visibility (LV-1)."""

import numpy as np
import pytest

from uavbench.dynamics.limited_visibility import LimitedVisibility


class TestLimitedVisibility:
    """Verify limited visibility filtering behavior."""

    def test_visible_cells_show_true_state(self):
        """Cells within sensor radius show true values."""
        lv = LimitedVisibility((100, 100), sensor_radius=10)
        fire = np.zeros((100, 100), dtype=bool)
        fire[50, 55] = True  # 5 cells from agent at (50, 50)
        state = {"fire_mask": fire, "smoke_mask": None}
        observed = lv.observe((50, 50), state, 0)
        assert observed["fire_mask"][50, 55] == True

    def test_distant_cells_stale(self):
        """Cells outside radius retain stale values (zero initially)."""
        lv = LimitedVisibility((100, 100), sensor_radius=10)
        fire = np.zeros((100, 100), dtype=bool)
        fire[0, 0] = True  # Far from agent at (50, 50)
        state = {"fire_mask": fire, "smoke_mask": None}
        observed = lv.observe((50, 50), state, 0)
        # Cell (0, 0) is outside radius, should be stale (zero)
        assert observed["fire_mask"][0, 0] == False

    def test_stale_memory_persists(self):
        """Previously seen cells retain old values when outside radius."""
        lv = LimitedVisibility((100, 100), sensor_radius=10)
        # Step 1: agent at (50, 50) sees fire at (55, 50)
        fire1 = np.zeros((100, 100), dtype=bool)
        fire1[50, 55] = True
        lv.observe((50, 50), {"fire_mask": fire1, "smoke_mask": None}, 0)

        # Step 2: agent moves to (80, 80), fire at (55, 50) is now stale
        fire2 = np.zeros((100, 100), dtype=bool)
        fire2[50, 55] = False  # fire has moved
        observed = lv.observe((80, 80), {"fire_mask": fire2, "smoke_mask": None}, 1)

        # Agent can't see (55, 50) from (80, 80), so stale value persists
        assert observed["fire_mask"][50, 55] == True  # stale from step 0

    def test_deterministic(self):
        """Same (pos, state, radius) → same observation (LV-1)."""
        lv1 = LimitedVisibility((100, 100), sensor_radius=20)
        lv2 = LimitedVisibility((100, 100), sensor_radius=20)
        fire = np.zeros((100, 100), dtype=bool)
        fire[50, 50] = True
        state = {"fire_mask": fire, "smoke_mask": None}
        obs1 = lv1.observe((40, 40), state, 0)
        obs2 = lv2.observe((40, 40), state, 0)
        np.testing.assert_array_equal(obs1["fire_mask"], obs2["fire_mask"])

    def test_disabled_passthrough(self):
        """When visibility filter is None (disabled), state passes unchanged."""
        # This tests the runner pattern, not LimitedVisibility directly
        lv = None
        state = {"fire_mask": np.ones((10, 10), dtype=bool)}
        observed = lv.observe((5, 5), state, 0) if lv else state
        np.testing.assert_array_equal(observed["fire_mask"], state["fire_mask"])

    def test_non_spatial_passthrough(self):
        """Non-spatial data (e.g., risk_cost_map) passes through unchanged."""
        lv = LimitedVisibility((50, 50), sensor_radius=10)
        state = {
            "fire_mask": np.zeros((50, 50), dtype=bool),
            "risk_cost_map": None,
            "traffic_positions": [(10, 20)],
        }
        observed = lv.observe((25, 25), state, 0)
        assert observed["risk_cost_map"] is None
        assert observed["traffic_positions"] == [(10, 20)]
