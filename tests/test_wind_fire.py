"""Tests for wind-driven fire spread (FD-5b, WD-1)."""

import math

import numpy as np
import pytest

from uavbench.dynamics.fire_ca import FireSpreadModel, BURNING, UNBURNED


class TestWindSpread:
    """Verify wind modulation behavior."""

    def _make_model(self, seed=42, wind_speed=0.0, wind_dir=0.0):
        """Helper: create fire model on 100x100 grid with center ignition."""
        rng = np.random.default_rng(seed)
        model = FireSpreadModel(
            map_shape=(100, 100),
            rng=rng,
            n_ignition=0,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
        )
        # Single center ignition
        model.force_cell_state(50, 50, BURNING)
        return model

    def test_wind_zero_equals_isotropic(self):
        """wind_speed=0 → bit-identical to no-wind model."""
        rng1 = np.random.default_rng(42)
        m1 = FireSpreadModel(
            map_shape=(100, 100), rng=rng1, n_ignition=0,
            wind_speed=0.0, wind_direction=0.0,
        )
        m1.force_cell_state(50, 50, BURNING)

        rng2 = np.random.default_rng(42)
        m2 = FireSpreadModel(
            map_shape=(100, 100), rng=rng2, n_ignition=0,
        )
        m2.force_cell_state(50, 50, BURNING)

        for _ in range(30):
            m1.step()
            m2.step()

        np.testing.assert_array_equal(m1.fire_mask, m2.fire_mask)

    def test_downwind_spreads_faster(self):
        """With wind from west (dir=0=East), fire spreads more to the east."""
        model = self._make_model(seed=42, wind_speed=2.0, wind_dir=0.0)
        for _ in range(30):
            model.step()

        fire = model.fire_mask
        center_y, center_x = 50, 50
        # Count burning cells east vs west of center
        east_burn = fire[center_y - 10:center_y + 10, center_x + 1:].sum()
        west_burn = fire[center_y - 10:center_y + 10, :center_x].sum()
        assert east_burn > west_burn, (
            f"Downwind (east={east_burn}) should spread more than upwind (west={west_burn})"
        )

    def test_wind_180_mirrors(self):
        """Rotating wind 180° should mirror the dominant spread direction."""
        m_east = self._make_model(seed=42, wind_speed=2.0, wind_dir=0.0)
        m_west = self._make_model(seed=42, wind_speed=2.0, wind_dir=math.pi)

        for _ in range(30):
            m_east.step()
            m_west.step()

        fire_east = m_east.fire_mask
        fire_west = m_west.fire_mask

        # East wind: more fire east of center
        east_dom = fire_east[45:55, 55:].sum() > fire_east[45:55, :45].sum()
        # West wind: more fire west of center
        west_dom = fire_west[45:55, :45].sum() > fire_west[45:55, 55:].sum()

        assert east_dom, "East wind should spread fire east"
        assert west_dom, "West wind should spread fire west"

    def test_determinism_with_wind(self):
        """Same seed + wind → identical fire across runs (WD-1)."""
        m1 = self._make_model(seed=99, wind_speed=1.5, wind_dir=math.pi / 4)
        m2 = self._make_model(seed=99, wind_speed=1.5, wind_dir=math.pi / 4)

        for _ in range(50):
            m1.step()
            m2.step()

        np.testing.assert_array_equal(m1.fire_mask, m2.fire_mask)
        np.testing.assert_array_equal(m1.smoke_mask, m2.smoke_mask)

    def test_wind_factors_precomputed(self):
        """Wind factors should be 8 values, all positive."""
        model = self._make_model(wind_speed=2.0, wind_dir=0.5)
        assert model._wind_factors.shape == (8,)
        assert all(f > 0 for f in model._wind_factors)

    def test_wind_speed_zero_factors_all_one(self):
        """When wind_speed=0, all wind factors should be 1.0."""
        model = self._make_model(wind_speed=0.0)
        np.testing.assert_array_equal(model._wind_factors, np.ones(8))
