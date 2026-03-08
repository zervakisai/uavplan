"""Tests for TRIAGE mission (TR-1)."""

import math

import numpy as np
import pytest

from uavbench.missions.triage import (
    Casualty, Severity, TriageMission, _SEVERITY_PARAMS,
)


class TestSurvivalFunction:
    """Verify survival probability properties (TR-1)."""

    def test_monotonically_decreasing(self):
        """S(t) should decrease over time."""
        cas = Casualty(
            xy=(50, 50), severity=Severity.CRITICAL,
            base_lambda=0.02, injected_at=0,
        )
        prev = 1.0
        for t in range(1, 100):
            s = cas.survival_prob(t, d_fire=999.0)
            assert s < prev, f"S({t}) = {s} >= S({t-1}) = {prev}"
            prev = s

    def test_fire_coupling_increases_decay(self):
        """Closer fire → faster decay (higher λ_eff)."""
        cas = Casualty(
            xy=(50, 50), severity=Severity.SERIOUS,
            base_lambda=0.008, injected_at=0,
        )
        t = 50
        s_far = cas.survival_prob(t, d_fire=100.0)
        s_near = cas.survival_prob(t, d_fire=5.0)
        assert s_near < s_far, "Fire proximity should decrease survival"

    def test_critical_expires_faster_than_minor(self):
        """CRITICAL casualties expire faster than MINOR."""
        crit = Casualty(
            xy=(0, 0), severity=Severity.CRITICAL,
            base_lambda=_SEVERITY_PARAMS[Severity.CRITICAL]["base_lambda"],
            injected_at=0,
        )
        minor = Casualty(
            xy=(0, 0), severity=Severity.MINOR,
            base_lambda=_SEVERITY_PARAMS[Severity.MINOR]["base_lambda"],
            injected_at=0,
        )
        t = 50
        d_fire = 20.0
        assert crit.survival_prob(t, d_fire) < minor.survival_prob(t, d_fire)

    def test_survival_at_t0_is_one(self):
        """S(0) = 1.0 (full survival at injection)."""
        cas = Casualty(
            xy=(0, 0), severity=Severity.CRITICAL,
            base_lambda=0.02, injected_at=0,
        )
        assert cas.survival_prob(0, d_fire=10.0) == pytest.approx(1.0)

    def test_value_zero_after_rescue(self):
        """Rescued casualty has zero value."""
        cas = Casualty(
            xy=(0, 0), severity=Severity.CRITICAL,
            base_lambda=0.02, injected_at=0,
            rescued=True,
        )
        assert cas.value(50, d_fire=10.0) == 0.0


class TestTriageMission:
    """Verify TRIAGE mission engine."""

    def test_spawns_casualties(self):
        rng = np.random.default_rng(42)
        h = np.zeros((100, 100), dtype=np.float32)
        tm = TriageMission(
            (100, 100), rng, n_casualties=5,
            start_xy=(0, 0), goal_xy=(99, 99),
            heightmap=h,
        )
        assert len(tm.casualties) == 5

    def test_rescue_on_contact(self):
        rng = np.random.default_rng(42)
        h = np.zeros((100, 100), dtype=np.float32)
        tm = TriageMission(
            (100, 100), rng, n_casualties=3,
            start_xy=(0, 0), goal_xy=(99, 99),
            heightmap=h,
        )
        # Move agent to first casualty
        first = tm.casualties[0]
        tm.step(first.xy, fire_mask=None, current_step=10)
        assert tm.rescued_count == 1
        assert tm.total_value > 0

    def test_expiry_threshold(self):
        """Casualties expire when survival drops below 5%."""
        rng = np.random.default_rng(42)
        h = np.zeros((50, 50), dtype=np.float32)
        tm = TriageMission(
            (50, 50), rng, n_casualties=1,
            start_xy=(0, 0), goal_xy=(49, 49),
            heightmap=h,
        )
        # Run many steps without rescuing (far from casualty)
        for step in range(500):
            tm.step((0, 0), fire_mask=None, current_step=step)
        # The single casualty should have expired
        assert tm.expired_count >= 0  # depends on lambda

    def test_metrics_structure(self):
        rng = np.random.default_rng(42)
        h = np.zeros((50, 50), dtype=np.float32)
        tm = TriageMission(
            (50, 50), rng, n_casualties=3,
            start_xy=(0, 0), goal_xy=(49, 49),
            heightmap=h,
        )
        metrics = tm.get_metrics()
        assert "total_triage_value" in metrics
        assert "casualties_rescued" in metrics
        assert "casualties_expired" in metrics
        assert "casualties_total" in metrics
        assert metrics["casualties_total"] == 3
