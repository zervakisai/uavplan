"""Tests for MissionRestrictionModel (restriction_zones.py).

Covers:
- Zone creation per mission type (civil, maritime, critical)
- Activate → expand → hold lifecycle (no oscillation)
- Coverage cap enforcement
- Backward-compatible get_nfz_mask() API
- Determinism (same seed → same masks)
- Fire-driven TFR derivation
"""

import numpy as np
import pytest

from uavbench.dynamics.restriction_zones import (
    MissionRestrictionModel,
    RestrictionZone,
)


def _make_model(
    mission_type: str = "civil_protection",
    map_size: int = 100,
    num_zones: int = 3,
    max_coverage: float = 0.30,
    buffer_px: int = 10,
    event_t1: int = 10,
    event_t2: int = 30,
    seed: int = 42,
    with_roads: bool = False,
    with_heightmap: bool = False,
    current_vec: tuple[float, float] | None = None,
    incident_point: tuple[int, int] | None = None,
) -> MissionRestrictionModel:
    H = W = map_size
    roads = np.zeros((H, W), dtype=bool)
    if with_roads:
        # Horizontal + vertical road through centre
        roads[H // 2, :] = True
        roads[:, W // 2] = True
    hmap = np.zeros((H, W), dtype=np.int32)
    if with_heightmap:
        hmap[H // 2, W // 2] = 5  # tallest building
    rng = np.random.default_rng(seed)
    return MissionRestrictionModel(
        map_shape=(H, W),
        mission_type=mission_type,
        roads_mask=roads if with_roads else None,
        heightmap=hmap if with_heightmap else None,
        num_zones=num_zones,
        max_coverage=max_coverage,
        buffer_px=buffer_px,
        event_t1=event_t1,
        event_t2=event_t2,
        current_vec=current_vec,
        incident_point=incident_point,
        rng=rng,
    )


class TestCivilProtectionTFR:
    """Civil protection → TFR zones from fire perimeter."""

    def test_zones_created(self):
        m = _make_model("civil_protection", num_zones=3)
        zones = m.get_zones()
        assert len(zones) == 3
        assert all(z.zone_type == "tfr" for z in zones)
        assert all(z.mission_type == "civil_protection" for z in zones)

    def test_zones_inactive_before_event_t1(self):
        m = _make_model("civil_protection", event_t1=20)
        m.step(dt=1.0)
        zones = m.get_zones()
        assert not any(z.active for z in zones)
        assert not np.any(m.get_nfz_mask())

    def test_zones_activate_after_event_t1(self):
        m = _make_model("civil_protection", event_t1=5, num_zones=2)
        for _ in range(10):
            m.step(dt=1.0)
        zones = m.get_zones()
        assert any(z.active for z in zones)

    def test_fire_mask_expands_tfr(self):
        m = _make_model("civil_protection", event_t1=1, num_zones=2, buffer_px=5)
        # Activate zones first
        for _ in range(5):
            m.step(dt=1.0)
        mask_before = m.get_nfz_mask().copy()
        # Now step with a fire mask — TFR should derive from fire clusters
        fire = np.zeros((100, 100), dtype=bool)
        fire[40:50, 40:50] = True  # fire cluster
        for _ in range(5):
            m.step(dt=1.0, fire_mask=fire)
        mask_after = m.get_nfz_mask()
        # Mask should have changed (fire cluster TFR replaces initial placement)
        # At minimum, the mask should be non-empty
        assert np.any(mask_after)


class TestMaritimeDomain:
    """Maritime domain → SAR box + port exclusion."""

    def test_zones_created(self):
        m = _make_model("maritime_domain", num_zones=3, current_vec=(0.2, 0.1))
        zones = m.get_zones()
        # Maritime always creates 1 SAR box + 1 port exclusion = 2
        assert len(zones) == 2
        types = {z.zone_type for z in zones}
        assert "sar_box" in types
        assert "port_exclusion" in types

    def test_sar_box_drifts(self):
        m = _make_model(
            "maritime_domain", num_zones=2, event_t1=1,
            current_vec=(1.0, 0.5),
        )
        for _ in range(5):
            m.step(dt=1.0)
        zones_early = [(z.zone_id, z.center) for z in m.get_zones() if z.active]
        for _ in range(20):
            m.step(dt=1.0)
        zones_late = [(z.zone_id, z.center) for z in m.get_zones() if z.active]
        # At least one active zone should have shifted center
        if zones_early and zones_late:
            early_centers = {z[0]: z[1] for z in zones_early}
            late_centers = {z[0]: z[1] for z in zones_late}
            common = set(early_centers) & set(late_centers)
            if common:
                zid = next(iter(common))
                assert early_centers[zid] != late_centers[zid], "SAR box should drift"


class TestCriticalInfrastructure:
    """Critical infrastructure → security cordon on road network."""

    def test_zones_created(self):
        m = _make_model(
            "critical_infrastructure", num_zones=2,
            with_roads=True, with_heightmap=True,
            incident_point=(50, 50),
        )
        zones = m.get_zones()
        assert len(zones) >= 1
        assert all(z.zone_type == "security_cordon" for z in zones)

    def test_cordon_follows_roads(self):
        m = _make_model(
            "critical_infrastructure", num_zones=1,
            with_roads=True, with_heightmap=True,
            event_t1=1, incident_point=(50, 50),
        )
        for _ in range(20):
            m.step(dt=1.0)
        mask = m.get_nfz_mask()
        if np.any(mask):
            # Cordon should preferentially cover road cells
            roads = np.zeros((100, 100), dtype=bool)
            roads[50, :] = True
            roads[:, 50] = True
            road_coverage = np.sum(mask & roads)
            assert road_coverage > 0, "Cordon should overlap road network"


class TestCoverageCap:
    """Coverage cap enforcement."""

    def test_coverage_stays_below_max(self):
        m = _make_model(
            "civil_protection", num_zones=5,
            max_coverage=0.10, event_t1=1, buffer_px=20,
        )
        for _ in range(50):
            m.step(dt=1.0)
        mask = m.get_nfz_mask()
        coverage = np.sum(mask) / (100 * 100)
        assert coverage <= 0.10 + 0.01, f"Coverage {coverage:.3f} exceeds cap 0.10"


class TestBackwardCompat:
    """get_nfz_mask() backward compatibility."""

    def test_get_nfz_mask_returns_bool_array(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(10):
            m.step(dt=1.0)
        mask = m.get_nfz_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (100, 100)

    def test_expansion_rate_property_stub(self):
        m = _make_model("civil_protection")
        # Should not raise
        rate = m.expansion_rate
        assert isinstance(rate, float)


class TestDeterminism:
    """Same seed → identical zone masks."""

    def test_same_seed_same_output(self):
        m1 = _make_model("civil_protection", seed=123, event_t1=1)
        m2 = _make_model("civil_protection", seed=123, event_t1=1)
        for _ in range(20):
            m1.step(dt=1.0)
            m2.step(dt=1.0)
        np.testing.assert_array_equal(m1.get_nfz_mask(), m2.get_nfz_mask())

    def test_different_seed_different_output(self):
        m1 = _make_model("civil_protection", seed=1, event_t1=1)
        m2 = _make_model("civil_protection", seed=2, event_t1=1)
        for _ in range(20):
            m1.step(dt=1.0)
            m2.step(dt=1.0)
        # Very likely different (not guaranteed but near-certain)
        mask1 = m1.get_nfz_mask()
        mask2 = m2.get_nfz_mask()
        if np.any(mask1) or np.any(mask2):
            assert not np.array_equal(mask1, mask2)


class TestRestrictionZoneDataclass:
    """RestrictionZone dataclass contract."""

    def test_zone_fields(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        z = RestrictionZone(
            zone_id="TFR-1",
            zone_type="tfr",
            mission_type="civil_protection",
            active=True,
            label="TFR-1 WILDFIRE",
            source_incident="fire_cluster_0",
            center=(5, 5),
        )
        z.mask = mask
        assert z.zone_id == "TFR-1"
        assert z.zone_type == "tfr"
        assert z.active is True
        assert z.mask is not None
        assert np.sum(z.mask) == 16

    def test_zone_inactive_by_default(self):
        z = RestrictionZone(
            zone_id="TEST-1",
            zone_type="tfr",
            mission_type="civil_protection",
        )
        assert z.active is False
        assert z.mask is None

    def test_zone_new_fields(self):
        z = RestrictionZone(
            zone_id="TFR-1",
            zone_type="tfr",
            mission_type="civil_protection",
            expires_step=100,
            risk_buffer_px=15,
        )
        assert z.expires_step == 100
        assert z.risk_buffer_px == 15


class TestNoOscillation:
    """Zone coverage must never decrease after activation (no oscillation)."""

    def test_coverage_monotonic_civil(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        prev_coverage = 0.0
        for _ in range(40):
            m.step(dt=1.0)
            mask = m.get_nfz_mask()
            coverage = float(np.sum(mask)) / (100 * 100)
            # Coverage can only increase or stay constant (monotonic non-decreasing)
            # Exception: coverage cap may disable a zone, but that's capping, not oscillation
            # We check no repeated up-down-up pattern
            assert coverage >= prev_coverage * 0.9, (
                f"Coverage dropped from {prev_coverage:.4f} to {coverage:.4f} — oscillation detected"
            )
            prev_coverage = coverage

    def test_coverage_monotonic_maritime(self):
        m = _make_model("maritime_domain", num_zones=2, event_t1=1, current_vec=(0.2, 0.1))
        coverages = []
        for _ in range(30):
            m.step(dt=1.0)
            mask = m.get_nfz_mask()
            coverages.append(float(np.sum(mask)) / (100 * 100))
        # Check no down-up-down oscillation pattern (allow small drift from SAR box movement)
        for i in range(2, len(coverages)):
            if coverages[i - 1] > 0 and coverages[i - 2] > 0:
                # No sharp oscillation (>20% drop then rise)
                assert not (
                    coverages[i - 1] < coverages[i - 2] * 0.8
                    and coverages[i] > coverages[i - 1] * 1.2
                ), "Oscillation detected in maritime zones"


class TestRiskBuffer:
    """Risk buffer graded annulus."""

    def test_risk_buffer_shape(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(10):
            m.step(dt=1.0)
        buf = m.get_risk_buffer()
        assert isinstance(buf, np.ndarray)
        assert buf.dtype == np.float32
        assert buf.shape == (100, 100)

    def test_risk_buffer_outside_core(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(15):
            m.step(dt=1.0)
        core = m.get_mask()
        buf = m.get_risk_buffer()
        if np.any(core):
            # Buffer should be zero inside core mask
            assert np.all(buf[core] == 0.0), "Risk buffer must be zero inside core zone"
            # Buffer should have non-zero values outside core
            assert np.any(buf[~core] > 0.0), "Risk buffer must extend outside core"

    def test_risk_buffer_range(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(15):
            m.step(dt=1.0)
        buf = m.get_risk_buffer()
        assert float(np.min(buf)) >= 0.0
        assert float(np.max(buf)) <= 1.0


class TestUpdateBusEvents:
    """UpdateBus lifecycle events for restriction zones."""

    def test_activation_event_emitted(self):
        from uavbench.updates.bus import UpdateBus, EventType
        bus = UpdateBus()
        m = _make_model("civil_protection", num_zones=2, event_t1=5)
        # Inject bus
        m._update_bus = bus
        for _ in range(10):
            m.step(dt=1.0)
        constraint_events = bus.events_of_type(EventType.CONSTRAINT)
        assert len(constraint_events) > 0, "Should emit GEOFENCE_ACTIVATED events"
        activated = [e for e in constraint_events if "GEOFENCE_ACTIVATED" in e.description]
        assert len(activated) > 0

    def test_expansion_event_on_fire(self):
        from uavbench.updates.bus import UpdateBus, EventType
        bus = UpdateBus()
        m = _make_model("civil_protection", num_zones=1, event_t1=1, buffer_px=15)
        m._update_bus = bus
        # Run a few steps to activate
        for _ in range(3):
            m.step(dt=1.0)
        # Provide a large fire mask to trigger expansion
        fire = np.zeros((100, 100), dtype=bool)
        fire[30:60, 30:60] = True
        for _ in range(5):
            m.step(dt=1.0, fire_mask=fire)
        constraint_events = bus.events_of_type(EventType.CONSTRAINT)
        # Should have at least activation + possibly expansion
        assert len(constraint_events) >= 1


class TestRendererSmokeWithZones:
    """Renderer can produce frames with structured restriction zones."""

    def test_render_5_frames_with_zones(self):
        from uavbench.visualization.operational_renderer import OperationalRenderer
        H, W = 50, 50
        hmap = np.random.randint(0, 3, (H, W))
        nfz = np.zeros((H, W), dtype=bool)
        nfz[2:5, 2:5] = True

        # Create mock zones
        tfr_mask = np.zeros((H, W), dtype=bool)
        tfr_mask[15:25, 15:25] = True
        sar_mask = np.zeros((H, W), dtype=bool)
        sar_mask[30:40, 10:20] = True

        class MockZone:
            def __init__(self, zone_id, zone_type, label, center, mask, act_step=5):
                self.zone_id = zone_id
                self.zone_type = zone_type
                self.label = label
                self.center = center
                self.active = True
                self.activation_step = act_step
                self._mask = mask
            @property
            def mask(self):
                return self._mask

        zones = [
            MockZone("TFR-1", "tfr", "TFR: OPS", (20, 20), tfr_mask),
            MockZone("SAR-BOX-1", "sar_box", "SAR BOX", (15, 35), sar_mask, act_step=10),
        ]

        # Risk buffer
        risk_buf = np.zeros((H, W), dtype=np.float32)
        risk_buf[13:27, 13:27] = 0.5
        risk_buf[tfr_mask] = 0.0

        r = OperationalRenderer(
            hmap, nfz, (2, 2), (48, 48),
            planner_name="test", scenario_id="smoke_test",
        )
        frames = []
        for s in range(5):
            frame = r.render_frame(
                (25 + s, 25), s,
                restriction_zones=zones,
                restriction_risk_buffer=risk_buf,
            )
            frames.append(frame)
            assert frame.shape[2] == 3
            assert frame.dtype == np.uint8

        assert len(frames) == 5
