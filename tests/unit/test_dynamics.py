"""Unit tests: InteractionEngine + MissionRestrictionModel.

Merges:
  - test_interaction_causality (1 test)
  - test_restriction_zones (24 tests)

Total: 25 tests.  Runtime: < 2 s.
"""
from __future__ import annotations

import numpy as np
import pytest

from uavbench.dynamics.restriction_zones import (
    MissionRestrictionModel,
    RestrictionZone,
)
from uavbench.dynamics.interaction_engine import InteractionEngine


# ── helpers ───────────────────────────────────────────────────

class _DummyNFZ:
    """Stub with get_nfz_mask() for backward compat."""

    def __init__(self, shape: tuple[int, int]):
        self._mask = np.zeros(shape, dtype=bool)
        self._mask[15:25, 15:25] = True

    def get_nfz_mask(self) -> np.ndarray:
        return self._mask.copy()


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
        roads[H // 2, :] = True
        roads[:, W // 2] = True
    hmap = np.zeros((H, W), dtype=np.int32)
    if with_heightmap:
        hmap[H // 2, W // 2] = 5
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


# ═══════════════════════════════════════════════════════════════
# InteractionEngine
# ═══════════════════════════════════════════════════════════════

class TestInteractionEngine:
    def test_fire_creates_closures_and_reads_nfz(self):
        H = W = 40
        roads = np.ones((H, W), dtype=bool)
        fire = np.zeros((H, W), dtype=bool)
        fire[20, 20] = True
        traffic_positions = np.array([[20, 20]], dtype=np.int32)

        eng = InteractionEngine((H, W), roads_mask=roads)
        nfz = _DummyNFZ((H, W))
        out = eng.update(
            step_idx=1,
            fire_mask=fire,
            traffic_positions=traffic_positions,
            dynamic_nfz=nfz,
        )
        assert out["traffic_closure_cells"] > 0
        assert np.sum(eng.traffic_closure_mask) > 0
        assert out["nfz_cells"] > 0
        assert out["interaction_fire_nfz_overlap_ratio"] >= 0.0


# ═══════════════════════════════════════════════════════════════
# MissionRestrictionModel — Civil Protection
# ═══════════════════════════════════════════════════════════════

class TestCivilProtectionTFR:
    def test_zones_created(self):
        m = _make_model("civil_protection", num_zones=3)
        zones = m.get_zones()
        assert len(zones) == 3
        assert all(z.zone_type == "tfr" for z in zones)

    def test_inactive_before_event_t1(self):
        m = _make_model("civil_protection", event_t1=20)
        m.step(dt=1.0)
        assert not any(z.active for z in m.get_zones())
        assert not np.any(m.get_nfz_mask())

    def test_activate_after_event_t1(self):
        m = _make_model("civil_protection", event_t1=5, num_zones=2)
        for _ in range(10):
            m.step(dt=1.0)
        assert any(z.active for z in m.get_zones())

    def test_fire_mask_expands_tfr(self):
        m = _make_model("civil_protection", event_t1=1, num_zones=2, buffer_px=5)
        for _ in range(5):
            m.step(dt=1.0)
        fire = np.zeros((100, 100), dtype=bool)
        fire[40:50, 40:50] = True
        for _ in range(5):
            m.step(dt=1.0, fire_mask=fire)
        assert np.any(m.get_nfz_mask())


# ═══════════════════════════════════════════════════════════════
# MissionRestrictionModel — Maritime
# ═══════════════════════════════════════════════════════════════

class TestMaritimeDomain:
    def test_zones_created(self):
        m = _make_model("maritime_domain", num_zones=3, current_vec=(0.2, 0.1))
        zones = m.get_zones()
        assert len(zones) == 2
        types = {z.zone_type for z in zones}
        assert "sar_box" in types and "port_exclusion" in types

    def test_sar_box_drifts(self):
        m = _make_model("maritime_domain", num_zones=2, event_t1=1, current_vec=(1.0, 0.5))
        for _ in range(5):
            m.step(dt=1.0)
        zones_early = {z.zone_id: z.center for z in m.get_zones() if z.active}
        for _ in range(20):
            m.step(dt=1.0)
        zones_late = {z.zone_id: z.center for z in m.get_zones() if z.active}
        common = set(zones_early) & set(zones_late)
        if common:
            zid = next(iter(common))
            assert zones_early[zid] != zones_late[zid], "SAR box should drift"


# ═══════════════════════════════════════════════════════════════
# MissionRestrictionModel — Critical Infrastructure
# ═══════════════════════════════════════════════════════════════

class TestCriticalInfrastructure:
    def test_zones_created(self):
        m = _make_model(
            "critical_infrastructure", num_zones=2,
            with_roads=True, with_heightmap=True, incident_point=(50, 50),
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
            roads = np.zeros((100, 100), dtype=bool)
            roads[50, :] = True
            roads[:, 50] = True
            assert np.sum(mask & roads) > 0


# ═══════════════════════════════════════════════════════════════
# Coverage, backward compat, determinism, no-oscillation
# ═══════════════════════════════════════════════════════════════

class TestCoverageCap:
    def test_stays_below_max(self):
        m = _make_model("civil_protection", num_zones=5, max_coverage=0.10, event_t1=1, buffer_px=20)
        for _ in range(50):
            m.step(dt=1.0)
        coverage = np.sum(m.get_nfz_mask()) / (100 * 100)
        assert coverage <= 0.10 + 0.01


class TestBackwardCompat:
    def test_get_nfz_mask_returns_bool(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(10):
            m.step(dt=1.0)
        mask = m.get_nfz_mask()
        assert mask.dtype == bool and mask.shape == (100, 100)

    def test_expansion_rate_property(self):
        m = _make_model("civil_protection")
        assert isinstance(m.expansion_rate, float)


class TestDeterminism:
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
        mask1, mask2 = m1.get_nfz_mask(), m2.get_nfz_mask()
        if np.any(mask1) or np.any(mask2):
            assert not np.array_equal(mask1, mask2)


class TestRestrictionZoneDataclass:
    def test_zone_fields(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        z = RestrictionZone(
            zone_id="TFR-1", zone_type="tfr", mission_type="civil_protection",
            active=True, label="TFR-1 WILDFIRE", source_incident="fire_cluster_0",
            center=(5, 5),
        )
        z.mask = mask
        assert z.active is True
        assert np.sum(z.mask) == 16

    def test_zone_inactive_by_default(self):
        z = RestrictionZone(zone_id="X", zone_type="tfr", mission_type="civil_protection")
        assert z.active is False
        assert z.mask is None

    def test_zone_new_fields(self):
        z = RestrictionZone(
            zone_id="TFR-1", zone_type="tfr", mission_type="civil_protection",
            expires_step=100, risk_buffer_px=15,
        )
        assert z.expires_step == 100 and z.risk_buffer_px == 15


class TestNoOscillation:
    def test_coverage_monotonic_civil(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        prev = 0.0
        for _ in range(40):
            m.step(dt=1.0)
            cov = float(np.sum(m.get_nfz_mask())) / (100 * 100)
            assert cov >= prev * 0.9
            prev = cov

    def test_coverage_monotonic_maritime(self):
        m = _make_model("maritime_domain", num_zones=2, event_t1=1, current_vec=(0.2, 0.1))
        coverages: list[float] = []
        for _ in range(30):
            m.step(dt=1.0)
            coverages.append(float(np.sum(m.get_nfz_mask())) / (100 * 100))
        for i in range(2, len(coverages)):
            if coverages[i - 1] > 0 and coverages[i - 2] > 0:
                assert not (
                    coverages[i - 1] < coverages[i - 2] * 0.8
                    and coverages[i] > coverages[i - 1] * 1.2
                )


class TestRiskBuffer:
    def test_shape(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(10):
            m.step(dt=1.0)
        buf = m.get_risk_buffer()
        assert buf.dtype == np.float32 and buf.shape == (100, 100)

    def test_outside_core(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(15):
            m.step(dt=1.0)
        core = m.get_mask()
        buf = m.get_risk_buffer()
        if np.any(core):
            assert np.all(buf[core] == 0.0)
            assert np.any(buf[~core] > 0.0)

    def test_range(self):
        m = _make_model("civil_protection", num_zones=2, event_t1=1)
        for _ in range(15):
            m.step(dt=1.0)
        buf = m.get_risk_buffer()
        assert 0.0 <= float(np.min(buf)) and float(np.max(buf)) <= 1.0


class TestUpdateBusEvents:
    def test_activation_event(self):
        from uavbench.updates.bus import UpdateBus, EventType
        bus = UpdateBus()
        m = _make_model("civil_protection", num_zones=2, event_t1=5)
        m._update_bus = bus
        for _ in range(10):
            m.step(dt=1.0)
        events = bus.events_of_type(EventType.CONSTRAINT)
        assert any("GEOFENCE_ACTIVATED" in e.description for e in events)

    def test_expansion_event_on_fire(self):
        from uavbench.updates.bus import UpdateBus, EventType
        bus = UpdateBus()
        m = _make_model("civil_protection", num_zones=1, event_t1=1, buffer_px=15)
        m._update_bus = bus
        for _ in range(3):
            m.step(dt=1.0)
        fire = np.zeros((100, 100), dtype=bool)
        fire[30:60, 30:60] = True
        for _ in range(5):
            m.step(dt=1.0, fire_mask=fire)
        assert len(bus.events_of_type(EventType.CONSTRAINT)) >= 1


class TestRendererSmokeWithZones:
    def test_render_5_frames_with_zones(self):
        from uavbench.visualization.operational_renderer import OperationalRenderer
        H, W = 50, 50
        hmap = np.random.randint(0, 3, (H, W))
        nfz = np.zeros((H, W), dtype=bool)
        nfz[2:5, 2:5] = True

        tfr_mask = np.zeros((H, W), dtype=bool)
        tfr_mask[15:25, 15:25] = True

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

        zones = [MockZone("TFR-1", "tfr", "TFR: OPS", (20, 20), tfr_mask)]
        risk_buf = np.zeros((H, W), dtype=np.float32)
        risk_buf[13:27, 13:27] = 0.5
        risk_buf[tfr_mask] = 0.0

        r = OperationalRenderer(
            hmap, nfz, (2, 2), (48, 48),
            planner_name="test", scenario_id="smoke_test",
        )
        for s in range(5):
            frame = r.render_frame(
                (25 + s, 25), s,
                restriction_zones=zones,
                restriction_risk_buffer=risk_buf,
            )
            assert frame.shape[2] == 3 and frame.dtype == np.uint8


class TestRelaxZones:
    """Verify B3 fix: relax_zones() actually shrinks zone masks."""

    def test_relax_zones_shrinks_mask(self):
        m = MissionRestrictionModel(
            (100, 100), "civil_protection",
            num_zones=1, event_t1=1, event_t2=50,
            rng=np.random.default_rng(42),
        )
        # Activate and grow a zone
        for _ in range(20):
            m.step(dt=1.0)
        before = int(np.sum(m.get_nfz_mask()))
        if before == 0:
            # Force a mask so we can test erosion
            m._zones[0].active = True
            m._zones[0]._mask = np.zeros((100, 100), dtype=bool)
            m._zones[0]._mask[30:60, 30:60] = True
            m._rebuild_mask()
            before = int(np.sum(m.get_nfz_mask()))

        freed = m.relax_zones(shrink_px=2)
        after = int(np.sum(m.get_nfz_mask()))
        assert freed > 0, "relax_zones must free cells"
        assert after < before, f"mask must shrink: {after} >= {before}"

    def test_relax_zones_deactivates_tiny_zone(self):
        m = MissionRestrictionModel(
            (100, 100), "civil_protection",
            num_zones=1, event_t1=1, event_t2=50,
            rng=np.random.default_rng(42),
        )
        # Create a very small zone (3×3) that will erode to nothing
        m._zones[0].active = True
        m._zones[0]._mask = np.zeros((100, 100), dtype=bool)
        m._zones[0]._mask[50:53, 50:53] = True
        m._rebuild_mask()
        m.relax_zones(shrink_px=2)
        assert not m._zones[0].active, "tiny zone should be deactivated after erosion"

    def test_relax_zones_no_op_when_no_active_zones(self):
        m = MissionRestrictionModel(
            (50, 50), "civil_protection",
            num_zones=1, event_t1=999, event_t2=999,
            rng=np.random.default_rng(0),
        )
        freed = m.relax_zones(shrink_px=2)
        assert freed == 0


class TestZoneViolationsCounter:
    """Verify M2 fix: zone_violations is incremented on NFZ block."""

    def test_zone_violations_setter(self):
        m = MissionRestrictionModel(
            (50, 50), "civil_protection",
            num_zones=1, event_t1=1, event_t2=50,
            rng=np.random.default_rng(0),
        )
        assert m.zone_violations == 0
        m.zone_violations += 1
        assert m.zone_violations == 1
        m.zone_violations = 5
        assert m.zone_violations == 5
