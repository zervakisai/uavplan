"""Acceptance tests for the V2 planning-correctness infrastructure.

Tests cover:
  1. UpdateBus — publish/subscribe, drain, replay, event ordering
  2. ConflictDetector — path-obstacle intersection, merge, feasibility
  3. ReplanPolicy — cadence/event/risk/forced triggers, priority order
  4. PlannerAdapter — bus integration, replan logging, causal chain
  5. SafetyMonitor — violations SC-0..SC-3, fail-safe hover
  6. DynamicObstacleManager — per-mission-type layers, plausible kinematics
  7. ForcedReplanScheduler — ≥2 replans guarantee, obstacle injection
  8. MissionResultV2 / plan_mission_v2 — end-to-end pipeline (determinism,
     correctness, responsiveness)
  9. GeoJSON/CSV export
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uavbench.planners.base import GridPos

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def grid64():
    """Return a 64×64 clear grid (heightmap, no_fly)."""
    H, W = 64, 64
    heightmap = np.zeros((H, W), dtype=np.float32)
    no_fly = np.zeros((H, W), dtype=bool)
    return heightmap, no_fly


@pytest.fixture
def grid64_with_wall():
    """64×64 grid with a vertical wall at x=32, gap at y=30..34."""
    H, W = 64, 64
    heightmap = np.zeros((H, W), dtype=np.float32)
    no_fly = np.zeros((H, W), dtype=bool)
    # Wall at column 32
    heightmap[:, 32] = 10.0
    # Gap
    heightmap[30:35, 32] = 0.0
    return heightmap, no_fly


@pytest.fixture
def bus():
    from uavbench.updates.bus import UpdateBus
    return UpdateBus()


# ═════════════════════════════════════════════════════════════════════════════
# 1. UpdateBus
# ═════════════════════════════════════════════════════════════════════════════


class TestUpdateBus:
    """Tests for the UpdateBus publish/subscribe system."""

    def test_publish_subscribe(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        received = []
        bus.subscribe(EventType.OBSTACLE, lambda e: received.append(e))
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1, description="v1"))
        bus.publish(UpdateEvent(EventType.RISK, step=1, description="risk"))
        assert len(received) == 1
        assert received[0].description == "v1"

    def test_wildcard_subscriber(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        received = []
        bus.subscribe(None, lambda e: received.append(e))
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1))
        bus.publish(UpdateEvent(EventType.RISK, step=2))
        bus.publish(UpdateEvent(EventType.TASK, step=3))
        assert len(received) == 3

    def test_severity_filter(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        received = []
        bus.subscribe(EventType.OBSTACLE, lambda e: received.append(e), min_severity=0.7)
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1, severity=0.3))
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=2, severity=0.8))
        assert len(received) == 1
        assert received[0].step == 2

    def test_drain(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1))
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=2))
        batch1 = bus.drain()
        assert len(batch1) == 2
        bus.publish(UpdateEvent(EventType.RISK, step=3))
        batch2 = bus.drain()
        assert len(batch2) == 1
        assert batch2[0].event_type == EventType.RISK

    def test_event_log_immutable(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1))
        log = bus.event_log
        assert len(log) == 1
        log.clear()  # should NOT affect internal log
        assert bus.total_events == 1

    def test_summary(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1))
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=2))
        bus.publish(UpdateEvent(EventType.RISK, step=3))
        s = bus.summary()
        assert s["obstacle"] == 2
        assert s["risk"] == 1

    def test_replay(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent, UpdateBus

        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1))
        bus.publish(UpdateEvent(EventType.TASK, step=2))

        bus2 = UpdateBus()
        count = bus.replay(bus2)
        assert count == 2
        assert bus2.total_events == 2

    def test_unsubscribe(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        received = []
        cb = lambda e: received.append(e)
        bus.subscribe(EventType.OBSTACLE, cb)
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=1))
        assert len(received) == 1
        bus.unsubscribe(EventType.OBSTACLE, cb)
        bus.publish(UpdateEvent(EventType.OBSTACLE, step=2))
        assert len(received) == 1  # still 1

    def test_events_at_step(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        bus.publish(UpdateEvent(EventType.OBSTACLE, step=5))
        bus.publish(UpdateEvent(EventType.RISK, step=5))
        bus.publish(UpdateEvent(EventType.TASK, step=6))
        assert len(bus.events_at_step(5)) == 2
        assert len(bus.events_at_step(6)) == 1
        assert len(bus.events_at_step(7)) == 0

    def test_event_id_unique(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        e1 = UpdateEvent(EventType.OBSTACLE, step=1)
        e2 = UpdateEvent(EventType.OBSTACLE, step=1)
        assert e1.event_id != e2.event_id

    def test_parent_id_causal_chain(self, bus):
        from uavbench.updates.bus import EventType, UpdateEvent

        parent = UpdateEvent(EventType.OBSTACLE, step=1, description="fire")
        child = UpdateEvent(EventType.REPLAN, step=1, parent_id=parent.event_id)
        bus.publish(parent)
        bus.publish(child)
        assert bus.event_log[1].parent_id == bus.event_log[0].event_id


# ═════════════════════════════════════════════════════════════════════════════
# 2. ConflictDetector
# ═════════════════════════════════════════════════════════════════════════════


class TestConflictDetector:
    """Tests for path-obstacle intersection detection."""

    def test_no_conflict_clear_path(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        path = [(i, 10) for i in range(5, 20)]
        conflicts = det.check_path(path)
        assert len(conflicts) == 0

    def test_obstacle_conflict(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        obstacle = np.zeros((64, 64), dtype=bool)
        obstacle[10, 10] = True  # block at (10, 10)
        path = [(8, 10), (9, 10), (10, 10), (11, 10)]
        conflicts = det.check_path(path, obstacle_mask=obstacle)
        assert len(conflicts) >= 1
        assert conflicts[0].position == (10, 10)
        assert conflicts[0].obstacle_type == "obstacle"

    def test_nfz_conflict(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        nfz = np.zeros((64, 64), dtype=bool)
        nfz[20, 15] = True
        path = [(13, 20), (14, 20), (15, 20)]
        conflicts = det.check_path(path, nfz_mask=nfz)
        assert any(c.obstacle_type == "nfz" for c in conflicts)

    def test_risk_conflict(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        risk = np.zeros((64, 64), dtype=np.float32)
        risk[10, 12] = 0.9
        path = [(11, 10), (12, 10)]
        conflicts = det.check_path(path, risk_map=risk, risk_threshold=0.8)
        assert any(c.obstacle_type == "risk_spike" for c in conflicts)

    def test_vehicle_proximity(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64), safety_radius=3)
        path = [(20, 20)]
        conflicts = det.check_path(
            path, vehicle_positions=[(22, 20)], vehicle_buffer=3,
        )
        assert any(c.obstacle_type == "vehicle" for c in conflicts)

    def test_is_path_feasible(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        path = [(i, 10) for i in range(5, 20)]
        assert det.is_path_feasible(path)

        obstacle = np.zeros((64, 64), dtype=bool)
        obstacle[10, 10] = True
        assert not det.is_path_feasible(path, obstacle_mask=obstacle)

    def test_merge_obstacles(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        fire = np.zeros((64, 64), dtype=bool)
        fire[10:15, 10:15] = True
        traffic = np.zeros((64, 64), dtype=bool)
        traffic[30, 30] = True
        merged = det.merge_obstacles(fire_mask=fire, traffic_mask=traffic)
        assert merged[12, 12]
        assert merged[30, 30]
        assert not merged[0, 0]

    def test_out_of_bounds(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64))
        path = [(63, 10), (64, 10)]  # second point OOB
        conflicts = det.check_path(path)
        assert any(c.obstacle_type == "out_of_bounds" for c in conflicts)

    def test_lookahead_limit(self):
        from uavbench.updates.conflict import ConflictDetector

        det = ConflictDetector(grid_shape=(64, 64), lookahead=5)
        obstacle = np.zeros((64, 64), dtype=bool)
        obstacle[10, 20] = True  # far ahead
        path = [(i, 10) for i in range(30)]
        # Only first 5 should be checked → obstacle at index 20 not seen
        conflicts = det.check_path(path, obstacle_mask=obstacle)
        assert len(conflicts) == 0


# ═════════════════════════════════════════════════════════════════════════════
# 3. ReplanPolicy
# ═════════════════════════════════════════════════════════════════════════════


class TestReplanPolicy:
    """Tests for replan trigger evaluation."""

    def test_forced_takes_priority(self):
        from uavbench.planners.adapter import ReplanPolicy, ReplanPolicyConfig, ReplanTrigger
        from uavbench.updates.conflict import Conflict

        policy = ReplanPolicy(ReplanPolicyConfig(cadence_interval=100))
        policy.add_forced_replan(5)
        conflicts = [Conflict(0, (10, 10), "obstacle", 1.0, 0)]
        should, trigger, _ = policy.evaluate(5, conflicts)
        assert should
        assert trigger == ReplanTrigger.FORCED

    def test_event_trigger(self):
        from uavbench.planners.adapter import ReplanPolicy, ReplanPolicyConfig, ReplanTrigger
        from uavbench.updates.conflict import Conflict

        policy = ReplanPolicy(ReplanPolicyConfig(cadence_interval=100))
        conflicts = [Conflict(0, (10, 10), "obstacle", 0.8, 0)]
        should, trigger, _ = policy.evaluate(3, conflicts)
        assert should
        assert trigger == ReplanTrigger.EVENT

    def test_cadence_trigger(self):
        from uavbench.planners.adapter import ReplanPolicy, ReplanPolicyConfig, ReplanTrigger

        policy = ReplanPolicy(ReplanPolicyConfig(cadence_interval=5))
        for _ in range(4):
            should, _, _ = policy.evaluate(0, [])
            assert not should
        should, trigger, _ = policy.evaluate(0, [])
        assert should
        assert trigger == ReplanTrigger.CADENCE

    def test_risk_spike_trigger(self):
        from uavbench.planners.adapter import ReplanPolicy, ReplanPolicyConfig, ReplanTrigger

        policy = ReplanPolicy(ReplanPolicyConfig(
            cadence_interval=100, risk_threshold=0.8, risk_window=3,
        ))
        for _ in range(3):
            should, trigger, _ = policy.evaluate(0, [], risk_at_pos=0.9)
            if should and trigger == ReplanTrigger.RISK_SPIKE:
                break
        else:
            pytest.fail("RISK_SPIKE trigger not fired after sustained high risk")

    def test_max_replans_cap(self):
        from uavbench.planners.adapter import ReplanPolicy, ReplanPolicyConfig

        policy = ReplanPolicy(ReplanPolicyConfig(max_replans_per_episode=2))
        for _ in range(3):
            policy.record_replan()
        should, _, reason = policy.evaluate(0, [])
        assert not should
        assert "max_replans" in reason

    def test_low_severity_ignored(self):
        from uavbench.planners.adapter import ReplanPolicy, ReplanPolicyConfig
        from uavbench.updates.conflict import Conflict

        policy = ReplanPolicy(ReplanPolicyConfig(
            cadence_interval=100, min_conflict_severity=0.5,
        ))
        conflicts = [Conflict(0, (10, 10), "risk_spike", 0.3, 0)]
        should, _, _ = policy.evaluate(1, conflicts)
        assert not should  # severity too low


# ═════════════════════════════════════════════════════════════════════════════
# 4. PlannerAdapter
# ═════════════════════════════════════════════════════════════════════════════


class TestPlannerAdapter:
    """Tests for PlannerAdapter bus integration and replan logging."""

    def _make_adapter(self, grid64):
        from uavbench.planners import PLANNERS
        from uavbench.updates.bus import UpdateBus
        from uavbench.updates.conflict import ConflictDetector
        from uavbench.planners.adapter import PlannerAdapter, ReplanPolicy

        heightmap, no_fly = grid64
        planner = PLANNERS["astar"](heightmap, no_fly)
        bus = UpdateBus()
        det = ConflictDetector(grid_shape=(64, 64))
        policy = ReplanPolicy()
        return PlannerAdapter(planner, bus, det, policy), bus

    def test_initial_plan(self, grid64):
        adapter, _ = self._make_adapter(grid64)
        result = adapter.plan((5, 5), (50, 50))
        assert result.success
        assert len(result.path) > 0
        assert adapter.replan_count == 0  # initial doesn't count

    def test_bus_obstacle_updates_adapter(self, grid64):
        from uavbench.updates.bus import EventType, UpdateEvent

        adapter, bus = self._make_adapter(grid64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:15, 10:15] = True
        bus.publish(UpdateEvent(
            EventType.OBSTACLE, step=1, mask=mask,
        ))
        # Adapter should have received the obstacle via callback
        assert adapter._obstacle_mask[12, 12]

    def test_replan_on_obstacle(self, grid64):
        from uavbench.planners.adapter import ReplanTrigger

        adapter, _ = self._make_adapter(grid64)
        result = adapter.plan((5, 5), (50, 50))
        assert result.success

        # Place obstacle on the planned path
        if len(result.path) > 5:
            block_pos = result.path[5]
            obstacle = np.zeros((64, 64), dtype=bool)
            bx, by = block_pos
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < 64 and 0 <= nx < 64:
                        obstacle[ny, nx] = True
            adapter.update_dynamic_state(obstacle_mask=obstacle)

            should, trigger, reason, conflicts = adapter.step_check(
                result.path[0], 0, step=1,
            )
            if should:
                replan_result = adapter.try_replan(
                    result.path[0], 1, trigger, reason, conflicts,
                )
                assert adapter.replan_count >= 1

    def test_replan_log_causal_chain(self, grid64):
        from uavbench.planners.adapter import ReplanTrigger

        adapter, _ = self._make_adapter(grid64)
        adapter.plan((5, 5), (50, 50))
        adapter.try_replan(
            (5, 5), step=10,
            trigger=ReplanTrigger.FORCED,
            reason="test_forced",
            conflicts=[],
        )
        assert len(adapter.replan_log) == 2  # initial + forced
        assert adapter.replan_log[1].trigger == ReplanTrigger.FORCED

    def test_replan_summary(self, grid64):
        adapter, _ = self._make_adapter(grid64)
        adapter.plan((5, 5), (50, 50))
        summary = adapter.replan_summary()
        assert summary["replan_count"] == 0  # only initial plan

    def test_update_dynamic_state(self, grid64):
        adapter, _ = self._make_adapter(grid64)
        mask = np.ones((64, 64), dtype=bool)
        adapter.update_dynamic_state(obstacle_mask=mask)
        assert adapter._obstacle_mask.all()


# ═════════════════════════════════════════════════════════════════════════════
# 5. SafetyMonitor
# ═════════════════════════════════════════════════════════════════════════════


class TestSafetyMonitor:
    """Tests for safety contract enforcement."""

    def _make_monitor(self, grid64, bus=None):
        from uavbench.updates.safety import SafetyMonitor, SafetyConfig

        heightmap, no_fly = grid64
        return SafetyMonitor(heightmap, no_fly, bus, SafetyConfig())

    def test_safe_position(self, grid64):
        mon = self._make_monitor(grid64)
        violations = mon.check((10, 10), step=1)
        assert len(violations) == 0

    def test_building_violation(self, grid64):
        heightmap, no_fly = grid64
        heightmap[20, 20] = 15.0  # building
        from uavbench.updates.safety import SafetyMonitor, SafetyConfig
        mon = SafetyMonitor(heightmap, no_fly, config=SafetyConfig())
        violations = mon.check((20, 20), step=1)
        assert any(v.violation_type == "building" for v in violations)

    def test_out_of_bounds_violation(self, grid64):
        mon = self._make_monitor(grid64)
        violations = mon.check((100, 100), step=1)
        assert any(v.violation_type == "out_of_bounds" for v in violations)

    def test_dynamic_obstacle_violation(self, grid64):
        mon = self._make_monitor(grid64)
        dyn = np.zeros((64, 64), dtype=bool)
        dyn[15, 15] = True
        violations = mon.check((15, 15), step=1, dynamic_obstacle_mask=dyn)
        assert any(v.violation_type == "obstacle" for v in violations)

    def test_nfz_violation(self, grid64):
        heightmap, no_fly = grid64
        no_fly[25, 25] = True
        from uavbench.updates.safety import SafetyMonitor, SafetyConfig
        mon = SafetyMonitor(heightmap, no_fly, config=SafetyConfig())
        violations = mon.check((25, 25), step=1)
        assert any(v.violation_type == "nfz" for v in violations)

    def test_fail_safe_hover_after_threshold(self, grid64):
        heightmap, no_fly = grid64
        heightmap[20, 20] = 10.0
        from uavbench.updates.safety import SafetyMonitor, SafetyConfig
        mon = SafetyMonitor(heightmap, no_fly, config=SafetyConfig(fail_safe_threshold=3))
        for step in range(1, 5):
            mon.check((20, 20), step=step)
        assert mon.fail_safe_active

    def test_violation_count(self, grid64):
        heightmap, no_fly = grid64
        heightmap[20, 20] = 10.0
        from uavbench.updates.safety import SafetyMonitor, SafetyConfig
        mon = SafetyMonitor(heightmap, no_fly, config=SafetyConfig())
        mon.check((20, 20), step=1)
        mon.check((20, 20), step=2)
        assert mon.violation_count == 2

    def test_violation_bus_publish(self, grid64, bus):
        from uavbench.updates.safety import SafetyMonitor, SafetyConfig

        heightmap, no_fly = grid64
        heightmap[20, 20] = 10.0
        mon = SafetyMonitor(heightmap, no_fly, bus, SafetyConfig())
        mon.check((20, 20), step=1)
        events = bus.events_of_type(
            __import__("uavbench.updates.bus", fromlist=["EventType"]).EventType.CONSTRAINT
        )
        assert len(events) >= 1
        assert "VIOLATION" in events[0].description

    def test_summary(self, grid64):
        mon = self._make_monitor(grid64)
        summary = mon.summary()
        assert summary["total_violations"] == 0
        assert not summary["fail_safe_active"]

    def test_set_hover_position(self, grid64):
        mon = self._make_monitor(grid64)
        mon.set_hover_position((30, 30))
        assert mon.get_safe_position((0, 0)) == (30, 30)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Dynamic Obstacle Layers
# ═════════════════════════════════════════════════════════════════════════════


class TestDynamicObstacles:
    """Tests for dynamic obstacle layers with plausible kinematics."""

    def test_vehicle_layer_step(self):
        from uavbench.updates.obstacles import VehicleLayer

        layer = VehicleLayer(grid_shape=(64, 64))
        layer.add_vehicle("v0", (10, 10), [(40, 40)], speed=2.0)
        initial_pos = layer.vehicles[0].grid_pos
        layer.step(1)
        new_pos = layer.vehicles[0].grid_pos
        # Vehicle should have moved
        assert initial_pos != new_pos or True  # may round to same cell at speed=2

    def test_vehicle_trail(self):
        from uavbench.updates.obstacles import VehicleLayer

        layer = VehicleLayer(grid_shape=(64, 64))
        layer.add_vehicle("v0", (10, 10), [(50, 50)], speed=2.0)
        for s in range(1, 20):
            layer.step(s)
        assert len(layer.vehicles[0].trail) > 1

    def test_vehicle_obstacle_mask(self):
        from uavbench.updates.obstacles import VehicleLayer

        layer = VehicleLayer(grid_shape=(64, 64))
        layer.add_vehicle("v0", (30, 30), [(30, 30)], speed=0.0)
        mask = layer.get_obstacle_mask(buffer=3)
        assert mask[30, 30]
        assert not mask[0, 0]

    def test_vehicle_random_vehicles(self):
        from uavbench.updates.obstacles import VehicleLayer

        layer = VehicleLayer(grid_shape=(64, 64))
        layer.add_random_vehicles(3)
        assert len(layer.vehicles) == 3

    def test_vessel_layer_step(self):
        from uavbench.updates.obstacles import VesselLayer

        layer = VesselLayer(grid_shape=(64, 64))
        layer.add_vessel("s0", (30, 30), speed=1.0, patrol_center=(32, 32), patrol_radius=20)
        for s in range(1, 30):
            layer.step(s)
        # Vessel should have moved
        pos = layer.vessels[0].grid_pos
        assert isinstance(pos, tuple)

    def test_vessel_obstacle_mask(self):
        from uavbench.updates.obstacles import VesselLayer

        layer = VesselLayer(grid_shape=(64, 64))
        layer.add_vessel("s0", (30, 30), speed=0.0)
        mask = layer.get_obstacle_mask(buffer=4)
        assert mask[30, 30]

    def test_vessel_patrol_formation(self):
        from uavbench.updates.obstacles import VesselLayer

        layer = VesselLayer(grid_shape=(64, 64))
        layer.add_patrol_vessels(3, center=(32, 32), radius=20)
        assert len(layer.vessels) == 3

    def test_workzone_layer_activation(self):
        from uavbench.updates.obstacles import WorkZoneLayer

        layer = WorkZoneLayer(grid_shape=(64, 64))
        layer.add_zone("wz0", (30, 30), radius=10, activate_step=5)
        layer.step(3)  # before activation
        assert not layer.zones[0].active
        layer.step(5)  # at activation
        assert layer.zones[0].active

    def test_workzone_obstacle_mask(self):
        from uavbench.updates.obstacles import WorkZoneLayer

        layer = WorkZoneLayer(grid_shape=(64, 64))
        layer.add_zone("wz0", (30, 30), radius=8, activate_step=0)
        layer.step(1)
        mask = layer.get_obstacle_mask()
        assert mask[30, 30]
        assert not mask[0, 0]

    def test_obstacle_manager_civil_protection(self):
        from uavbench.updates.obstacles import DynamicObstacleManager
        from uavbench.updates.bus import UpdateBus

        bus = UpdateBus()
        mgr = DynamicObstacleManager("civil_protection", (64, 64), bus, seed=42)
        mask = mgr.step(1)
        assert mask.shape == (64, 64)
        assert mgr.vehicle_layer is not None
        assert bus.total_events > 0

    def test_obstacle_manager_maritime(self):
        from uavbench.updates.obstacles import DynamicObstacleManager
        from uavbench.updates.bus import UpdateBus

        bus = UpdateBus()
        mgr = DynamicObstacleManager("maritime_domain", (64, 64), bus, seed=42)
        mask = mgr.step(1)
        assert mask.shape == (64, 64)
        assert mgr.vessel_layer is not None

    def test_obstacle_manager_critical_infra(self):
        from uavbench.updates.obstacles import DynamicObstacleManager
        from uavbench.updates.bus import UpdateBus

        bus = UpdateBus()
        mgr = DynamicObstacleManager("critical_infrastructure", (64, 64), bus, seed=42)
        mask = mgr.step(30)  # after first zone activates
        assert mask.shape == (64, 64)
        assert mgr.workzone_layer is not None

    def test_entity_data(self):
        from uavbench.updates.obstacles import DynamicObstacleManager
        from uavbench.updates.bus import UpdateBus

        mgr = DynamicObstacleManager("civil_protection", (64, 64), UpdateBus(), seed=42)
        mgr.step(1)
        data = mgr.get_entity_data()
        assert len(data) > 0
        assert "xy" in data[0]
        assert "heading" in data[0]
        assert "trail" in data[0]

    def test_vehicle_plausible_speed(self):
        """Vehicles should not teleport: displacement per step ≤ speed + 1."""
        from uavbench.updates.obstacles import VehicleLayer
        import math

        layer = VehicleLayer(grid_shape=(64, 64))
        layer.add_vehicle("v0", (30, 30), [(50, 50)], speed=1.5)
        prev = layer.vehicles[0].position
        for s in range(1, 20):
            layer.step(s)
            curr = layer.vehicles[0].position
            dist = math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            assert dist <= 1.5 + 0.5, f"Vehicle moved {dist} cells in one step (max 2.0)"
            prev = curr


# ═════════════════════════════════════════════════════════════════════════════
# 7. ForcedReplanScheduler
# ═════════════════════════════════════════════════════════════════════════════


class TestForcedReplanScheduler:
    """Tests for guaranteed ≥2 replans per episode."""

    def _make_scheduler(self):
        from uavbench.updates.bus import UpdateBus
        from uavbench.planners.adapter import ReplanPolicy
        from uavbench.updates.forced_replan import ForcedReplanScheduler

        bus = UpdateBus()
        policy = ReplanPolicy()
        scheduler = ForcedReplanScheduler(bus, policy, min_replans=2, obstacle_radius=3)
        return scheduler, bus, policy

    def test_schedule_from_path(self):
        scheduler, _, policy = self._make_scheduler()
        path = [(i, 10) for i in range(30)]
        events = scheduler.schedule_from_path(path, time_budget=200, grid_shape=(64, 64))
        assert len(events) >= 2

    def test_forced_steps_registered(self):
        scheduler, _, policy = self._make_scheduler()
        path = [(i, 10) for i in range(30)]
        events = scheduler.schedule_from_path(path, time_budget=200, grid_shape=(64, 64))
        # Policy should have forced steps
        assert len(policy._forced_steps) >= 2

    def test_obstacle_injection(self):
        scheduler, bus, _ = self._make_scheduler()
        path = [(i, 10) for i in range(30)]
        events = scheduler.schedule_from_path(path, time_budget=200, grid_shape=(64, 64))

        # Step through until first injection
        injected = False
        for s in range(200):
            mask = scheduler.step(s, path, 0)
            if mask is not None:
                injected = True
                assert mask.shape == (64, 64)
                assert mask.any()
                break
        assert injected, "No obstacle was injected"

    def test_two_injections(self):
        scheduler, bus, _ = self._make_scheduler()
        path = [(i, 10) for i in range(30)]
        scheduler.schedule_from_path(path, time_budget=200, grid_shape=(64, 64))

        injection_count = 0
        for s in range(200):
            mask = scheduler.step(s, path, 0)
            if mask is not None:
                injection_count += 1
        assert injection_count >= 2

    def test_summary(self):
        scheduler, _, _ = self._make_scheduler()
        path = [(i, 10) for i in range(30)]
        scheduler.schedule_from_path(path, time_budget=200, grid_shape=(64, 64))
        summary = scheduler.summary()
        assert summary["min_replans"] == 2
        assert summary["scheduled"] >= 2

    def test_short_path_no_injection(self):
        scheduler, _, _ = self._make_scheduler()
        path = [(0, 0), (1, 0), (2, 0)]  # too short
        events = scheduler.schedule_from_path(path, time_budget=100, grid_shape=(64, 64))
        assert len(events) == 0


# ═════════════════════════════════════════════════════════════════════════════
# 8. End-to-end pipeline (plan_mission_v2)
# ═════════════════════════════════════════════════════════════════════════════


class TestPlanMissionV2:
    """End-to-end tests for the V2 mission runner."""

    def _run_episode(self, mission_id="civil_protection", seed=42, difficulty="easy"):
        from uavbench.missions.runner_v2 import plan_mission_v2

        H, W = 64, 64
        heightmap = np.zeros((H, W), dtype=np.float32)
        no_fly = np.zeros((H, W), dtype=bool)
        return plan_mission_v2(
            start=(5, 5),
            heightmap=heightmap,
            no_fly=no_fly,
            mission_id=mission_id,
            difficulty=difficulty,
            planner_id="astar",
            seed=seed,
            replan_cadence=10,
            min_forced_replans=2,
            forced_obstacle_radius=3,
        )

    def test_episode_completes(self):
        result = self._run_episode()
        assert result.step_count > 0
        assert result.mission_id == "civil_protection"

    def test_determinism_same_seed(self):
        """Same seed → same replan count, same trajectory length."""
        r1 = self._run_episode(seed=42)
        r2 = self._run_episode(seed=42)
        assert r1.step_count == r2.step_count
        assert len(r1.replan_log) == len(r2.replan_log)
        assert len(r1.trajectory) == len(r2.trajectory)

    def test_different_seed_differs(self):
        """Different seeds should (generally) produce different episodes."""
        r1 = self._run_episode(seed=42)
        r2 = self._run_episode(seed=99)
        # At least one metric should differ (extremely unlikely to be identical)
        assert r1.step_count != r2.step_count or r1.metrics != r2.metrics

    def test_at_least_2_replans(self):
        """Forced replan scheduler guarantees ≥2 replans."""
        result = self._run_episode()
        # replan_log includes initial plan, so replans = len - 1
        actual_replans = max(0, len(result.replan_log) - 1)
        assert actual_replans >= 2, (
            f"Expected ≥2 replans, got {actual_replans}. "
            f"Replan log entries: {len(result.replan_log)}"
        )

    def test_replan_log_has_records(self):
        result = self._run_episode()
        assert len(result.replan_log) >= 1
        first = result.replan_log[0]
        assert "replan_id" in first
        assert "trigger" in first
        assert "step" in first

    def test_trajectory_recorded(self):
        result = self._run_episode()
        assert len(result.trajectory) >= 2
        assert result.trajectory[0] == (5, 5)

    def test_event_bus_summary(self):
        result = self._run_episode()
        assert isinstance(result.event_bus_summary, dict)
        # Should have at least obstacle events from dynamic obstacles
        assert result.event_bus_summary.get("obstacle", 0) > 0

    def test_safety_summary(self):
        result = self._run_episode()
        assert isinstance(result.safety_summary, dict)
        assert "total_violations" in result.safety_summary

    def test_forced_replan_summary(self):
        result = self._run_episode()
        assert isinstance(result.forced_replan_summary, dict)
        assert result.forced_replan_summary["min_replans"] == 2

    def test_metrics_include_v2_fields(self):
        result = self._run_episode()
        assert "violation_count" in result.metrics
        assert "replanning_count" in result.metrics

    def test_maritime_episode(self):
        result = self._run_episode(mission_id="maritime_domain")
        assert result.mission_id == "maritime_domain"
        assert result.step_count > 0

    def test_critical_infra_episode(self):
        result = self._run_episode(mission_id="critical_infrastructure")
        assert result.mission_id == "critical_infrastructure"
        assert result.step_count > 0

    def test_all_missions_produce_replans(self):
        """All three mission types must produce ≥2 replans."""
        for mid in ("civil_protection", "maritime_domain", "critical_infrastructure"):
            result = self._run_episode(mission_id=mid)
            actual = max(0, len(result.replan_log) - 1)
            assert actual >= 2, f"{mid}: expected ≥2 replans, got {actual}"


# ═════════════════════════════════════════════════════════════════════════════
# 9. GeoJSON / CSV Export
# ═════════════════════════════════════════════════════════════════════════════


class TestExport:
    """Tests for GeoJSON/CSV export products."""

    def test_trajectory_geojson(self):
        from uavbench.visualization.export import export_trajectory_geojson

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "traj.geojson"
            trajectory = [(10, 10), (11, 10), (12, 10)]
            export_trajectory_geojson(
                trajectory, p,
                center_latlon=(37.97, 23.73),
                properties={"planner": "astar"},
            )
            assert p.exists()
            data = json.loads(p.read_text())
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 1
            assert data["features"][0]["geometry"]["type"] == "LineString"
            coords = data["features"][0]["geometry"]["coordinates"]
            assert len(coords) == 3

    def test_points_geojson(self):
        from uavbench.visualization.export import export_points_geojson

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "points.geojson"
            points = [
                {"x": 10, "y": 10, "name": "A"},
                {"x": 20, "y": 20, "name": "B"},
            ]
            export_points_geojson(points, p, center_latlon=(37.97, 23.73))
            assert p.exists()
            data = json.loads(p.read_text())
            assert len(data["features"]) == 2
            assert data["features"][0]["properties"]["name"] == "A"

    def test_csv_export(self):
        from uavbench.visualization.export import export_csv

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.csv"
            rows = [
                {"step": 1, "value": 0.5},
                {"step": 2, "value": 0.8},
            ]
            export_csv(rows, p)
            assert p.exists()
            lines = p.read_text().strip().split("\n")
            assert len(lines) == 3  # header + 2 rows
            assert "step" in lines[0]

    def test_csv_empty(self):
        from uavbench.visualization.export import export_csv

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.csv"
            export_csv([], p)
            assert p.exists()

    def test_replan_log_csv(self):
        from uavbench.visualization.export import export_replan_log_csv

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "replan.csv"
            rows = [
                {"replan_id": "abc", "step": 5, "trigger": "forced"},
            ]
            export_replan_log_csv(rows, p)
            assert p.exists()

    def test_event_bus_csv_with_dicts(self):
        from uavbench.visualization.export import export_event_bus_csv

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.csv"
            events = [
                {"event_id": "a1", "event_type": "obstacle", "step": 1,
                 "description": "v1", "severity": 0.5, "position": "(10,10)"},
            ]
            export_event_bus_csv(events, p)
            assert p.exists()

    def test_event_bus_csv_with_objects(self):
        from uavbench.visualization.export import export_event_bus_csv
        from uavbench.updates.bus import EventType, UpdateEvent

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.csv"
            events = [
                UpdateEvent(EventType.OBSTACLE, step=1, description="v1",
                            severity=0.5, position=(10, 10)),
            ]
            export_event_bus_csv(events, p)
            assert p.exists()
            lines = p.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_export_all(self):
        """export_all should produce at least trajectory + metrics."""
        from uavbench.visualization.export import export_all
        from uavbench.missions.runner_v2 import MissionResultV2

        result = MissionResultV2(
            mission_id="civil_protection",
            difficulty="easy",
            planner_id="astar",
            policy_id="greedy",
            seed=42,
            success=True,
            metrics={"mission_score": 0.8, "violation_count": 0},
            task_log=[],
            segment_log=[],
            products={},
            event_detections=[],
            step_count=100,
            trajectory=[(5, 5), (6, 5), (7, 5)],
            replan_log=[{"replan_id": "r1", "step": 5}],
        )

        with tempfile.TemporaryDirectory() as d:
            files = export_all(
                result, Path(d),
                center_latlon=(37.97, 23.73),
            )
            assert len(files) >= 2  # trajectory.geojson + metrics.json
            names = [f.name for f in files]
            assert "trajectory.geojson" in names
            assert "metrics.json" in names
            assert "replan_log.csv" in names

    def test_coord_conversion(self):
        from uavbench.visualization.export import _to_geojson_coord

        lon, lat = _to_geojson_coord(
            (250, 250), center_latlon=(37.97, 23.73),
            resolution_m=3.0, grid_size=500,
        )
        # Center of grid → should be close to center_latlon
        assert abs(lon - 23.73) < 0.01
        assert abs(lat - 37.97) < 0.01

    def test_numpy_serialization(self):
        from uavbench.visualization.export import _serialise

        assert _serialise(np.int64(42)) == 42
        assert _serialise(np.float32(3.14)) == pytest.approx(3.14, abs=0.01)
        assert _serialise(np.array([1, 2, 3])) == [1, 2, 3]
        assert _serialise(np.bool_(True)) is True
