"""Tests for fire-spawned tasks (Upgrade 11)."""
from __future__ import annotations

import numpy as np
import pytest

from uavbench.dynamics.fire_ca import FireSpreadModel, BURNING, UNBURNED


def test_pop_events_empty_initially():
    """No events before any fire reaches buildings."""
    rng = np.random.default_rng(42)
    model = FireSpreadModel((20, 20), rng, n_ignition=0)
    events = model.pop_events()
    assert events == []


def test_pop_events_clears():
    """pop_events returns and clears the event queue."""
    rng = np.random.default_rng(42)
    model = FireSpreadModel((20, 20), rng, n_ignition=0)
    # Manually inject a fire event
    model._fire_events.append({"type": "building_fire", "x": 5, "y": 5, "step": 0})
    events = model.pop_events()
    assert len(events) == 1
    assert events[0]["type"] == "building_fire"
    # Second call should be empty
    assert model.pop_events() == []


def test_building_fire_event_emitted():
    """Fire reaching a building cell (landuse=2) emits building_fire event."""
    rng = np.random.default_rng(42)
    landuse = np.full((20, 20), 2, dtype=np.int8)  # all urban
    model = FireSpreadModel((20, 20), rng, n_ignition=0, landuse_map=landuse)

    # Place fire next to an unburned urban cell
    model.force_cell_state(10, 10, BURNING)

    # Run many steps to let fire spread
    for _ in range(50):
        model.step()

    # Collect all events
    events = model.pop_events()
    # Should have building fire events (fire spread to urban cells)
    building_fires = [e for e in events if e["type"] == "building_fire"]
    assert len(building_fires) > 0
    # Each event has required fields
    for e in building_fires:
        assert "x" in e and "y" in e and "step" in e


def test_no_event_for_non_building():
    """Fire on non-building cells (landuse=0,1,4) does NOT emit events."""
    rng = np.random.default_rng(42)
    landuse = np.full((20, 20), 0, dtype=np.int8)  # all empty (landuse=0)
    model = FireSpreadModel((20, 20), rng, n_ignition=0, landuse_map=landuse)
    model.force_cell_state(10, 10, BURNING)
    for _ in range(50):
        model.step()
    events = model.pop_events()
    building_fires = [e for e in events if e["type"] == "building_fire"]
    assert len(building_fires) == 0


def test_inject_casualty():
    """TriageMission.inject_casualty adds a new casualty."""
    from uavbench.missions.triage import TriageMission, Severity
    rng = np.random.default_rng(42)
    mission = TriageMission((20, 20), rng, n_casualties=0)
    assert len(mission.casualties) == 0

    mission.inject_casualty((5, 5), Severity.CRITICAL, step=10)
    assert len(mission.casualties) == 1
    cas = mission.casualties[0]
    assert cas.xy == (5, 5)
    assert cas.severity == Severity.CRITICAL
    assert cas.injected_at == 10
