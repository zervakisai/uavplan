"""Unit tests for CollapseModel (structural debris dynamics)."""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.dynamics.collapse import CollapseModel


@pytest.fixture
def simple_heightmap() -> np.ndarray:
    """10x10 heightmap with a 3x3 building block at (3,3)-(5,5)."""
    h = np.zeros((10, 10), dtype=np.float32)
    h[3:6, 3:6] = 5.0  # building
    return h


def test_no_collapse_without_fire(simple_heightmap: np.ndarray) -> None:
    """Buildings don't collapse without fire exposure."""
    rng = np.random.default_rng(42)
    cm = CollapseModel(simple_heightmap, rng, collapse_delay=5)
    for _ in range(20):
        cm.step(fire_mask=None)
    assert not cm.debris_mask.any()
    assert not cm.collapsed_mask.any()


def test_collapse_after_delay(simple_heightmap: np.ndarray) -> None:
    """Building collapses after collapse_delay steps of fire exposure."""
    rng = np.random.default_rng(42)
    cm = CollapseModel(simple_heightmap, rng, collapse_delay=5, debris_prob=1.0)

    fire = np.zeros((10, 10), dtype=bool)
    fire[3:6, 3:6] = True  # fire covers the building

    # Should not collapse before delay
    for i in range(4):
        cm.step(fire_mask=fire, step_idx=i)
    assert not cm.collapsed_mask.any()

    # Step 5 triggers collapse
    cm.step(fire_mask=fire, step_idx=5)
    assert cm.collapsed_mask.any()
    assert cm.debris_mask.any()


def test_debris_is_permanent(simple_heightmap: np.ndarray) -> None:
    """Once debris spawns, it never disappears."""
    rng = np.random.default_rng(42)
    cm = CollapseModel(simple_heightmap, rng, collapse_delay=3, debris_prob=1.0)

    fire = np.zeros((10, 10), dtype=bool)
    fire[3:6, 3:6] = True

    for i in range(10):
        cm.step(fire_mask=fire, step_idx=i)

    debris_after_collapse = cm.debris_mask.copy()
    assert debris_after_collapse.any()

    # Run 10 more steps with NO fire — debris stays
    for i in range(10, 20):
        cm.step(fire_mask=None, step_idx=i)

    np.testing.assert_array_equal(cm.debris_mask, debris_after_collapse)


def test_debris_not_on_buildings(simple_heightmap: np.ndarray) -> None:
    """Debris scatters to free cells, not onto existing buildings."""
    rng = np.random.default_rng(42)
    cm = CollapseModel(simple_heightmap, rng, collapse_delay=2, debris_prob=1.0)

    fire = np.zeros((10, 10), dtype=bool)
    fire[3:6, 3:6] = True

    for i in range(5):
        cm.step(fire_mask=fire, step_idx=i)

    # Debris should NOT be on building cells
    building_cells = simple_heightmap > 0
    overlap = cm.debris_mask & building_cells
    assert not overlap.any()


def test_determinism(simple_heightmap: np.ndarray) -> None:
    """Same seed → identical collapse and debris (DC-1)."""
    fire = np.zeros((10, 10), dtype=bool)
    fire[3:6, 3:6] = True

    results = []
    for _ in range(2):
        rng = np.random.default_rng(99)
        cm = CollapseModel(simple_heightmap, rng, collapse_delay=3)
        for i in range(10):
            cm.step(fire_mask=fire, step_idx=i)
        results.append(cm.debris_mask.copy())

    np.testing.assert_array_equal(results[0], results[1])


def test_zero_debris_prob(simple_heightmap: np.ndarray) -> None:
    """debris_prob=0 means no debris spawns despite collapse."""
    rng = np.random.default_rng(42)
    cm = CollapseModel(simple_heightmap, rng, collapse_delay=2, debris_prob=0.0)

    fire = np.zeros((10, 10), dtype=bool)
    fire[3:6, 3:6] = True

    for i in range(10):
        cm.step(fire_mask=fire, step_idx=i)

    # Buildings collapsed but no debris
    assert not cm.debris_mask.any()


def test_events_emitted(simple_heightmap: np.ndarray) -> None:
    """Collapse events are emitted when buildings collapse."""
    rng = np.random.default_rng(42)
    cm = CollapseModel(simple_heightmap, rng, collapse_delay=3, debris_prob=1.0)

    fire = np.zeros((10, 10), dtype=bool)
    fire[3:6, 3:6] = True

    all_events = []
    for i in range(10):
        cm.step(fire_mask=fire, step_idx=i)
        all_events.extend(cm.pop_events())

    assert len(all_events) > 0
    assert all(e["type"] == "building_collapse" for e in all_events)
    assert all("x" in e and "y" in e and "step" in e for e in all_events)
