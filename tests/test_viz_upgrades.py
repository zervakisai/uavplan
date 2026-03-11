"""Tests for visualization upgrades (risk heatmap)."""
from __future__ import annotations

import numpy as np
import pytest


def test_draw_risk_heatmap_no_risk():
    """Risk heatmap with zero cost map does nothing."""
    from uavbench.visualization.overlays import draw_risk_heatmap
    frame = np.full((20, 20, 3), 200, dtype=np.uint8)
    original = frame.copy()
    cost_map = np.zeros((10, 10), dtype=np.float32)
    draw_risk_heatmap(frame, cost_map, 2)
    np.testing.assert_array_equal(frame, original)


def test_draw_risk_heatmap_with_risk():
    """Risk heatmap modifies frame where cost > 0."""
    from uavbench.visualization.overlays import draw_risk_heatmap
    frame = np.full((20, 20, 3), 200, dtype=np.uint8)
    original = frame.copy()
    cost_map = np.zeros((10, 10), dtype=np.float32)
    cost_map[5, 5] = 0.8  # high risk cell
    draw_risk_heatmap(frame, cost_map, 2)
    # Frame should be modified at the high-risk cell pixels
    assert not np.array_equal(frame, original)


def test_draw_risk_heatmap_color_gradient():
    """Low risk = green-ish, high risk = red-ish."""
    from uavbench.visualization.overlays import draw_risk_heatmap
    # Low risk
    frame_low = np.full((4, 4, 3), 200, dtype=np.uint8)
    cost_low = np.full((2, 2), 0.1, dtype=np.float32)
    draw_risk_heatmap(frame_low, cost_low, 2, alpha=1.0)
    # High risk
    frame_high = np.full((4, 4, 3), 200, dtype=np.uint8)
    cost_high = np.full((2, 2), 0.9, dtype=np.float32)
    draw_risk_heatmap(frame_high, cost_high, 2, alpha=1.0)
    # Low risk should have more green, high risk more red
    assert frame_low[0, 0, 1] > frame_high[0, 0, 1]  # green channel
    assert frame_low[0, 0, 0] < frame_high[0, 0, 0]  # red channel


