"""Tests for visualization upgrades (13: fog split-screen, 14: risk heatmap)."""
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


def test_fog_comparison_render():
    """Fog comparison produces a wider frame than single render."""
    from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty
    from uavbench.visualization.renderer import Renderer
    config = ScenarioConfig(
        name="test", mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY, map_size=20,
    )
    renderer = Renderer(config, mode="ops_full")
    heightmap = np.zeros((20, 20), dtype=np.float32)
    state = {"agent_xy": (10, 10), "start_xy": (0, 0), "goal_xy": (19, 19)}
    dyn_state = {}

    single_frame, _ = renderer.render_frame(heightmap, state, dyn_state)
    fog_frame = renderer.render_fog_comparison(heightmap, state, dyn_state, dyn_state)

    # Fog comparison should be roughly 2x wider (two panels + divider)
    assert fog_frame.shape[1] > single_frame.shape[1]
