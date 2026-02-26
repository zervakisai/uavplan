"""Visualization renderer (VC-1, VC-2, VC-3, VZ-1, VZ-3).

Two modes: paper_min (high-DPI PNG) and ops_full (animated GIF).
Renders frames deterministically with path overlays, HUD badges,
and forced block lifecycle markers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from uavbench2.scenarios.schema import ScenarioConfig
from uavbench2.visualization.hud import compute_badges, render_hud_text
from uavbench2.visualization.overlays import (
    draw_agent,
    draw_fire,
    draw_forced_blocks,
    draw_goal,
    draw_path,
    draw_start,
    draw_trajectory,
)

# Pixel-per-cell scaling
_CELL_PX_PAPER = 15   # 300 DPI paper mode
_CELL_PX_OPS = 8      # 150 DPI ops mode


class Renderer:
    """Frame renderer for UAVBench v2 episodes.

    Enforces VC-1 (path visibility), VC-2 (plan badges),
    VC-3 (forced block lifecycle), VZ-3 (determinism).
    """

    def __init__(
        self,
        config: ScenarioConfig,
        mode: Literal["paper_min", "ops_full"] = "ops_full",
    ) -> None:
        self.config = config
        self.mode = mode
        self._cell_px = _CELL_PX_PAPER if mode == "paper_min" else _CELL_PX_OPS
        self._map_size = config.map_size

    def render_frame(
        self,
        heightmap: np.ndarray,
        state: dict[str, Any],
        dynamic_state: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Render a single frame.

        Args:
            heightmap: (H, W) terrain array
            state: frame state dict with plan, mission, lifecycle fields
            dynamic_state: optional dynamic layer masks

        Returns:
            (frame, meta) where frame is uint8 (H, W, 3) RGB array
            and meta is a dict with rendering metadata.
        """
        H, W = heightmap.shape
        cell = self._cell_px
        img_h = H * cell
        img_w = W * cell

        # Z=1: Base map
        frame = self._render_basemap(heightmap, H, W, cell)

        # Z=3-4: Fire/smoke (if dynamic_state provided)
        if dynamic_state is not None:
            fire_mask = dynamic_state.get("fire_mask")
            smoke_mask = dynamic_state.get("smoke_mask")
            if fire_mask is not None:
                draw_fire(frame, fire_mask, cell)

        # Z=8: Forced block markers
        forced_active = state.get("forced_block_active", False)
        forced_lifecycle = state.get("forced_block_lifecycle", "none")
        if forced_active and dynamic_state is not None:
            fb_mask = dynamic_state.get("forced_block_mask")
            if fb_mask is not None:
                draw_forced_blocks(frame, fb_mask, cell)

        # Z=9: Trajectory + planned path
        trajectory = state.get("trajectory", [])
        if len(trajectory) > 1:
            draw_trajectory(frame, trajectory, cell)

        plan_path = state.get("plan_path", [])
        plan_len = state.get("plan_len", len(plan_path))
        path_rendered = False

        if plan_len > 1 and len(plan_path) > 1:
            draw_path(frame, plan_path, cell)
            path_rendered = True

        # Z=9.6: Start/Goal markers
        agent_xy = state.get("agent_xy", (0, 0))
        goal_xy = state.get("goal_xy", (H - 1, W - 1))
        draw_start(frame, agent_xy, cell)
        draw_goal(frame, goal_xy, cell)

        # Z=10: Agent icon
        draw_agent(frame, agent_xy, cell)

        # Z=12: HUD badges (compute metadata)
        badges = compute_badges(state)

        # Render HUD text onto frame
        if self.mode == "ops_full":
            render_hud_text(frame, state, badges)
        else:
            # paper_min: minimal HUD (step + planner only)
            render_hud_text(frame, state, badges, minimal=True)

        meta = {
            "path_rendered": path_rendered,
            "plan_badge": badges["plan_badge"],
            "block_badge": badges["block_badge"],
            "mode": self.mode,
            "frame_shape": frame.shape,
        }

        return frame, meta

    def _render_basemap(
        self,
        heightmap: np.ndarray,
        H: int, W: int,
        cell: int,
    ) -> np.ndarray:
        """Render base map layer (z=1)."""
        img_h = H * cell
        img_w = W * cell
        frame = np.full((img_h, img_w, 3), 230, dtype=np.uint8)  # light ground

        # Buildings: dark grey
        for y in range(H):
            for x in range(W):
                if heightmap[y, x] > 0:
                    y0, y1 = y * cell, (y + 1) * cell
                    x0, x1 = x * cell, (x + 1) * cell
                    # Building color: darker with height
                    intensity = max(40, 120 - int(heightmap[y, x] * 15))
                    frame[y0:y1, x0:x1] = [intensity, intensity, intensity]

        return frame
