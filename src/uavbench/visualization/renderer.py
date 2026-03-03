"""Visualization renderer (VC-1, VC-2, VC-3, VZ-1, VZ-3).

Two modes: paper_min (high-DPI PNG) and ops_full (animated GIF).
Renders frames deterministically with path overlays, HUD badges,
and forced block lifecycle markers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from uavbench.scenarios.schema import ScenarioConfig
from uavbench.visualization.hud import (
    _render_text,
    compute_badges,
    render_hud_text,
)
from uavbench.visualization.overlays import (
    COLOR_AGENT,
    COLOR_CYAN,
    COLOR_FIRE,
    COLOR_FORCED_BLOCK,
    COLOR_GOAL,
    COLOR_SMOKE,
    COLOR_START,
    COLOR_TRAJ_BLUE,
    draw_agent,
    draw_fire,
    draw_forced_blocks,
    draw_goal,
    draw_path,
    draw_smoke,
    draw_start,
    draw_trajectory,
)

# Pixel-per-cell scaling
_CELL_PX_PAPER = 15   # 300 DPI paper mode
_CELL_PX_OPS = 10     # ops mode (meets 480px min for 50x50 grid)


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
            if smoke_mask is not None:
                draw_smoke(frame, smoke_mask, cell)

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

        # Z=13: Color legend (paper mode only)
        if self.mode == "paper_min":
            frame = self._render_legend(frame)

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
        """Render base map layer (z=1), vectorized."""
        img_h = H * cell
        img_w = W * cell
        frame = np.full((img_h, img_w, 3), 230, dtype=np.uint8)  # light ground

        # Buildings: intensity varies with height (vectorized)
        building_mask = heightmap > 0
        if building_mask.any():
            intensity = np.clip(120 - (heightmap * 15).astype(int), 40, 120)
            # Create per-cell color array
            color_map = np.where(
                building_mask[:, :, np.newaxis],
                np.stack([intensity] * 3, axis=-1).astype(np.uint8),
                230,
            ).astype(np.uint8)
            # Upscale to pixel resolution
            frame = np.repeat(np.repeat(color_map, cell, axis=0), cell, axis=1)

        return frame

    @staticmethod
    def _render_legend(frame: np.ndarray) -> np.ndarray:
        """Render color legend bar at bottom of frame (paper mode)."""
        H, W = frame.shape[:2]
        legend_h = 20
        legend = np.full((legend_h, W, 3), 255, dtype=np.uint8)

        items = [
            (COLOR_START, "START"),
            (COLOR_GOAL, "GOAL"),
            (COLOR_AGENT, "UAV"),
            (COLOR_CYAN, "PLAN"),
            (COLOR_TRAJ_BLUE, "TRAJ"),
            (COLOR_FIRE, "FIRE"),
            (COLOR_SMOKE, "SMOKE"),
            (COLOR_FORCED_BLOCK, "BLOCK"),
        ]
        x = 4
        swatch_size = 8
        scale = 2
        for color, label in items:
            if x + swatch_size + 4 > W:
                break
            # Draw color swatch
            sy = (legend_h - swatch_size) // 2
            legend[sy:sy + swatch_size, x:x + swatch_size] = color
            # Draw label
            _render_text(legend, label, x + swatch_size + 2, sy, (40, 40, 40), scale)
            x += swatch_size + 4 + len(label) * (4 + 1) * scale + 8

        return np.vstack([frame, legend])
