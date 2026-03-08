"""Visualization renderer (VC-1, VC-2, VZ-1, VZ-3).

Two modes: paper_min (high-DPI PNG) and ops_full (animated GIF).
Renders frames deterministically with path overlays and HUD badges.
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
    COLOR_FIRE_BUFFER,
    COLOR_GOAL,
    COLOR_NFZ,
    COLOR_SMOKE,
    COLOR_START,
    COLOR_TRAFFIC,
    COLOR_TRAJ_BLUE,
    draw_agent,
    draw_fire,
    draw_fire_buffer,
    draw_goal,
    draw_nfz,
    draw_path,
    draw_risk_heatmap,
    draw_smoke,
    draw_start,
    draw_traffic,
    draw_trajectory,
)

# Target max image dimension (~1200px).  Auto-scale replaces old fixed
# _CELL_PX_OPS = 10 which produced 5000px frames on 500×500 maps.
_TARGET_IMG_PX = 1200
_CELL_PX_PAPER = 15   # 300 DPI paper mode

# Landuse color table: code → RGB uint8
_LANDUSE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (232, 224, 208),  # ground / free — warm cream
    1: (58, 125, 68),    # forest — dark green
    2: (232, 224, 208),  # urban — same as ground
    3: (200, 184, 152),  # industrial — light brown
    4: (74, 144, 217),   # water — blue
}
_ROAD_COLOR = np.array([170, 170, 178], dtype=np.uint8)     # asphalt grey (contrast vs cream ground)
_GROUND_COLOR = np.array([232, 224, 208], dtype=np.uint8)   # warm cream


class Renderer:
    """Frame renderer for UAVBench v2 episodes.

    Enforces VC-1 (path visibility), VC-2 (plan badges),
    VZ-3 (determinism).
    """

    def __init__(
        self,
        config: ScenarioConfig,
        mode: Literal["paper_min", "ops_full"] = "ops_full",
    ) -> None:
        self.config = config
        self.mode = mode
        self._map_size = config.map_size
        # Auto-scale: target ~1200px max dimension
        if mode == "paper_min":
            self._cell_px = _CELL_PX_PAPER
        else:
            self._cell_px = max(2, _TARGET_IMG_PX // config.map_size)
        # Basemap cache (static per episode — buildings + landuse + roads)
        self._basemap_cache: np.ndarray | None = None

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

        # Z=1: Base map (cached — static per episode)
        if self._basemap_cache is None:
            landuse_map = state.get("landuse_map")
            roads_mask = state.get("roads_mask")
            self._basemap_cache = self._render_basemap(
                heightmap, H, W, cell, landuse_map, roads_mask,
            )
        frame = self._basemap_cache.copy()

        # Z=3-4: Smoke/Fire buffer/Fire (if dynamic_state provided)
        if dynamic_state is not None:
            fire_mask = dynamic_state.get("fire_mask")
            smoke_mask = dynamic_state.get("smoke_mask")
            # Z=3.5: Smoke overlay (drawn first — lowest z in this group)
            if smoke_mask is not None:
                draw_smoke(frame, smoke_mask, cell)
            # Z=3.8: Fire buffer zone (safety exclusion ring)
            if fire_mask is not None:
                fire_buffer_r = getattr(self.config, "fire_buffer_radius", 3)
                draw_fire_buffer(frame, fire_mask, fire_buffer_r, cell)
            # Z=4: Fire overlay (drawn last — highest z in this group)
            if fire_mask is not None:
                draw_fire(frame, fire_mask, cell)

        # Z=3.2: Risk heatmap overlay (green→yellow→red)
        cost_map = state.get("cost_map")
        if cost_map is not None:
            draw_risk_heatmap(frame, cost_map, cell)

        # Z=5: Dynamic NFZ (restriction zones)
        if dynamic_state is not None:
            nfz_mask = dynamic_state.get("dynamic_nfz_mask")
            if nfz_mask is not None:
                draw_nfz(frame, nfz_mask, cell)

        # Z=6: Traffic closures
        if dynamic_state is not None:
            traffic_mask = dynamic_state.get("traffic_closure_mask")
            if traffic_mask is not None:
                draw_traffic(frame, traffic_mask, cell)
            traffic_occ = dynamic_state.get("traffic_occupancy_mask")
            if traffic_occ is not None:
                draw_traffic(frame, traffic_occ, cell)

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
        # Start uses original start position, NOT current agent position
        start_xy = state.get("start_xy", state.get("agent_xy", (0, 0)))
        agent_xy = state.get("agent_xy", (0, 0))
        goal_xy = state.get("goal_xy", (H - 1, W - 1))
        draw_start(frame, start_xy, cell)
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

        # Z=13: Color legend (both modes)
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
        landuse_map: np.ndarray | None = None,
        roads_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render base map layer (z=1) with landuse, roads, buildings."""
        # Start with ground color
        color_map = np.broadcast_to(
            _GROUND_COLOR, (H, W, 3),
        ).copy()

        # Landuse layer (before roads and buildings)
        if landuse_map is not None:
            for code, rgb in _LANDUSE_COLORS.items():
                mask = landuse_map == code
                if mask.any():
                    color_map[mask] = rgb

        # Roads layer
        if roads_mask is not None and roads_mask.any():
            color_map[roads_mask] = _ROAD_COLOR

        # Buildings: grey, intensity varies with height
        building_mask = heightmap > 0
        if building_mask.any():
            intensity = np.clip(120 - (heightmap * 15).astype(int), 40, 120)
            bldg_colors = np.stack([intensity] * 3, axis=-1).astype(np.uint8)
            color_map[building_mask] = bldg_colors[building_mask]

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
            (COLOR_FIRE_BUFFER, "BUFFER"),
            (COLOR_SMOKE, "SMOKE"),
            (COLOR_NFZ, "NFZ"),
            (COLOR_TRAFFIC, "TRAFFIC"),
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

    def render_briefing_card(
        self,
        heightmap: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        """Render a mission briefing title card (shown before animation).

        Shows mission context: objective, origin, destination, constraints,
        planner name, and a static view of the map with start/goal markers.
        """
        H, W = heightmap.shape
        cell = self._cell_px

        # Render base map (reuse cache if available)
        if self._basemap_cache is not None:
            frame = self._basemap_cache.copy()
        else:
            landuse_map = state.get("landuse_map")
            roads_mask = state.get("roads_mask")
            frame = self._render_basemap(
                heightmap, H, W, cell, landuse_map, roads_mask,
            )

        # Start and goal markers (prominent)
        start_xy = state.get("start_xy", state.get("agent_xy", (0, 0)))
        goal_xy = state.get("goal_xy", (H - 1, W - 1))
        draw_start(frame, start_xy, cell)
        draw_goal(frame, goal_xy, cell)

        # Mission briefing overlay — large dark box with mission details
        img_h, img_w = frame.shape[:2]
        box_h = min(int(img_h * 0.6), 260)
        box_w = min(img_w - 20, img_w)
        box_y = (img_h - box_h) // 2
        box_x = (img_w - box_w) // 2

        # Dark semi-transparent background
        region = frame[box_y:box_y + box_h, box_x:box_x + box_w].astype(np.uint16)
        bg = np.array([15, 20, 35], dtype=np.uint16)
        frame[box_y:box_y + box_h, box_x:box_x + box_w] = (
            (region * 30 + bg * 226) >> 8
        ).astype(np.uint8)

        # Text rendering — scale based on frame width, auto-reduce for long text
        base_scale = max(2, min(img_w // 150, 4))
        tx = box_x + 12
        ty = box_y + 10
        avail_w = box_w - 24  # text area width

        def _auto_scale(text: str, preferred: int) -> int:
            """Reduce scale if text would overflow."""
            char_w = 5  # (CHAR_W + SPACING) pixels per char at scale=1
            needed = len(text) * char_w * preferred
            if needed <= avail_w:
                return preferred
            return max(1, avail_w // (len(text) * char_w))

        # Mission title (large, yellow)
        title = "MISSION BRIEFING"
        s = _auto_scale(title, base_scale)
        line_h = 6 * s + 4
        _render_text(frame, title, tx, ty, (255, 220, 100), s)
        ty += line_h + 2

        # Objective (may need smaller scale for long names)
        obj_label = state.get("objective_label", "UAV Mission")
        s = _auto_scale(obj_label, base_scale)
        line_h = 6 * s + 4
        _render_text(frame, obj_label, tx, ty, (232, 232, 232), s)
        ty += line_h

        # Origin → Destination
        origin = state.get("origin_name", "")
        dest = state.get("destination_name", "")
        detail_scale = _auto_scale(f"FROM: {origin}", base_scale)
        line_h = 6 * detail_scale + 3
        if origin and dest:
            _render_text(frame, f"FROM: {origin}", tx, ty, (180, 200, 220), detail_scale)
            ty += line_h
            _render_text(frame, f"TO:   {dest}", tx, ty, (180, 200, 220), detail_scale)
            ty += line_h

        # Planner + Difficulty
        planner = state.get("planner_name", "")
        difficulty = state.get("difficulty", "")
        diff_tag = f"  [{difficulty.upper()}]" if difficulty else ""
        _render_text(frame, f"PLANNER: {planner}{diff_tag}", tx, ty, (200, 200, 200), detail_scale)
        ty += line_h

        # Deliverable
        deliverable = state.get("deliverable_name", "")
        if deliverable:
            _render_text(frame, f"CARGO: {deliverable}", tx, ty, (180, 200, 220), detail_scale)
            ty += line_h

        # Constraints (if available and space permits)
        constraints = state.get("constraints", [])
        if constraints and ty + line_h < box_y + box_h - 4:
            for c in constraints[:2]:  # max 2 constraints
                if ty + line_h >= box_y + box_h - 4:
                    break
                _render_text(frame, f"! {c}", tx, ty, (213, 94, 0), detail_scale)
                ty += line_h

        # Add legend
        frame = self._render_legend(frame)

        return frame

    def render_fog_comparison(
        self,
        heightmap: np.ndarray,
        state: dict[str, Any],
        fog_state: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> np.ndarray:
        """Render side-by-side fog comparison: agent view vs ground truth.

        Left panel: what the planner sees (fog-filtered state)
        Right panel: actual environment state (ground truth)
        """
        # Render fog view (left)
        fog_frame, _ = self.render_frame(heightmap, state, fog_state)

        # Render ground truth (right) — create state without cost_map for cleaner view
        truth_state = dict(state)
        truth_state.pop("cost_map", None)
        truth_frame, _ = self.render_frame(heightmap, truth_state, ground_truth)

        # Ensure same height
        h1, w1 = fog_frame.shape[:2]
        h2, w2 = truth_frame.shape[:2]
        h = max(h1, h2)
        if h1 < h:
            fog_frame = np.vstack([fog_frame, np.full((h - h1, w1, 3), 255, dtype=np.uint8)])
        if h2 < h:
            truth_frame = np.vstack([truth_frame, np.full((h - h2, w2, 3), 255, dtype=np.uint8)])

        # Divider (2px black line)
        divider = np.zeros((h, 2, 3), dtype=np.uint8)

        # Composite
        combined = np.hstack([fog_frame, divider, truth_frame])

        # Add labels at top
        _render_text(combined, "AGENT VIEW (FOG)", 4, 4, (255, 200, 0), 2)
        _render_text(combined, "GROUND TRUTH", w1 + 6, 4, (0, 200, 0), 2)

        return combined
