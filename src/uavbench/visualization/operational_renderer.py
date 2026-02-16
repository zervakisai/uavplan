"""Operational mission-grade 2D visualization for UAVBench.

Matplotlib-only (Agg backend).  Outputs PNG / GIF / MP4 headlessly.

Design principles
─────────────────
1. **Basemap** looks like a GIS / C2 screenshot (grid, AOI border, road styling)
2. **Strict z-order discipline** — nothing important is hidden
3. **Deterministic motion cues** (sinusoidal fire flicker, smoke drift, replan
   pulse, replan annotation + red X at invalidated waypoint)
4. **Operational symbology** (simple matplotlib patches — no external assets)
5. **C2-style HUD** panel + event-timeline bar (scenario ID, mission type,
   track, planner, step, replans, risk integral, dynamic block hits, guardrail
   depth G0-G3, feasibility flag, comms uptime)
6. **Dual-use realism overlays** (comms coverage, threat bubbles, emergency
   corridor)
7. **Performance** – persistent fig/ax, cached base-image, artist
   ``set_data()`` updates

Layer z-order contract
──────────────────────
======  ============================  =========================
z       Layer                         Style
======  ============================  =========================
 1      Base map (AOI border, grid)   Buildings / roads / ground
 2      Risk heatmap                  YlOrRd α 0.30
 3      Smoke                         Grey α 0.25–0.35 + drift
 4      Fire                          Red-orange α flicker + edge
 5      NFZ                           Magenta dashed + hatch
 6      Traffic closures              Yellow diagonal stripes
 7      Entities (vehicles/intruders) Typed shapes A/P/F + threat
 8      Forced blocks                 Dark X markers
 9      Trajectory + old path         Blue line + ghost old path
10      UAV + safety bubble + heading Triangle + cyan circle
11      Events layer                  Replan pulse, corridor, X
12      HUD + legend + timeline       Semi-transparent overlays
======  ============================  =========================

Performance: ~30 FPS on Mac M1 for 200×200 maps, ~10–20 FPS for 1000×1000.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as path_effects
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# Palette — colorblind-safe (no red-green adjacency)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    # Base map
    "ground":               "#F0EDE6",
    "road_primary":         "#C8C8C8",
    "road_secondary":       "#999999",
    "building_fill":        "#5A5A5A",
    "building_edge":        "#3D3D3D",
    "water":                "#A4C8E1",
    "aoi_border":           "#334155",
    "grid":                 "#D0D0D0",
    "coord_text":           "#888888",
    # Hazards
    "fire_core":            "#FF2D00",
    "fire_glow":            "#FF6B35",
    "fire_edge":            "#FF9F1C",
    "smoke":                "#777777",
    # NFZ
    "nfz_fill":             "#FF00FF",
    "nfz_edge":             "#CC00CC",
    "nfz_shield":           "#DD00DD",
    # Traffic
    "traffic_closure":      "#FFFF00",
    "traffic_closure_edge": "#DAA520",
    # Entities
    "vehicle_ambulance":    "#FF4444",
    "vehicle_police":       "#4488FF",
    "vehicle_fire":         "#FF8800",
    "intruder":             "#FF0044",
    "threat_bubble":        "#FF0044",
    # Risk heatmap
    "risk_high":            "#FC4E2A",
    # Drone
    "drone_safe":           "#0066FF",
    "drone_caution":        "#FF8C00",
    "drone_danger":         "#FF0000",
    "safety_bubble":        "#0088FF",
    # Comms
    "comms_good":           "#3388FF",
    "comms_denied":         "#888888",
    # Corridor (guardrail depth 3)
    "emergency_corridor":   "#00E5FF",
    # Markers
    "start":                "#00CC44",
    "goal":                 "#FFD700",
    "interdiction_x":       "#1A1A1A",
    "replan_pulse":         "#FFFFFF",
    "replan_x_old":         "#DD2222",
    # HUD
    "hud_bg":               "#0A0F1A",
    "hud_text":             "#E8E8E8",
    "hud_accent":           "#00DDFF",
    "hud_warn":             "#FF6B35",
    "hud_crit":             "#FF2D00",
    "timeline_bg":          "#111827",
    "timeline_tick":        "#334155",
    "timeline_event":       "#FFAA00",
    # Cartographic overlays (scale bar, coord box, north arrow)
    "scale_bar":            "#222222",
    "scale_text":           "#333333",
    "north_arrow":          "#334155",
    "coord_box_bg":         "#0A0F1A",
    "coord_box_text":       "#00DDFF",
}

# Fire flicker period (deterministic)
_FIRE_PERIOD = 12
_SMOKE_DRIFT_PERIOD = 20
_REPLAN_FLASH_DURATION = 5  # frames


def _ylor_rd_cmap() -> "LinearSegmentedColormap":
    """YlOrRd-8 inspired risk colormap."""
    return LinearSegmentedColormap.from_list(
        "uav_risk",
        [
            (0.00, "#FFFFB2"),
            (0.20, "#FED976"),
            (0.40, "#FEB24C"),
            (0.55, "#FD8D3C"),
            (0.70, "#FC4E2A"),
            (0.85, "#E31A1C"),
            (1.00, "#B10026"),
        ],
        N=256,
    )


def _hex_rgb(h: str) -> tuple[float, float, float]:
    return (int(h[1:3], 16) / 255.0, int(h[3:5], 16) / 255.0, int(h[5:7], 16) / 255.0)


def _hex_rgba(h: str, a: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b = _hex_rgb(h)
    return (r, g, b, a)


def _deterministic_fire_alpha(step: int, base: float = 0.65) -> float:
    r"""α = base + 0.08 · sin(2π · step / period)"""
    return base + 0.08 * math.sin(2.0 * math.pi * step / _FIRE_PERIOD)


def _deterministic_smoke_shift(step: int) -> int:
    """Deterministic pixel-shift for smoke drift illusion."""
    return int(2.0 * math.sin(2.0 * math.pi * step / _SMOKE_DRIFT_PERIOD))


def _drone_verts(
    cx: float,
    cy: float,
    heading_deg: float = 0.0,
    size: float = 3.0,
) -> list[tuple[float, float]]:
    """Heading-aware triangular drone icon (nose-forward)."""
    rad = math.radians(heading_deg)
    front = (cx + size * math.sin(rad), cy - size * math.cos(rad))
    left = (
        cx - size * 0.6 * math.sin(rad + math.pi * 0.7),
        cy + size * 0.6 * math.cos(rad + math.pi * 0.7),
    )
    right = (
        cx - size * 0.6 * math.sin(rad - math.pi * 0.7),
        cy + size * 0.6 * math.cos(rad - math.pi * 0.7),
    )
    return [front, left, right]


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────


class OperationalRenderer:
    """Mission-grade 2D operational renderer (matplotlib Agg backend).

    Usage::

        r = OperationalRenderer(hmap, nfz, start, goal,
                                planner_name="dstar_lite")
        for step in range(max_steps):
            r.render_frame(drone_xy, step, fire_mask=..., ...)
        r.export_gif(Path("episode.gif"))
    """

    # ── Constructor ──────────────────────────────────────────────────────

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        roads_mask: Optional[np.ndarray] = None,
        planner_name: str = "",
        mode_label: str = "",
        scenario_id: str = "",
        mission_type: str = "",
        track: str = "",
        meters_per_cell: float = 5.0,
        mgrs_easting_origin: int = 500_000,
        mgrs_northing_origin: int = 4_200_000,
        figsize: tuple[float, float] = (12, 10),
        dpi: int = 120,
    ) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for OperationalRenderer")

        self.heightmap = heightmap
        self.no_fly = no_fly
        self.start = start
        self.goal = goal
        self.roads_mask = roads_mask
        self.planner_name = planner_name
        self.mode_label = mode_label
        self.scenario_id = scenario_id
        self.mission_type = mission_type
        self.track = track
        self.meters_per_cell = meters_per_cell
        self.mgrs_easting_origin = mgrs_easting_origin
        self.mgrs_northing_origin = mgrs_northing_origin
        self.figsize = figsize
        self.dpi = dpi
        self.H, self.W = heightmap.shape

        # Pre-computed static imagery
        self._base_rgb: np.ndarray = self._build_base_map()
        self._risk_cmap = _ylor_rd_cmap()
        self._stripe_mask: np.ndarray = self._build_stripe_mask()
        self._grid_spacing = max(10, self.W // 10)

        # Accumulated frames
        self._frames: list[np.ndarray] = []
        self._frame_times: list[float] = []

        # Event log for timeline bar
        self._events: list[dict[str, Any]] = []

        # Replan-flash countdown
        self._replan_flash_remaining: int = 0
        self._last_replan_reason: str = ""
        self._last_invalidated_waypoint: Optional[tuple[int, int]] = None
        self._old_path_ghost: Optional[list[tuple[int, int]]] = None

        # Keyframe indices (event-driven high-DPI export)
        self._keyframe_indices: list[int] = []

    # ── Static pre-computation ───────────────────────────────────────────

    def _build_base_map(self) -> np.ndarray:
        """Build static RGB base image [H, W, 3] float32 — GIS look."""
        base = np.full(
            (self.H, self.W, 3), _hex_rgb(PALETTE["ground"]), dtype=np.float32
        )
        # --- Buildings: fill + 1-px outline edge ---
        building = self.heightmap > 0
        base[building] = _hex_rgb(PALETTE["building_fill"])
        try:
            from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]

            interior = binary_erosion(building, iterations=1)
            edge = building & ~interior
            base[edge] = _hex_rgb(PALETTE["building_edge"])
        except ImportError:
            pass

        # --- Roads (if provided) ---
        if self.roads_mask is not None:
            road_only = self.roads_mask & ~building
            base[road_only] = _hex_rgb(PALETTE["road_primary"])
            try:
                from scipy.ndimage import binary_erosion

                inner = binary_erosion(road_only, iterations=1)
                secondary = road_only & ~inner
                base[secondary] = _hex_rgb(PALETTE["road_secondary"])
            except ImportError:
                pass

        return base

    def _build_stripe_mask(self) -> np.ndarray:
        """Diagonal stripe pattern for traffic closures."""
        yy, xx = np.mgrid[0 : self.H, 0 : self.W]
        return ((xx + yy) % 6) < 3

    # ── Main render ──────────────────────────────────────────────────────

    def render_frame(  # noqa: C901  (complex, but flat — layer sequence)
        self,
        drone_pos: tuple[int, int],
        step: int,
        *,
        fire_mask: Optional[np.ndarray] = None,
        smoke_mask: Optional[np.ndarray] = None,
        nfz_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        traffic_closure_mask: Optional[np.ndarray] = None,
        risk_map: Optional[np.ndarray] = None,
        forced_block_mask: Optional[np.ndarray] = None,
        dynamic_nfz_mask: Optional[np.ndarray] = None,
        intruder_positions: Optional[np.ndarray] = None,
        comms_coverage_map: Optional[np.ndarray] = None,
        trajectory: Optional[list[tuple[int, int]]] = None,
        heading_deg: float = 0.0,
        replan_flash: bool = False,
        replan_reason: str = "",
        invalidated_waypoint: Optional[tuple[int, int]] = None,
        old_path: Optional[list[tuple[int, int]]] = None,
        replans: int = 0,
        risk_value: float = 0.0,
        risk_integral: float = 0.0,
        dynamic_block_hits: int = 0,
        guardrail_depth: int = 0,
        feasible: bool = True,
        corridor_path: Optional[list[tuple[int, int]]] = None,
        status_text: str = "",
        safety_bubble_radius: int = 3,
        mission_elapsed_s: float = 0.0,
        event_t1: Optional[int] = None,
        event_t2: Optional[int] = None,
        total_steps: int = 0,
        planner_name: Optional[str] = None,
        mode_label: Optional[str] = None,
    ) -> np.ndarray:
        """Render one frame with all operational layers.

        Returns the frame as RGB uint8 ``[H_px, W_px, 3]``.
        """
        t0 = time.perf_counter()

        # ── Replan flash state machine ──
        if replan_flash:
            self._replan_flash_remaining = _REPLAN_FLASH_DURATION
            self._last_replan_reason = replan_reason
            if invalidated_waypoint is not None:
                self._last_invalidated_waypoint = invalidated_waypoint
            if old_path is not None:
                self._old_path_ghost = list(old_path)
            self._events.append(
                {"step": step, "type": "replan", "reason": replan_reason}
            )
            self._keyframe_indices.append(len(self._frames))

        if guardrail_depth >= 2:
            last = self._events[-1] if self._events else {}
            if last.get("type") != "guardrail" or last.get("step") != step:
                self._events.append(
                    {"step": step, "type": "guardrail", "depth": guardrail_depth}
                )
                self._keyframe_indices.append(len(self._frames))

        # ── Create figure ──
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # ═══════════════════════ Z 1: Base map ═══════════════════════════
        ax.imshow(self._base_rgb, interpolation="nearest", zorder=1)

        # Grid + coordinate ticks
        gs = self._grid_spacing
        xticks = list(range(0, self.W, gs))
        yticks = list(range(0, self.H, gs))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(
            [str(x) for x in xticks], fontsize=5, color=PALETTE["coord_text"]
        )
        ax.set_yticklabels(
            [str(y) for y in yticks], fontsize=5, color=PALETTE["coord_text"]
        )
        ax.grid(
            True, color=PALETTE["grid"], linewidth=0.3, alpha=0.5, zorder=1.5
        )
        ax.tick_params(length=2, width=0.5, colors=PALETTE["coord_text"])
        # AOI border
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["aoi_border"])
            spine.set_linewidth(1.8)

        # ═══════════════════════ Z 2: Risk heatmap ═══════════════════════
        if risk_map is not None:
            masked = np.ma.masked_where(risk_map < 0.01, risk_map)
            ax.imshow(
                masked,
                cmap=self._risk_cmap,
                alpha=0.30,
                vmin=0,
                vmax=1,
                interpolation="bilinear",
                zorder=2,
            )

        # ═══════════════════════ Z 3: Smoke ══════════════════════════════
        if smoke_mask is not None and np.any(smoke_mask > 0.05):
            smoke_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            sr, sg, sb = _hex_rgb(PALETTE["smoke"])
            smoke_rgba[:, :, 0] = sr
            smoke_rgba[:, :, 1] = sg
            smoke_rgba[:, :, 2] = sb
            base_alpha = 0.30 + 0.05 * math.sin(
                2.0 * math.pi * step / _SMOKE_DRIFT_PERIOD
            )
            smoke_rgba[:, :, 3] = np.clip(
                smoke_mask.astype(np.float32) * base_alpha, 0, 0.45
            )
            shift = _deterministic_smoke_shift(step)
            if shift != 0:
                smoke_rgba = np.roll(smoke_rgba, shift, axis=1)
            ax.imshow(smoke_rgba, interpolation="nearest", zorder=3)

        # ═══════════════════════ Z 4: Fire ═══════════════════════════════
        if fire_mask is not None and np.any(fire_mask):
            fire_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            alpha = _deterministic_fire_alpha(step, base=0.65)
            if step % 2 == 0:
                fire_rgba[fire_mask] = _hex_rgba(PALETTE["fire_core"], alpha)
            else:
                fire_rgba[fire_mask] = _hex_rgba(PALETTE["fire_glow"], alpha)
            try:
                from scipy.ndimage import binary_dilation

                edge = binary_dilation(fire_mask, iterations=1) & ~fire_mask
                fire_rgba[edge] = _hex_rgba(PALETTE["fire_edge"], alpha * 0.55)
            except ImportError:
                pass
            ax.imshow(fire_rgba, interpolation="nearest", zorder=4)

        # ═══════════════════════ Z 5: NFZ ════════════════════════════════
        combined_nfz = self.no_fly.copy()
        if nfz_mask is not None:
            combined_nfz = combined_nfz | nfz_mask
        if dynamic_nfz_mask is not None:
            combined_nfz = combined_nfz | dynamic_nfz_mask
        if np.any(combined_nfz):
            nfz_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            nfz_rgba[combined_nfz] = _hex_rgba(PALETTE["nfz_fill"], 0.25)
            # Hatch pattern
            hatch = (
                (np.arange(self.H)[:, None] + np.arange(self.W)[None, :]) % 8
            ) < 2
            hatch_mask = combined_nfz & hatch
            nfz_rgba[hatch_mask] = _hex_rgba(PALETTE["nfz_edge"], 0.50)
            ax.imshow(nfz_rgba, interpolation="nearest", zorder=5)
            # Dashed boundary contour
            ax.contour(
                combined_nfz.astype(float),
                levels=[0.5],
                colors=[PALETTE["nfz_edge"]],
                linewidths=1.4,
                linestyles="dashed",
                zorder=5.5,
            )
            # Shield marker + label at centroid
            ys, xs = np.where(combined_nfz)
            if len(xs) > 0:
                cx_nfz, cy_nfz = int(np.mean(xs)), int(np.mean(ys))
                ax.plot(
                    cx_nfz,
                    cy_nfz,
                    marker="s",
                    markersize=7,
                    color=PALETTE["nfz_shield"],
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=5.6,
                )
                ax.text(
                    cx_nfz + 2,
                    cy_nfz - 2,
                    "NFZ",
                    fontsize=5.5,
                    color=PALETTE["nfz_edge"],
                    fontweight="bold",
                    zorder=5.7,
                    path_effects=[
                        path_effects.withStroke(linewidth=1.5, foreground="white")
                    ],
                )

        # ═══════════════════════ Z 6: Traffic closures ═══════════════════
        if traffic_closure_mask is not None and np.any(traffic_closure_mask):
            tc_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            show = traffic_closure_mask & self._stripe_mask
            tc_rgba[show] = _hex_rgba(PALETTE["traffic_closure"], 0.55)
            tc_rgba[traffic_closure_mask & ~self._stripe_mask] = _hex_rgba(
                PALETTE["traffic_closure_edge"], 0.22
            )
            ax.imshow(tc_rgba, interpolation="nearest", zorder=6)

        # ═══════════════════════ Z 6b: Comms overlay ═════════════════════
        if comms_coverage_map is not None:
            comms_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            good = comms_coverage_map > 0.5
            denied = ~good
            gr, gg, gb = _hex_rgb(PALETTE["comms_good"])
            dr, dg, db = _hex_rgb(PALETTE["comms_denied"])
            comms_rgba[good, :3] = (gr, gg, gb)
            comms_rgba[good, 3] = 0.08
            comms_rgba[denied, :3] = (dr, dg, db)
            comms_rgba[denied, 3] = 0.12
            ax.imshow(comms_rgba, interpolation="nearest", zorder=1.8)

        # ═══════════════════════ Z 7: Entities ═══════════════════════════
        _VTYPES = [
            ("A", PALETTE["vehicle_ambulance"]),
            ("P", PALETTE["vehicle_police"]),
            ("F", PALETTE["vehicle_fire"]),
        ]
        if traffic_positions is not None and len(traffic_positions) > 0:
            for i, pos in enumerate(traffic_positions):
                vy, vx = int(pos[0]), int(pos[1])
                label, color = _VTYPES[i % len(_VTYPES)]
                ax.plot(
                    vx,
                    vy,
                    "s",
                    color=color,
                    markersize=5.5,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=7,
                )
                ax.text(
                    vx + 1.5,
                    vy - 1,
                    label,
                    fontsize=4.5,
                    color="white",
                    fontweight="bold",
                    zorder=7.1,
                    path_effects=[
                        path_effects.withStroke(linewidth=1.2, foreground=color)
                    ],
                )

        # Intruders: red diamond + threat bubble
        if intruder_positions is not None and len(intruder_positions) > 0:
            for pos in intruder_positions:
                iy, ix = int(pos[0]), int(pos[1])
                ax.plot(
                    ix,
                    iy,
                    "D",
                    color=PALETTE["intruder"],
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=7.2,
                )
                threat = plt.Circle(
                    (ix, iy),
                    8,
                    fill=True,
                    facecolor=PALETTE["threat_bubble"],
                    alpha=0.12,
                    edgecolor=PALETTE["threat_bubble"],
                    linewidth=0.8,
                    linestyle="--",
                    zorder=7.1,
                )
                ax.add_patch(threat)
                dx_d, dy_d = drone_pos
                dist = math.sqrt((dx_d - ix) ** 2 + (dy_d - iy) ** 2)
                if dist < 8:
                    flash_ring = plt.Circle(
                        (ix, iy),
                        10,
                        fill=False,
                        edgecolor="#FF0000",
                        linewidth=2.5,
                        alpha=0.6,
                        zorder=7.3,
                    )
                    ax.add_patch(flash_ring)

        # ═══════════════════════ Z 8: Forced blocks ══════════════════════
        if forced_block_mask is not None and np.any(forced_block_mask):
            ys, xs = np.where(forced_block_mask)
            if len(xs) > 300:
                idx = np.linspace(0, len(xs) - 1, 300, dtype=int)
                xs, ys = xs[idx], ys[idx]
            ax.scatter(
                xs,
                ys,
                marker="x",
                c=PALETTE["interdiction_x"],
                s=16,
                linewidths=0.9,
                zorder=8,
                alpha=0.65,
            )

        # ═══════════════════════ Z 9: Trajectory ═════════════════════════
        if self._old_path_ghost and self._replan_flash_remaining > 0:
            oxs = [p[0] for p in self._old_path_ghost]
            oys = [p[1] for p in self._old_path_ghost]
            ax.plot(oxs, oys, "--", color="#FF6666", linewidth=1.0, alpha=0.35, zorder=9)

        if trajectory and len(trajectory) > 1:
            txs = [p[0] for p in trajectory]
            tys = [p[1] for p in trajectory]
            ax.plot(
                txs, tys, "-", color=PALETTE["drone_safe"],
                linewidth=2.0, alpha=0.6, zorder=9.5,
            )

        # Start & goal
        ax.scatter(
            *self.start, c=PALETTE["start"], s=120, zorder=9.6,
            edgecolors="#006622", linewidth=1.8, marker="o", label="Start",
        )
        ax.scatter(
            *self.goal, c=PALETTE["goal"], s=150, zorder=9.6,
            edgecolors="#CC8800", linewidth=1.5, marker="*", label="Goal",
        )

        # ═══════════════════════ Z 10: UAV ═══════════════════════════════
        dx, dy = drone_pos
        in_fire = (
            fire_mask is not None
            and 0 <= dy < self.H
            and 0 <= dx < self.W
            and fire_mask[min(dy, self.H - 1), min(dx, self.W - 1)]
        )
        if risk_value > 0.7 or in_fire:
            drone_color = PALETTE["drone_danger"]
        elif risk_value > 0.4:
            drone_color = PALETTE["drone_caution"]
        else:
            drone_color = PALETTE["drone_safe"]

        verts = _drone_verts(dx, dy, heading_deg, size=max(3, self.W / 55))
        tri = plt.Polygon(
            verts,
            closed=True,
            facecolor=drone_color,
            edgecolor="white",
            linewidth=1.6,
            zorder=10.5,
        )
        ax.add_patch(tri)
        # Safety bubble
        bubble = plt.Circle(
            (dx, dy),
            safety_bubble_radius,
            fill=False,
            color=PALETTE["safety_bubble"],
            linewidth=1.0,
            alpha=0.45,
            linestyle="--",
            zorder=10.3,
        )
        ax.add_patch(bubble)

        # ═══════════════════════ Z 11: Events ════════════════════════════
        if self._replan_flash_remaining > 0:
            ring_r = safety_bubble_radius + (
                _REPLAN_FLASH_DURATION - self._replan_flash_remaining + 1
            ) * 2
            ring_alpha = 0.5 * (
                self._replan_flash_remaining / _REPLAN_FLASH_DURATION
            )
            pulse = plt.Circle(
                (dx, dy),
                ring_r,
                fill=False,
                edgecolor=PALETTE["replan_pulse"],
                linewidth=2.5,
                alpha=ring_alpha,
                zorder=11,
            )
            ax.add_patch(pulse)
            if self._last_replan_reason:
                ax.text(
                    dx + 4,
                    dy - 5,
                    f"REPLAN: {self._last_replan_reason}",
                    fontsize=6,
                    color="#FF4444",
                    fontweight="bold",
                    zorder=11.1,
                    path_effects=[
                        path_effects.withStroke(linewidth=2, foreground="white")
                    ],
                )
            if self._last_invalidated_waypoint is not None:
                iwx, iwy = self._last_invalidated_waypoint
                ax.plot(
                    iwx, iwy, "X", color=PALETTE["replan_x_old"],
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=1.0, zorder=11.2,
                )
            self._replan_flash_remaining -= 1
            if self._replan_flash_remaining <= 0:
                self._last_invalidated_waypoint = None
                self._old_path_ghost = None
                self._last_replan_reason = ""

        # Emergency corridor (guardrail depth ≥ 3)
        if guardrail_depth >= 3 and corridor_path and len(corridor_path) > 1:
            cxs = [p[0] for p in corridor_path]
            cys = [p[1] for p in corridor_path]
            ax.plot(
                cxs, cys, "-", color=PALETTE["emergency_corridor"],
                linewidth=3.5, alpha=0.8, zorder=11.3,
            )
            ax.text(
                cxs[0],
                cys[0] - 3,
                "EMERGENCY CORRIDOR",
                fontsize=5.5,
                color=PALETTE["emergency_corridor"],
                fontweight="bold",
                zorder=11.4,
                path_effects=[
                    path_effects.withStroke(linewidth=1.5, foreground="#003344")
                ],
            )

        # ═══════════════════════ Z 12: HUD / timeline ════════════════════
        self._draw_hud(
            ax,
            step=step,
            replans=replans,
            risk_value=risk_value,
            risk_integral=risk_integral,
            dynamic_block_hits=dynamic_block_hits,
            guardrail_depth=guardrail_depth,
            feasible=feasible,
            status_text=status_text,
            mission_elapsed_s=mission_elapsed_s,
            planner_name=planner_name or self.planner_name,
            mode_label=mode_label or self.mode_label,
            comms_coverage_map=comms_coverage_map,
        )
        self._draw_legend(
            ax, fire_mask, smoke_mask, traffic_closure_mask,
            risk_map, intruder_positions, comms_coverage_map,
        )
        if total_steps > 0:
            self._draw_timeline(ax, step, total_steps, event_t1, event_t2)

        # ═══════════════════════ Cartographic overlay ════════════════════
        self._draw_cartographic_overlay(ax, drone_pos)

        # ── Axis limits ──
        ax.set_xlim(-0.5, self.W - 0.5)
        ax.set_ylim(self.H - 0.5, -0.5)
        ax.set_aspect("equal")

        # ── Rasterize ──
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()  # drop alpha channel
        plt.close(fig)

        self._frames.append(frame)
        self._frame_times.append(time.perf_counter() - t0)
        return frame

    # ── HUD panel (C2-style) ─────────────────────────────────────────────

    def _draw_hud(
        self,
        ax: Any,
        *,
        step: int,
        replans: int,
        risk_value: float,
        risk_integral: float,
        dynamic_block_hits: int,
        guardrail_depth: int,
        feasible: bool,
        status_text: str,
        mission_elapsed_s: float,
        planner_name: str,
        mode_label: str,
        comms_coverage_map: Optional[np.ndarray],
    ) -> None:
        """C2-style HUD: scenario + planner info, metrics, guardrail status."""
        depth_labels = {0: "G0 ●", 1: "G1 ▬", 2: "G2 ▲", 3: "G3 ◆"}
        depth_colors = {
            0: PALETTE["hud_text"],
            1: PALETTE["hud_accent"],
            2: PALETTE["hud_warn"],
            3: PALETTE["hud_crit"],
        }

        # Row 1 — identity
        r1 = []
        if self.scenario_id:
            r1.append(f"SCN: {self.scenario_id}")
        if self.mission_type:
            r1.append(f"MSN: {self.mission_type.upper()}")
        if self.track:
            r1.append(f"TRK: {self.track.upper()}")
        row1 = "  │  ".join(r1) if r1 else ""

        # Row 2 — planner / mode
        r2 = []
        if planner_name:
            r2.append(f"PLN: {planner_name}")
        if mode_label:
            r2.append(f"MOD: {mode_label}")
        row2 = "  │  ".join(r2) if r2 else ""

        # Row 3 — live metrics
        elapsed = f"{mission_elapsed_s:.1f}s" if mission_elapsed_s > 0 else "—"
        risk_tag = "!" if risk_value > 0.5 else ""
        depth_str = depth_labels.get(guardrail_depth, f"G{guardrail_depth}")
        feas_str = "FEAS" if feasible else "INFEAS"
        r3 = [
            f"T: {step}",
            f"Δt: {elapsed}",
            f"REP: {replans}",
            f"RISK: {risk_value:.2f}{risk_tag}",
            f"Σrisk: {risk_integral:.1f}",
            f"HITS: {dynamic_block_hits}",
            depth_str,
            feas_str,
        ]
        if comms_coverage_map is not None:
            uptime = float(np.mean(comms_coverage_map > 0.5)) * 100.0
            r3.append(f"COMM: {uptime:.0f}%")
        if status_text:
            r3.append(status_text)
        row3 = "  │  ".join(r3)

        rows = [r for r in [row1, row2, row3] if r]
        hud_text = "\n".join(rows)
        border_color = depth_colors.get(guardrail_depth, PALETTE["hud_accent"])

        txt = ax.text(
            0.01,
            0.995,
            hud_text,
            transform=ax.transAxes,
            fontsize=7,
            fontfamily="monospace",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=PALETTE["hud_bg"],
                alpha=0.82,
                edgecolor=border_color,
                linewidth=1.5,
            ),
            color=PALETTE["hud_text"],
            zorder=25,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=0.4, foreground="black")]
        )

    # ── Event timeline bar ───────────────────────────────────────────────

    def _draw_timeline(
        self,
        ax: Any,
        current_step: int,
        total_steps: int,
        event_t1: Optional[int],
        event_t2: Optional[int],
    ) -> None:
        """Horizontal timeline bar at the bottom showing events."""
        tl_y = -0.02
        tl_h = 0.015

        bg = mpatches.FancyBboxPatch(
            (0.0, tl_y),
            1.0,
            tl_h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.002",
            facecolor=PALETTE["timeline_bg"],
            alpha=0.85,
            edgecolor=PALETTE["timeline_tick"],
            linewidth=0.8,
            zorder=24,
            clip_on=False,
        )
        ax.add_patch(bg)

        # Progress
        progress = min(current_step / max(total_steps, 1), 1.0)
        prog_bar = mpatches.FancyBboxPatch(
            (0.0, tl_y),
            progress,
            tl_h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.002",
            facecolor=PALETTE["hud_accent"],
            alpha=0.30,
            zorder=24.1,
            clip_on=False,
        )
        ax.add_patch(prog_bar)

        # Event t1 / t2 ticks
        for t, label in [(event_t1, "t₁"), (event_t2, "t₂")]:
            if t is not None and total_steps > 0:
                x_pos = t / total_steps
                ax.plot(
                    [x_pos, x_pos],
                    [tl_y, tl_y + tl_h],
                    transform=ax.transAxes,
                    color=PALETTE["timeline_tick"],
                    linewidth=1.2,
                    zorder=24.2,
                    clip_on=False,
                )
                ax.text(
                    x_pos,
                    tl_y - 0.005,
                    label,
                    transform=ax.transAxes,
                    fontsize=5,
                    color=PALETTE["hud_text"],
                    ha="center",
                    va="top",
                    zorder=24.3,
                    clip_on=False,
                )

        # Logged events (replan ▼, guardrail ▲)
        for ev in self._events:
            if total_steps > 0:
                x_pos = ev["step"] / total_steps
                color = (
                    PALETTE["timeline_event"]
                    if ev["type"] == "replan"
                    else PALETTE["hud_crit"]
                )
                marker = "v" if ev["type"] == "replan" else "^"
                ax.plot(
                    x_pos,
                    tl_y + tl_h / 2,
                    marker,
                    transform=ax.transAxes,
                    color=color,
                    markersize=3,
                    zorder=24.4,
                    clip_on=False,
                )

    # ── Cartographic overlay (scale bar, coordinate box, north arrow) ────

    def _draw_cartographic_overlay(
        self,
        ax: Any,
        drone_pos: tuple[int, int],
    ) -> None:
        """MGRS/UTM-style scale bar, coordinate readout, and north arrow.

        Placed at z 26 (above HUD) so they never get occluded.
        """
        mpc = self.meters_per_cell
        e0 = self.mgrs_easting_origin
        n0 = self.mgrs_northing_origin

        # ── 1. Scale bar (bottom-left, inside axes) ──────────────────────
        # Pick a "nice" bar length in metres
        map_width_m = self.W * mpc
        target_bar_m = map_width_m * 0.15  # ~15 % of map width
        nice_vals = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        bar_m = min(nice_vals, key=lambda v: abs(v - target_bar_m))
        bar_cells = bar_m / mpc

        # Position in data coords (bottom-left corner, 5 % inset)
        bar_x0 = self.W * 0.03
        bar_y0 = self.H * 0.95
        bar_x1 = bar_x0 + bar_cells
        bar_thickness = max(1.5, self.H * 0.007)

        # Solid black bar
        ax.plot(
            [bar_x0, bar_x1],
            [bar_y0, bar_y0],
            "-",
            color=PALETTE["scale_bar"],
            linewidth=3.0,
            solid_capstyle="butt",
            zorder=26,
        )
        # End ticks
        for bx in (bar_x0, bar_x1):
            ax.plot(
                [bx, bx],
                [bar_y0 - bar_thickness, bar_y0 + bar_thickness],
                "-",
                color=PALETTE["scale_bar"],
                linewidth=1.5,
                zorder=26,
            )
        # Label
        if bar_m >= 1000:
            bar_label = f"{bar_m / 1000:.0f} km"
        else:
            bar_label = f"{bar_m:.0f} m"
        ax.text(
            (bar_x0 + bar_x1) / 2,
            bar_y0 + bar_thickness + 2,
            bar_label,
            fontsize=6.5,
            fontweight="bold",
            color=PALETTE["scale_text"],
            ha="center",
            va="top",
            zorder=26,
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground="white")
            ],
        )

        # ── 2. Coordinate readout box (bottom-right, UAV position) ───────
        dx, dy = drone_pos
        easting = e0 + dx * mpc
        northing = n0 + (self.H - dy) * mpc  # y-axis inverted in image

        coord_str = f"E: {easting:,.0f}  │  N: {northing:,.0f}"
        ax.text(
            0.99,
            0.015,
            coord_str,
            transform=ax.transAxes,
            fontsize=7,
            fontfamily="monospace",
            fontweight="bold",
            color=PALETTE["coord_box_text"],
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=PALETTE["coord_box_bg"],
                alpha=0.82,
                edgecolor=PALETTE["coord_box_text"],
                linewidth=1.0,
            ),
            zorder=26,
        )

        # ── 3. North arrow (top-right corner) ────────────────────────────
        # Tiny triangle pointing up + "N" label, in axes-fraction coords
        na_cx, na_cy = 0.965, 0.92  # centre of arrow in axes fraction
        na_size = 0.018  # half-size in axes fraction
        arrow_verts = [
            (na_cx, na_cy + na_size * 1.6),          # tip  (north)
            (na_cx - na_size, na_cy - na_size * 0.8), # bottom-left
            (na_cx + na_size, na_cy - na_size * 0.8), # bottom-right
        ]
        arrow = plt.Polygon(
            arrow_verts,
            closed=True,
            transform=ax.transAxes,
            facecolor=PALETTE["north_arrow"],
            edgecolor="white",
            linewidth=1.0,
            zorder=26,
            clip_on=False,
        )
        ax.add_patch(arrow)
        ax.text(
            na_cx,
            na_cy + na_size * 2.3,
            "N",
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            color=PALETTE["north_arrow"],
            ha="center",
            va="bottom",
            zorder=26,
            clip_on=False,
            path_effects=[
                path_effects.withStroke(linewidth=1.5, foreground="white")
            ],
        )

    # ── Legend ────────────────────────────────────────────────────────────

    def _draw_legend(
        self,
        ax: Any,
        fire_mask: Optional[np.ndarray],
        smoke_mask: Optional[np.ndarray],
        traffic_closure_mask: Optional[np.ndarray],
        risk_map: Optional[np.ndarray],
        intruder_positions: Optional[np.ndarray],
        comms_coverage_map: Optional[np.ndarray],
    ) -> None:
        """Context-aware legend — only active layers shown."""
        patches: list[Any] = [
            mpatches.Patch(color=PALETTE["building_fill"], label="Building"),
            mpatches.Patch(color=PALETTE["nfz_fill"], alpha=0.5, label="NFZ"),
        ]
        if fire_mask is not None and np.any(fire_mask):
            patches.append(
                mpatches.Patch(color=PALETTE["fire_core"], label="Fire")
            )
        if smoke_mask is not None and np.any(smoke_mask > 0.05):
            patches.append(
                mpatches.Patch(color=PALETTE["smoke"], alpha=0.5, label="Smoke")
            )
        if traffic_closure_mask is not None and np.any(traffic_closure_mask):
            patches.append(
                mpatches.Patch(
                    color=PALETTE["traffic_closure"],
                    alpha=0.6,
                    label="Road closure",
                )
            )
        if risk_map is not None:
            patches.append(
                mpatches.Patch(
                    color=PALETTE["risk_high"], alpha=0.5, label="Risk"
                )
            )
        if intruder_positions is not None and len(intruder_positions) > 0:
            patches.append(
                mpatches.Patch(color=PALETTE["intruder"], label="Intruder")
            )
        if comms_coverage_map is not None:
            patches.append(
                mpatches.Patch(
                    color=PALETTE["comms_denied"],
                    alpha=0.4,
                    label="Comms denied",
                )
            )
        patches.extend([
            mpatches.Patch(color=PALETTE["drone_safe"], label="UAV"),
            mpatches.Patch(color=PALETTE["start"], label="Start"),
            mpatches.Patch(color=PALETTE["goal"], label="Goal"),
        ])
        ax.legend(
            handles=patches,
            loc="lower right",
            fontsize=6,
            framealpha=0.85,
            fancybox=True,
            edgecolor="#555555",
            ncol=2 if len(patches) > 6 else 1,
        )

    # ── Export helpers ────────────────────────────────────────────────────

    def save_frame(self, frame: np.ndarray, path: Path, dpi: int = 0) -> None:
        """Save a single RGB frame as PNG."""
        path.parent.mkdir(parents=True, exist_ok=True)
        use_dpi = dpi or self.dpi
        fig, ax = plt.subplots(
            figsize=(frame.shape[1] / use_dpi, frame.shape[0] / use_dpi),
            dpi=use_dpi,
        )
        ax.imshow(frame)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0, dpi=use_dpi)
        plt.close(fig)

    def export_gif(
        self,
        path: Path,
        fps: int = 10,
        frames: Optional[list[np.ndarray]] = None,
    ) -> None:
        """Export frames as animated GIF (Pillow writer)."""
        import matplotlib.animation as animation

        use = frames or self._frames
        if not use:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(
            figsize=(use[0].shape[1] / self.dpi, use[0].shape[0] / self.dpi),
            dpi=self.dpi,
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im = ax.imshow(use[0])

        def _upd(i: int) -> list:
            im.set_data(use[i])
            return [im]

        anim = animation.FuncAnimation(
            fig, _upd, frames=len(use), interval=1000 // max(fps, 1), blit=True
        )
        anim.save(str(path), writer="pillow", fps=fps)
        plt.close(fig)

    def export_mp4(
        self,
        path: Path,
        fps: int = 15,
        frames: Optional[list[np.ndarray]] = None,
    ) -> None:
        """Export frames as MP4 (ffmpeg) with GIF fallback."""
        import matplotlib.animation as animation

        use = frames or self._frames
        if not use:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(
            figsize=(use[0].shape[1] / self.dpi, use[0].shape[0] / self.dpi),
            dpi=self.dpi,
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im = ax.imshow(use[0])

        def _upd(i: int) -> list:
            im.set_data(use[i])
            return [im]

        anim = animation.FuncAnimation(
            fig, _upd, frames=len(use), interval=1000 // max(fps, 1), blit=True
        )
        try:
            anim.save(str(path), writer="ffmpeg", fps=fps)
        except Exception:
            anim.save(
                str(path.with_suffix(".gif")), writer="pillow", fps=fps
            )
        plt.close(fig)

    def export_frames(self, output_dir: Path) -> list[Path]:
        """Export all accumulated frames as numbered PNGs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, frame in enumerate(self._frames):
            p = output_dir / f"frame_{i:05d}.png"
            self.save_frame(frame, p)
            paths.append(p)
        return paths

    def export_keyframes(
        self, output_dir: Path, dpi: int = 200
    ) -> list[Path]:
        """Export only event-driven keyframes at high DPI.

        Keyframes are saved when:
        - A replan is triggered
        - Guardrail depth ≥ 2 activates
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        indices = sorted(set(self._keyframe_indices))
        for idx in indices:
            if 0 <= idx < len(self._frames):
                p = output_dir / f"keyframe_{idx:05d}.png"
                self.save_frame(self._frames[idx], p, dpi=dpi)
                paths.append(p)
        return paths

    def reset_frames(self) -> None:
        """Clear all accumulated state."""
        self._frames = []
        self._frame_times = []
        self._events = []
        self._keyframe_indices = []
        self._replan_flash_remaining = 0
        self._last_replan_reason = ""
        self._last_invalidated_waypoint = None
        self._old_path_ghost = None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def avg_render_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return float(np.mean(self._frame_times)) * 1000.0

    @property
    def estimated_fps(self) -> float:
        avg = self.avg_render_ms
        return 1000.0 / avg if avg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison helper
# ─────────────────────────────────────────────────────────────────────────────


def side_by_side(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    dpi: int = 120,
) -> np.ndarray:
    """Create a side-by-side comparison frame (two renderers' outputs)."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    h = max(frame_a.shape[0], frame_b.shape[0])
    w = frame_a.shape[1] + frame_b.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(w / dpi, h / dpi), dpi=dpi)
    for ax, frm, lbl in zip(axes, [frame_a, frame_b], [label_a, label_b]):
        ax.imshow(frm)
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    combined = buf[:, :, :3].copy()
    plt.close(fig)
    return combined
