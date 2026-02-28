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
 4.9    Static no-fly                 Crimson solid + no hatch
 5      Dynamic restriction zones     Per-type: TFR/SAR/cordon
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

from uavbench.visualization.basemap_style import (
    BasemapStyleConfig,
    DEFAULT_STYLE,
    build_styled_basemap,
)


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
    # Static no-fly (permanent, map-baked) — crimson
    "nfz_fill":             "#CC2222",
    "nfz_edge":             "#991111",
    "nfz_shield":           "#FF3333",
    # Dynamic restriction zones — per-type (distinct from fire's red-orange)
    "zone_tfr_fill":        "#FF8C00",   # TFR (orange)
    "zone_tfr_edge":        "#CC6600",
    "zone_tfr_label":       "#FFB347",
    "zone_sar_fill":        "#1E90FF",   # SAR box (blue)
    "zone_sar_edge":        "#1565C0",
    "zone_sar_label":       "#64B5F6",
    "zone_port_fill":       "#1E90FF",   # Port exclusion (blue, same family)
    "zone_port_edge":       "#1565C0",
    "zone_port_label":      "#64B5F6",
    "zone_cordon_fill":     "#9B59B6",   # Security cordon (purple)
    "zone_cordon_edge":     "#7D3C98",
    "zone_cordon_label":    "#BB8FCE",
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


# Cache the version pin string (computed once at import time)
_VERSION_PIN: str | None = None


def _get_version_pin() -> str:
    """Return version string with git commit hash for artifact provenance."""
    global _VERSION_PIN
    if _VERSION_PIN is not None:
        return _VERSION_PIN
    version = "1.0.0"
    try:
        import importlib.metadata
        version = importlib.metadata.version("uavbench")
    except Exception:
        pass
    commit = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except Exception:
        pass
    _VERSION_PIN = f"UAVBench v{version} \u2022 commit {commit}"
    return _VERSION_PIN


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
        landuse_map: Optional[np.ndarray] = None,
        osm_tile_id: Optional[str] = None,
        basemap_style: Optional[BasemapStyleConfig] = None,
        mission_pois: Optional[list[dict[str, Any]]] = None,
        active_incidents: Optional[list[str]] = None,
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
        self.landuse_map = landuse_map
        self.osm_tile_id = osm_tile_id
        self.basemap_style = basemap_style or DEFAULT_STYLE
        self._mission_pois = mission_pois or []
        self._active_incidents = active_incidents or []
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

        # Icon library for mission POIs
        try:
            from uavbench.visualization.icons.library import IconLibrary, IconID
            self._icon_lib = IconLibrary(icon_size=max(18, self.W // 12))
            self._IconID = IconID
        except ImportError:
            self._icon_lib = None
            self._IconID = None

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
        """Build static RGB base image [H, W, 3] float32 — GIS look.

        Delegates to :func:`build_styled_basemap` for landuse, road tiers,
        hillshade, and water coastlines.
        """
        return build_styled_basemap(
            self.heightmap,
            self.roads_mask,
            landuse_map=self.landuse_map,
            style=self.basemap_style,
        )

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
        restriction_zones: Optional[list] = None,
        restriction_risk_buffer: Optional[np.ndarray] = None,
        intruder_positions: Optional[np.ndarray] = None,
        comms_coverage_map: Optional[np.ndarray] = None,
        trajectory: Optional[list[tuple[int, int]]] = None,
        planned_path: Optional[list[tuple[int, int]]] = None,
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
        plan_len: int = 0,
        plan_stale: bool = False,
        plan_reason: str = "none",
        forced_block_active: bool = False,
        forced_block_cleared: bool = False,
        # Mission HUD overlay
        mission_objective: str = "",
        mission_destination: str = "",
        mission_status: str = "",
        distance_to_goal: int = 0,
        mission_max_steps: int = 0,
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

        # ═══════════════════════ Z 4.9: Static no-fly (permanent) ══════
        if np.any(self.no_fly):
            snfz_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            snfz_rgba[self.no_fly] = _hex_rgba(PALETTE["nfz_fill"], 0.35)
            ax.imshow(snfz_rgba, interpolation="nearest", zorder=4.9)
            ax.contour(
                self.no_fly.astype(float),
                levels=[0.5],
                colors=[PALETTE["nfz_edge"]],
                linewidths=1.8,
                linestyles="solid",
                zorder=4.95,
            )
            # "PERM. NFZ" label at centroid
            ys_s, xs_s = np.where(self.no_fly)
            if len(xs_s) > 0:
                cx_s, cy_s = int(np.mean(xs_s)), int(np.mean(ys_s))
                ax.text(
                    cx_s, cy_s, "PERM. NFZ",
                    fontsize=6, fontweight="bold",
                    color=PALETTE["nfz_shield"],
                    ha="center", va="center", zorder=4.97,
                    path_effects=[
                        path_effects.withStroke(linewidth=2.0, foreground="black")
                    ],
                )

        # ═══════════════════════ Z 5: Dynamic restriction zones ════════
        # Per-zone typed rendering with distinct colors and hatch patterns.
        # Zone-type visual spec:
        #   TFR:             orange, solid border, 45-degree hatch
        #   SAR box:         blue, dashed border, horizontal hatch
        #   Port exclusion:  blue, solid border, vertical hatch
        #   Security cordon: purple, solid border, diagonal-back hatch
        _zone_styles = {
            "tfr": {
                "fill": PALETTE["zone_tfr_fill"],
                "edge": PALETTE["zone_tfr_edge"],
                "label_color": PALETTE["zone_tfr_label"],
                "linestyle": "solid",
                "hatch_fn": lambda h, w: ((np.arange(h)[:, None] + np.arange(w)[None, :]) % 5) < 2,
            },
            "sar_box": {
                "fill": PALETTE["zone_sar_fill"],
                "edge": PALETTE["zone_sar_edge"],
                "label_color": PALETTE["zone_sar_label"],
                "linestyle": "dashed",
                "hatch_fn": lambda h, w: np.broadcast_to((np.arange(h)[:, None] % 4) < 2, (h, w)),
            },
            "port_exclusion": {
                "fill": PALETTE["zone_port_fill"],
                "edge": PALETTE["zone_port_edge"],
                "label_color": PALETTE["zone_port_label"],
                "linestyle": "solid",
                "hatch_fn": lambda h, w: np.broadcast_to((np.arange(w)[None, :] % 4) < 2, (h, w)),
            },
            "security_cordon": {
                "fill": PALETTE["zone_cordon_fill"],
                "edge": PALETTE["zone_cordon_edge"],
                "label_color": PALETTE["zone_cordon_label"],
                "linestyle": "solid",
                "hatch_fn": lambda h, w: ((np.arange(h)[:, None] - np.arange(w)[None, :]) % 5) < 2,
            },
        }
        # Risk buffer gradient (faint warning halo UNDER zone cores)
        if restriction_risk_buffer is not None and np.any(restriction_risk_buffer > 0):
            buf_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
            mask_buf = restriction_risk_buffer > 0.01
            buf_vals = restriction_risk_buffer[mask_buf]
            buf_rgba[mask_buf, 0] = 1.0    # warm tint
            buf_rgba[mask_buf, 1] = 0.7
            buf_rgba[mask_buf, 2] = 0.3
            buf_rgba[mask_buf, 3] = buf_vals * 0.15  # very faint
            ax.imshow(buf_rgba, interpolation="nearest", zorder=4.85)

        _rendered_zones = False
        if restriction_zones:
            for zi, zone in enumerate(restriction_zones):
                if not getattr(zone, "active", False):
                    continue
                zmask = getattr(zone, "mask", None)
                if zmask is None or not np.any(zmask):
                    continue
                _rendered_zones = True
                ztype = getattr(zone, "zone_type", "tfr")
                style = _zone_styles.get(ztype, _zone_styles["tfr"])
                z_base = 5.0 + zi * 0.05  # stagger z-order per zone

                # Fill
                z_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
                z_rgba[zmask] = _hex_rgba(style["fill"], 0.35)
                # Typed hatch
                hatch = style["hatch_fn"](self.H, self.W)
                z_rgba[zmask & hatch] = _hex_rgba(style["edge"], 0.55)
                ax.imshow(z_rgba, interpolation="nearest", zorder=z_base)

                # Contour
                ax.contour(
                    zmask.astype(float),
                    levels=[0.5],
                    colors=[style["edge"]],
                    linewidths=2.2,
                    linestyles=style["linestyle"],
                    zorder=z_base + 0.02,
                )

                # Per-zone label + activation time at centroid
                ys, xs = np.where(zmask)
                if len(xs) > 0:
                    cx_z, cy_z = int(np.mean(xs)), int(np.mean(ys))
                    zlabel = getattr(zone, "label", "") or getattr(zone, "zone_id", "")
                    ax.text(
                        cx_z, cy_z, zlabel,
                        fontsize=6.5, fontweight="bold",
                        color=style["label_color"],
                        ha="center", va="center",
                        zorder=z_base + 0.04,
                        path_effects=[
                            path_effects.withStroke(
                                linewidth=2.0, foreground="black"
                            )
                        ],
                    )
                    # Activation time annotation
                    act_step = getattr(zone, "activation_step", 0)
                    ax.text(
                        cx_z, cy_z + max(4, self.H // 40),
                        f"T={act_step}",
                        fontsize=4.5, fontfamily="monospace",
                        color=style["label_color"],
                        ha="center", va="top", alpha=0.7,
                        zorder=z_base + 0.04,
                        path_effects=[
                            path_effects.withStroke(
                                linewidth=1.5, foreground="black"
                            )
                        ],
                    )

        # Fallback: if no structured zones, render merged dynamic_nfz_mask
        # (backward compat with old DynamicNFZModel or nfz_mask param)
        if not _rendered_zones:
            fallback_nfz = np.zeros((self.H, self.W), dtype=bool)
            if nfz_mask is not None:
                fallback_nfz |= nfz_mask
            if dynamic_nfz_mask is not None:
                fallback_nfz |= dynamic_nfz_mask
            if np.any(fallback_nfz):
                nfz_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
                nfz_rgba[fallback_nfz] = _hex_rgba(PALETTE["nfz_fill"], 0.40)
                hatch = (
                    (np.arange(self.H)[:, None] + np.arange(self.W)[None, :]) % 5
                ) < 2
                nfz_rgba[fallback_nfz & hatch] = _hex_rgba(PALETTE["nfz_edge"], 0.65)
                ax.imshow(nfz_rgba, interpolation="nearest", zorder=5)
                ax.contour(
                    fallback_nfz.astype(float),
                    levels=[0.5],
                    colors=[PALETTE["nfz_edge"]],
                    linewidths=2.5,
                    linestyles="solid",
                    zorder=5.5,
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
        # Mission-aware vehicle rendering
        mt_lower = self.mission_type.lower() if self.mission_type else ""
        if "maritime" in mt_lower:
            # Ship hull polygon (pentagon pointing up)
            from matplotlib.path import Path as MplPath
            _ship_verts = [(-0.4, -0.5), (-0.4, 0.2), (0.0, 0.5),
                           (0.4, 0.2), (0.4, -0.5), (-0.4, -0.5)]
            _ship_codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                           MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
            _SHIP_MARKER = MplPath(_ship_verts, _ship_codes)
            _VTYPES = [
                ("CG", "#3366CC"),    # Coast Guard vessel (blue)
                ("SAR", "#CC4444"),   # SAR vessel (red)
                ("FV", "#66AA44"),    # Fishing vessel (green)
            ]
        elif "critical" in mt_lower or "infra" in mt_lower:
            _SHIP_MARKER = None
            _VTYPES = [
                ("P", PALETTE["vehicle_police"]),
                ("P", PALETTE["vehicle_police"]),
                ("P", PALETTE["vehicle_police"]),
            ]
        else:
            _SHIP_MARKER = None
            _VTYPES = [
                ("A", PALETTE["vehicle_ambulance"]),
                ("P", PALETTE["vehicle_police"]),
                ("F", PALETTE["vehicle_fire"]),
            ]
        if traffic_positions is not None and len(traffic_positions) > 0:
            for i, pos in enumerate(traffic_positions):
                vy, vx = int(pos[0]), int(pos[1])
                label, color = _VTYPES[i % len(_VTYPES)]
                if _SHIP_MARKER is not None:
                    # Large ship marker on water
                    ax.plot(
                        vx, vy, marker=_SHIP_MARKER,
                        color=color, markersize=12,
                        markeredgecolor="white", markeredgewidth=1.0,
                        zorder=7,
                    )
                    ax.text(
                        vx, vy,
                        label, fontsize=3.5, color="white",
                        fontweight="bold", ha="center", va="center",
                        zorder=7.1,
                    )
                    # Wake trail (small translucent dots behind ship)
                    ax.plot(
                        vx, vy + 3, "o", color="white", alpha=0.3,
                        markersize=3, zorder=6.9,
                    )
                else:
                    ax.plot(
                        vx, vy, "s",
                        color=color, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=7,
                    )
                    ax.text(
                        vx + 1.5, vy - 1,
                        label, fontsize=4.5, color="white",
                        fontweight="bold", zorder=7.1,
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
            # White outline for contrast against busy overlays
            ax.plot(
                txs, tys, "-", color="white",
                linewidth=4.5, alpha=0.7, zorder=9.4,
                solid_capstyle="round",
            )
            ax.plot(
                txs, tys, "-", color=PALETTE["drone_safe"],
                linewidth=2.5, alpha=0.9, zorder=9.5,
                solid_capstyle="round",
            )

        # ═══════════════════════ Z 9.55: Planned lookahead path ══════════════
        # Drawn above trajectory so the blue planned line is always visible.
        if planned_path and len(planned_path) > 1:
            pxs = [p[0] for p in planned_path]
            pys = [p[1] for p in planned_path]
            # Black outline for legibility over fire / smoke
            ax.plot(
                pxs, pys, "-", color="black",
                linewidth=3.5, alpha=0.6, zorder=9.55,
                solid_capstyle="round",
            )
            ax.plot(
                pxs, pys, "--", color="#4FC3F7",
                linewidth=1.8, alpha=0.95, zorder=9.56,
                solid_capstyle="round",
                label="planned_path",
                path_effects=[
                    path_effects.withStroke(linewidth=3.0, foreground="black", alpha=0.4)
                ],
            )

        # Start & goal — large markers with halo
        _marker_sz = max(180, self.W * 0.6)
        ax.scatter(
            *self.start, c=PALETTE["start"], s=_marker_sz, zorder=9.6,
            edgecolors="white", linewidth=2.5, marker="o", label="Start",
        )
        ax.scatter(
            *self.goal, c=PALETTE["goal"], s=_marker_sz * 1.2, zorder=9.6,
            edgecolors="white", linewidth=2.5, marker="*", label="Goal",
        )

        # ═══════════════════════ Z 9.7: Mission POIs ══════════════════════
        if self.mission_type:
            self._draw_mission_pois(ax, step)

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
            plan_len=plan_len,
            plan_stale=plan_stale,
            plan_reason=plan_reason,
            forced_block_active=forced_block_active,
            forced_block_cleared=forced_block_cleared,
        )
        # ── Mission HUD (top-right) ──
        if mission_objective or mission_destination:
            from uavbench.visualization.hud import (
                MissionHUD,
                MissionHUDState,
                derive_plan_status,
            )
            _mhud = MissionHUD(has_briefing=True, bg_color=PALETTE["hud_bg"],
                               text_color=PALETTE["hud_text"],
                               accent_color=PALETTE["hud_accent"],
                               warn_color=PALETTE["hud_warn"])
            _mhud.draw(ax, MissionHUDState(
                objective=mission_objective,
                destination_name=mission_destination,
                mission_status=mission_status or "EN_ROUTE",
                distance_to_goal=distance_to_goal,
                plan_status=derive_plan_status(
                    plan_len=plan_len, plan_stale=plan_stale,
                    plan_reason=plan_reason,
                ),
                step=step,
                max_steps=mission_max_steps or total_steps,
            ))

        self._draw_legend(
            ax, fire_mask, smoke_mask, traffic_closure_mask,
            risk_map, intruder_positions, comms_coverage_map,
            restriction_zones=restriction_zones,
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
        try:
            fig.tight_layout(pad=0.5)
        except Exception:
            pass  # skip tight_layout if axes can't accommodate decorations
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
        plan_len: int = 0,
        plan_stale: bool = False,
        plan_reason: str = "none",
        forced_block_active: bool = False,
        forced_block_cleared: bool = False,
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
        if plan_len > 1 and not plan_stale:
            r3.append(f"PLAN: {plan_len}wp")
        elif plan_stale:
            r3.append(f"STALE PLAN ({plan_reason})")
        elif plan_len <= 1 and plan_reason not in ("none", "initial", ""):
            r3.append("NO PLAN")
        if forced_block_cleared:
            r3.append("FORCED BLOCK: CLEARED")
        elif forced_block_active:
            r3.append("FORCED BLOCK: ACTIVE")
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

        # ── 4. ODbL attribution (bottom-center, small) ──────────────────
        ax.text(
            0.50,
            0.002,
            self.basemap_style.attribution_text,
            transform=ax.transAxes,
            fontsize=5,
            color="#888888",
            fontstyle="italic",
            ha="center",
            va="bottom",
            zorder=26,
            clip_on=False,
            path_effects=[
                path_effects.withStroke(linewidth=1.0, foreground="white")
            ],
        )

        # ── 5. Version pin (bottom-right, tiny) ───────────────────────
        ax.text(
            0.99,
            0.002,
            _get_version_pin(),
            transform=ax.transAxes,
            fontsize=4.5,
            fontfamily="monospace",
            color="#666666",
            ha="right",
            va="bottom",
            zorder=26,
            clip_on=False,
            alpha=0.6,
            path_effects=[
                path_effects.withStroke(linewidth=0.8, foreground="white")
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
        restriction_zones: Optional[list] = None,
    ) -> None:
        """Context-aware legend — only active layers shown."""
        patches: list[Any] = [
            mpatches.Patch(color=PALETTE["building_fill"], label="Building"),
        ]
        # Static no-fly always shown if present
        if np.any(self.no_fly):
            patches.append(
                mpatches.Patch(color=PALETTE["nfz_fill"], alpha=0.5, label="Static No-Fly")
            )
        # Per-zone-type legend entries (deduplicated by type)
        _zone_legend = {
            "tfr": ("TFR", PALETTE["zone_tfr_fill"]),
            "sar_box": ("SAR Box", PALETTE["zone_sar_fill"]),
            "port_exclusion": ("Port Exclusion", PALETTE["zone_port_fill"]),
            "security_cordon": ("Security Cordon", PALETTE["zone_cordon_fill"]),
        }
        _seen_types: set[str] = set()
        if restriction_zones:
            for zone in restriction_zones:
                if not getattr(zone, "active", False):
                    continue
                ztype = getattr(zone, "zone_type", "tfr")
                if ztype not in _seen_types:
                    _seen_types.add(ztype)
                    lbl, clr = _zone_legend.get(ztype, (ztype.upper(), PALETTE["nfz_fill"]))
                    patches.append(
                        mpatches.Patch(color=clr, alpha=0.5, label=lbl)
                    )
        # Landuse entries when landuse_map is present
        if self.landuse_map is not None:
            if np.any(self.landuse_map == 4):
                patches.append(
                    mpatches.Patch(color=PALETTE["water"], label="Water")
                )
            if np.any(self.landuse_map == 3):
                patches.append(
                    mpatches.Patch(
                        color=self.basemap_style.color_vegetation,
                        label="Vegetation",
                    )
                )
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

    # ── Mission POIs ───────────────────────────────────────────────────

    def _draw_mission_pois(self, ax: Any, step: int) -> None:
        """Draw mission-specific POI icons at z-order 9.7.

        Rendered above trajectory (9.5) but below UAV (10.5) so they
        remain visible over fire/smoke/NFZ overlays.  Each icon gets a
        white halo circle for contrast.
        """
        if self._icon_lib is None:
            return
        IconID = self._IconID
        _z = 9.7
        _halo_r = self._icon_lib.icon_size * 0.55

        def _halo(x: float, y: float) -> None:
            """White disc behind icon for readability."""
            ax.add_patch(plt.Circle(
                (x, y), _halo_r,
                facecolor="white", edgecolor="#CCCCCC",
                linewidth=0.8, alpha=0.85, zorder=_z - 0.05,
            ))

        # Start marker — HOME icon (green)
        _halo(self.start[0], self.start[1])
        self._icon_lib.stamp(
            IconID.HOME,
            (self.start[0], self.start[1]),
            ax,
            color=PALETTE["start"],
            zorder=_z,
        )

        # Goal marker — mission-specific icon
        goal_icon = IconID.HOME
        goal_color = PALETTE["goal"]
        mt = self.mission_type.lower() if self.mission_type else ""
        if "civil" in mt or "protection" in mt:
            goal_icon = IconID.HOME
        elif "maritime" in mt or "domain" in mt:
            goal_icon = IconID.ANCHOR
            goal_color = "#0088CC"
        elif "critical" in mt or "infra" in mt:
            goal_icon = IconID.INSPECTION
            goal_color = "#00CC88"

        _halo(self.goal[0], self.goal[1])
        self._icon_lib.stamp(
            goal_icon,
            (self.goal[0], self.goal[1]),
            ax,
            color=goal_color,
            zorder=_z,
        )

        # Explicit POIs from _compute_mission_pois
        for poi in self._mission_pois:
            xy = poi.get("xy", (0, 0))
            icon_id = poi.get("icon", IconID.WAYPOINT)
            color = poi.get("color", "#00CC44")
            label = poi.get("label")
            _halo(xy[0], xy[1])
            self._icon_lib.stamp(
                icon_id, (xy[0], xy[1]), ax,
                color=color, label=label, zorder=_z,
            )

    # ── Title card ────────────────────────────────────────────────────

    def _render_title_card(self, num_frames: int = 18) -> list[np.ndarray]:
        """Render an animated COP briefing-style title card.

        Phase 1 (frames 0-5):   Mission name fades in
        Phase 2 (frames 6-11):  Incident chips appear one by one
        Phase 3 (frames 12-17): Metadata + disclaimer + version pin
        """
        # Resolve mission profile
        profile = None
        try:
            from uavbench.visualization.stakeholder_renderer import MISSION_PROFILES
            mt = self.mission_type.lower() if self.mission_type else ""
            for key, prof in MISSION_PROFILES.items():
                if key in mt or mt in key:
                    profile = prof
                    break
        except ImportError:
            pass

        accent = profile.accent_color if profile else PALETTE["hud_accent"]
        mission_name = (
            profile.name if profile
            else (self.mission_type.replace("_", " ").title()
                  if self.mission_type else "Mission")
        )

        # Subtitle parts
        parts: list[str] = []
        if profile:
            parts.append(profile.agency)
        if self.scenario_id:
            parts.append(self.scenario_id)
        if self.planner_name:
            parts.append(f"Planner: {self.planner_name}")
        if self.osm_tile_id:
            parts.append(f"Tile: {self.osm_tile_id}")
        subtitle = "  \u2502  ".join(parts)   # │ separator

        version_str = _get_version_pin()
        disclaimer = (
            "Incident-driven route replanning benchmark "
            "(not a flight dynamics simulator)"
        )

        # Chip color lookup
        _chip_colors = {
            "FIRE": "#FF6B35",
            "TRAFFIC": "#FFCC00",
            "TFR": PALETTE["zone_tfr_label"],
            "SAR": PALETTE["zone_sar_label"],
            "CORDON": PALETTE["zone_cordon_label"],
            "NFZ": PALETTE["nfz_shield"],
            "WIND": "#AAAAAA",
            "CORRIDOR": "#00E5FF",
            "COMMS": "#4488FF",
        }

        def _chip_color(text: str) -> str:
            for kw, c in _chip_colors.items():
                if kw in text:
                    return c
            return PALETTE["hud_accent"]

        frames: list[np.ndarray] = []
        for i in range(num_frames):
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(PALETTE["hud_bg"])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # ── Phase 1: Mission name fade-in (frames 0-5) ──────────
            name_alpha = min(1.0, (i + 1) / 6.0)
            ax.text(
                0.5, 0.72, mission_name,
                fontsize=30, fontweight="bold", ha="center", va="center",
                color=accent, alpha=name_alpha,
                transform=ax.transAxes,
            )
            # Greek subtitle (appears from frame 2)
            if profile and profile.name_el and i >= 2:
                greek_alpha = min(1.0, (i - 1) / 5.0)
                ax.text(
                    0.5, 0.63, profile.name_el,
                    fontsize=14, ha="center", va="center",
                    color="#BBBBBB", alpha=greek_alpha,
                    transform=ax.transAxes,
                )

            # ── Phase 2: Incident chips (frames 6-11) ───────────────
            if i >= 6 and self._active_incidents:
                total = len(self._active_incidents)
                # Progressive reveal: show more chips each frame
                chips_to_show = min(
                    total,
                    int((i - 5) * total / 6) + 1,
                )
                chip_width = min(0.18, 0.90 / max(total, 1))
                start_x = 0.5 - (total * chip_width) / 2

                for ci in range(chips_to_show):
                    cx = start_x + ci * chip_width + chip_width / 2
                    # Staggered fade per chip
                    chip_alpha = min(
                        1.0,
                        max(0.0, (i - 5 - ci * 6.0 / total) / 2.0),
                    )
                    cc = _chip_color(self._active_incidents[ci])
                    ax.text(
                        cx, 0.50, self._active_incidents[ci],
                        fontsize=8, fontweight="bold",
                        ha="center", va="center",
                        color=cc, alpha=chip_alpha,
                        transform=ax.transAxes,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor=PALETTE["hud_bg"],
                            edgecolor=cc,
                            linewidth=1.2,
                            alpha=chip_alpha * 0.7,
                        ),
                    )

            # ── Phase 3: Metadata + disclaimer + version (frames 12-17)
            if i >= 12:
                meta_alpha = min(1.0, (i - 11) / 6.0)

                # Subtitle (agency | scenario | planner | tile)
                ax.text(
                    0.5, 0.38, subtitle,
                    fontsize=11, ha="center", va="center",
                    color=PALETTE["hud_text"], alpha=meta_alpha,
                    transform=ax.transAxes,
                )

                # Mission description
                if profile and profile.description:
                    ax.text(
                        0.5, 0.30, profile.description,
                        fontsize=10, ha="center", va="center",
                        color="#AAAAAA", fontstyle="italic",
                        alpha=meta_alpha,
                        transform=ax.transAxes,
                    )

                # Claimed realism disclaimer
                ax.text(
                    0.5, 0.18, disclaimer,
                    fontsize=8, ha="center", va="center",
                    color="#888888", fontstyle="italic",
                    alpha=meta_alpha,
                    transform=ax.transAxes,
                )

                # ODbL attribution
                ax.text(
                    0.5, 0.08, self.basemap_style.attribution_text,
                    fontsize=7, ha="center", va="center",
                    color="#666666", fontstyle="italic",
                    alpha=meta_alpha,
                    transform=ax.transAxes,
                )

                # Version pin (bottom-right)
                ax.text(
                    0.97, 0.03, version_str,
                    fontsize=7, ha="right", va="bottom",
                    fontfamily="monospace",
                    color="#555555", alpha=meta_alpha,
                    transform=ax.transAxes,
                )

            fig.tight_layout(pad=0)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frame = buf[:, :, :3].copy()
            plt.close(fig)
            frames.append(frame)

        return frames

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
        title_card_seconds: float = 1.8,
        max_frames: int = 200,
    ) -> None:
        """Export frames as animated GIF (Pillow writer).

        Prepends a mission title card (dark HUD background) for
        *title_card_seconds*.  Set to 0 to disable.

        When the episode has more than *max_frames*, frames are uniformly
        sub-sampled while preserving the first frame, last frame, and all
        keyframe indices (replan / guardrail events).
        """
        import matplotlib.animation as animation

        episode_frames = list(frames or self._frames)
        if not episode_frames:
            return

        # ── Sub-sample long episodes ─────────────────────────────────────
        if len(episode_frames) > max_frames:
            # Always keep first, last, and keyframes
            keep: set[int] = {0, len(episode_frames) - 1}
            keep.update(
                i for i in self._keyframe_indices
                if 0 <= i < len(episode_frames)
            )
            # Fill remaining budget with uniform samples
            budget = max_frames - len(keep)
            if budget > 0:
                step = max(1, len(episode_frames) // budget)
                for i in range(0, len(episode_frames), step):
                    keep.add(i)
                    if len(keep) >= max_frames:
                        break
            indices = sorted(keep)[:max_frames]
            episode_frames = [episode_frames[i] for i in indices]

        # ── Prepend title card ───────────────────────────────────────────
        if title_card_seconds > 0:
            n_title = max(1, int(title_card_seconds * fps))
            title_frames = self._render_title_card(num_frames=n_title)
            # Resize title frames to match episode frame dimensions if needed
            if title_frames and title_frames[0].shape != episode_frames[0].shape:
                from PIL import Image
                target_h, target_w = episode_frames[0].shape[:2]
                resized = []
                for tf in title_frames:
                    img = Image.fromarray(tf)
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                    resized.append(np.array(img))
                title_frames = resized
            use = title_frames + episode_frames
        else:
            use = episode_frames

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
