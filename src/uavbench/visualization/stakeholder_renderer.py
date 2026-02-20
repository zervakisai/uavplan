"""Stakeholder-ready mission visualization — 4-pane layout with OSM basemaps.

Produces cinema-quality frames for Greek government agency demos:
  - ΓΓΠΠ (Civil Protection) — wildfire on Penteli WUI tile
  - ΛΣ-ΕΛΑΚΤ (Coast Guard) — maritime SAR on Piraeus port tile
  - ΥΠΕΘΑ (ISR-support) — critical infrastructure on Athens downtown tile

Design goals
────────────
1. **At-a-glance** comprehension (<5 sec for non-technical audience)
2. **Icon-first** — universal pictograms, no jargon
3. **Real OSM basemaps** — offline rasterised tiles
4. **Deterministic replay** — reproducible from episode log
5. **1080p MP4 export** — cinema-quality with OSM attribution

4-Pane Layout
─────────────
┌─────────────────────────────┬──────────────┐
│                             │  METRICS     │
│      MAP VIEWPORT           │  (KPIs,      │
│      (60% width)            │   gauges)    │
│                             │              │
├─────────────────────────────┼──────────────┤
│  TIMELINE BAR               │  LEGEND      │
│  (events, phase markers)    │  (icons)     │
└─────────────────────────────┴──────────────┘

Layer z-order (extends operational_renderer)
────────────────────────────────────────────
Z1  OSM basemap (buildings, roads, landuse)
Z2  Risk / elevation heatmap
Z3  Dynamic hazard overlays (fire, smoke, waves)
Z4  Restricted zones (NFZ, traffic closure)
Z5  Mission POIs (icons: waypoints, tasks, products)
Z6  Entity tracks (vehicle positions, vessel AIS)
Z7  Planned path + completed segments
Z8  UAV icon + safety bubble + heading
Z9  Event annotations (replan flash, distress pulse)
Z10 Cartographic overlay (scale bar, north arrow, coords)
Z11 HUD panels (metrics, legend, timeline)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as path_effects
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from uavbench.visualization.icons import IconLibrary, IconID
from uavbench.visualization.basemap_style import build_styled_basemap

# Re-use palette from operational renderer
try:
    from uavbench.visualization.operational_renderer import PALETTE
except ImportError:
    PALETTE = {}


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette extensions for stakeholder view
# ─────────────────────────────────────────────────────────────────────────────

STAKEHOLDER_PALETTE = {
    # Basemap
    "ground":            "#F0EDE6",
    "road_primary":      "#C8C8C8",
    "road_secondary":    "#999999",
    "building_fill":     "#5A5A5A",
    "building_edge":     "#3D3D3D",
    "water":             "#A4C8E1",
    "vegetation":        "#8FBC8F",
    "open_land":         "#E8E4D8",
    # Landuse classes (from rasterize.py: 0=other, 1=residential, 2=commercial,
    #   3=vegetation, 4=water)
    "landuse_residential": "#E8D8C8",
    "landuse_commercial":  "#D8D0C8",
    "landuse_vegetation":  "#B8D8A8",
    "landuse_water":       "#A4C8E1",
    "landuse_other":       "#F0EDE6",
    # Mission accent colours
    "civil_protection":  "#FF6B35",   # warm orange
    "maritime_domain":   "#0088CC",   # ocean blue
    "critical_infra":    "#00CC88",   # teal green
    # HUD / panels
    "panel_bg":          "#0A0F1ACC",
    "panel_text":        "#E8E8E8",
    "panel_accent":      "#00DDFF",
    "panel_warn":        "#FF6B35",
    "panel_ok":          "#00CC44",
    "panel_border":      "#334155",
    # Attribution
    "attribution_bg":    "#FFFFFFAA",
    "attribution_text":  "#555555",
}


# ─────────────────────────────────────────────────────────────────────────────
# Mission visual profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MissionVisualProfile:
    """Visual identity for a mission type."""
    name: str
    name_el: str                    # Greek name
    accent_color: str
    agency: str
    agency_el: str                  # Greek agency name
    tile_id: str
    icon_poi: str                   # primary POI icon
    icon_hazard: str                # hazard icon
    icon_task: str                  # task/product icon
    description: str = ""


MISSION_PROFILES: dict[str, MissionVisualProfile] = {
    "civil_protection": MissionVisualProfile(
        name="Wildfire Monitoring",
        name_el="Παρακολούθηση Πυρκαγιάς",
        accent_color="#FF6B35",
        agency="GSCP (ΓΓΠΠ)",
        agency_el="Γενική Γραμματεία Πολιτικής Προστασίας",
        tile_id="penteli",
        icon_poi=IconID.FIRE,
        icon_hazard=IconID.ALERT,
        icon_task=IconID.CAMERA,
        description="WUI perimeter monitoring & corridor inspection",
    ),
    "maritime_domain": MissionVisualProfile(
        name="Maritime SAR",
        name_el="Θαλάσσια Έρευνα & Διάσωση",
        accent_color="#0088CC",
        agency="HCG (ΛΣ-ΕΛΑΚΤ)",
        agency_el="Λιμενικό Σώμα – Ελληνική Ακτοφυλακή",
        tile_id="piraeus",
        icon_poi=IconID.SHIP,
        icon_hazard=IconID.DISTRESS,
        icon_task=IconID.ANCHOR,
        description="Port area patrol & distress response",
    ),
    "critical_infrastructure": MissionVisualProfile(
        name="Infrastructure Inspection",
        name_el="Επιθεώρηση Υποδομών",
        accent_color="#00CC88",
        agency="MoD (ΥΠΕΘΑ)",
        agency_el="Υπουργείο Εθνικής Άμυνας",
        tile_id="downtown",
        icon_poi=IconID.BUILDING,
        icon_hazard=IconID.SHIELD,
        icon_task=IconID.INSPECTION,
        description="Critical site inspection & monitoring",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Tile loader
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TileData:
    """Loaded OSM tile data for rendering."""
    tile_id: str
    heightmap: np.ndarray          # [H, W] float32  (building heights)
    roads_mask: np.ndarray         # [H, W] bool
    landuse_map: np.ndarray        # [H, W] int8 (0-4)
    risk_map: np.ndarray           # [H, W] float32 (0-1)
    nfz_mask: np.ndarray           # [H, W] bool
    # Optional graph
    roads_graph_nodes: np.ndarray | None = None
    roads_graph_edges: np.ndarray | None = None
    # Metadata
    center_latlon: tuple[float, float] = (0.0, 0.0)
    resolution_m: float = 3.0
    grid_size: int = 500
    crs: str = "EPSG:32634"
    # Affine transform (from raster_meta)
    affine_transform: list[float] = field(default_factory=list)


def load_tile(tile_id: str, data_dir: Path | None = None) -> TileData:
    """Load an OSM tile from the data/maps directory.

    Parameters
    ----------
    tile_id : str
        One of "penteli", "downtown", "piraeus".
    data_dir : Path, optional
        Override data directory.  Default: project ``data/maps/``.

    Returns
    -------
    TileData
    """
    if data_dir is None:
        # src/uavbench/visualization/ -> parents[2] = project root
        data_dir = Path(__file__).resolve().parents[3] / "data" / "maps"

    npz_path = data_dir / f"{tile_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Tile {tile_id!r} not found at {npz_path}.  "
            f"Run `python tools/osm_pipeline/fetch.py && python tools/osm_pipeline/rasterize.py`"
        )

    data = np.load(str(npz_path))

    td = TileData(
        tile_id=tile_id,
        heightmap=data.get("heightmap", np.zeros((500, 500), dtype=np.float32)),
        roads_mask=data.get("roads_mask", np.zeros((500, 500), dtype=bool)),
        landuse_map=data.get("landuse_map", np.zeros((500, 500), dtype=np.int8)),
        risk_map=data.get("risk_map", np.zeros((500, 500), dtype=np.float32)),
        nfz_mask=data.get("nfz_mask", np.zeros((500, 500), dtype=bool)),
        roads_graph_nodes=data.get("roads_graph_nodes"),
        roads_graph_edges=data.get("roads_graph_edges"),
    )

    # Load metadata
    meta_dir = data_dir / tile_id
    fetch_meta = meta_dir / "fetch_meta.json"
    if fetch_meta.exists():
        with open(fetch_meta) as f:
            fm = json.load(f)
        td.center_latlon = tuple(fm.get("center_latlon", [0, 0]))
        td.resolution_m = fm.get("resolution_m", 3.0)
        td.grid_size = fm.get("grid_size", 500)

    raster_meta = meta_dir / "raster_meta.json"
    if raster_meta.exists():
        with open(raster_meta) as f:
            rm = json.load(f)
        td.crs = rm.get("target_crs", "EPSG:32634")
        td.affine_transform = rm.get("transform", [])

    return td


# ─────────────────────────────────────────────────────────────────────────────
# Basemap builder
# ─────────────────────────────────────────────────────────────────────────────

def _hex_rgb(hex_color: str) -> np.ndarray:
    """Convert '#RRGGBB' to float32 [R, G, B] in [0, 1]."""
    h = hex_color.lstrip("#")
    if len(h) == 8:  # RGBA
        h = h[:6]
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


_LANDUSE_COLORS = {
    0: "landuse_other",
    1: "landuse_residential",
    2: "landuse_commercial",
    3: "landuse_vegetation",
    4: "landuse_water",
}


def build_basemap(tile: TileData) -> np.ndarray:
    """Build a rich RGB basemap [H, W, 3] float32 from tile data.

    Delegates to the shared :func:`build_styled_basemap` so both
    ``StakeholderRenderer`` and ``OperationalRenderer`` render identical
    cartography (landuse fills, 3-tier roads, building edges, hillshade,
    water coastlines).
    """
    return build_styled_basemap(
        tile.heightmap,
        tile.roads_mask,
        landuse_map=tile.landuse_map,
    )


# ─────────────────────────────────────────────────────────────────────────────
# StakeholderRenderer
# ─────────────────────────────────────────────────────────────────────────────

class StakeholderRenderer:
    """4-pane stakeholder visualization renderer.

    Produces 1920×1080 frames suitable for demo MP4 export.

    Parameters
    ----------
    tile : TileData
        Loaded OSM tile.
    mission_type : str
        One of "civil_protection", "maritime_domain", "critical_infrastructure".
    scenario_id : str
        Scenario identifier for HUD display.
    planner_name : str
        Planner name for HUD.
    difficulty : str
        "easy", "medium", or "hard".
    icon_size : float
        Icon size in grid-cell units.  Default 10.
    figsize : tuple
        Figure size in inches.  Default (19.2, 10.8) for 1080p at 100 DPI.
    dpi : int
        Output DPI.  Default 100  (→ 1920×1080 px).
    """

    def __init__(
        self,
        tile: TileData,
        mission_type: str = "civil_protection",
        *,
        scenario_id: str = "",
        planner_name: str = "astar",
        difficulty: str = "medium",
        icon_size: float = 10.0,
        figsize: tuple[float, float] = (19.2, 10.8),
        dpi: int = 100,
    ) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for StakeholderRenderer")

        self.tile = tile
        self.mission_type = mission_type
        self.profile = MISSION_PROFILES.get(mission_type, MISSION_PROFILES["civil_protection"])
        self.scenario_id = scenario_id
        self.planner_name = planner_name
        self.difficulty = difficulty
        self.icon_size = icon_size
        self.figsize = figsize
        self.dpi = dpi
        self.H, self.W = tile.heightmap.shape

        # Icon library
        self.icons = IconLibrary(icon_size=icon_size)

        # Pre-compute basemap
        self._basemap_rgb = build_basemap(tile)

        # Frame buffer
        self._frames: list[np.ndarray] = []
        self._events: list[dict[str, Any]] = []
        self._keyframe_indices: list[int] = []

        # Build the figure with 4-pane layout
        self._fig: Figure
        self._ax_map: Axes
        self._ax_metrics: Axes
        self._ax_timeline: Axes
        self._ax_legend: Axes
        self._build_figure()

    def _build_figure(self) -> None:
        """Create the 4-pane matplotlib figure."""
        self._fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self._fig.set_facecolor(STAKEHOLDER_PALETTE["panel_bg"][:7])

        # GridSpec: 2 rows, 2 cols
        # Row 0: map (75%) | metrics (25%)
        # Row 1: timeline (75%) | legend (25%)
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[3, 1],
            height_ratios=[5, 1],
            hspace=0.04,
            wspace=0.03,
            left=0.02, right=0.98,
            top=0.95, bottom=0.02,
        )

        self._ax_map = self._fig.add_subplot(gs[0, 0])
        self._ax_metrics = self._fig.add_subplot(gs[0, 1])
        self._ax_timeline = self._fig.add_subplot(gs[1, 0])
        self._ax_legend = self._fig.add_subplot(gs[1, 1])

        # Configure map axes
        self._ax_map.set_xlim(-0.5, self.W - 0.5)
        self._ax_map.set_ylim(self.H - 0.5, -0.5)
        self._ax_map.set_aspect("equal")
        self._ax_map.set_xticks([])
        self._ax_map.set_yticks([])
        for spine in self._ax_map.spines.values():
            spine.set_color(self.profile.accent_color)
            spine.set_linewidth(2)

        # Configure side panels
        for ax in [self._ax_metrics, self._ax_timeline, self._ax_legend]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(STAKEHOLDER_PALETTE["panel_border"])
                spine.set_linewidth(1)

        # Title bar
        title_text = f"{self.profile.name}  |  {self.profile.name_el}"
        self._fig.suptitle(
            title_text,
            fontsize=16, fontweight="bold",
            color=self.profile.accent_color,
            y=0.98,
        )

    # ─── Z1: Basemap ────────────────────────────────────────────────────

    def _draw_basemap(self) -> None:
        """Z1 — render the pre-computed OSM basemap."""
        self._ax_map.imshow(
            self._basemap_rgb,
            origin="upper",
            extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
            zorder=1,
            interpolation="nearest",
        )

    # ─── Z2: Risk / elevation heatmap ───────────────────────────────────

    def _draw_risk_overlay(
        self,
        risk_map: np.ndarray | None = None,
        alpha: float = 0.25,
    ) -> None:
        """Z2 — risk heatmap overlay."""
        rm = risk_map if risk_map is not None else self.tile.risk_map
        if rm is None or not rm.any():
            return
        cmap = LinearSegmentedColormap.from_list(
            "risk", ["#FFFFFF00", "#FFCC00", "#FF6600", "#FF0000"], N=256,
        )
        self._ax_map.imshow(
            rm.astype(np.float32),
            origin="upper",
            extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
            cmap=cmap,
            alpha=alpha,
            zorder=2,
            interpolation="bilinear",
        )

    # ─── Z3: Dynamic hazard overlays ────────────────────────────────────

    def _draw_fire_overlay(
        self,
        fire_mask: np.ndarray | None,
        smoke_mask: np.ndarray | None,
        step: int = 0,
    ) -> None:
        """Z3 — fire + smoke with flicker animation."""
        if fire_mask is not None and fire_mask.any():
            # Sinusoidal flicker
            flicker = 0.6 + 0.3 * math.sin(step * 0.5)
            fire_rgba = np.zeros((*fire_mask.shape, 4), dtype=np.float32)
            fire_rgba[fire_mask] = [1.0, 0.17, 0.0, flicker]
            # Edge glow
            try:
                from scipy.ndimage import binary_dilation
                edge = binary_dilation(fire_mask, iterations=2) & ~fire_mask
                fire_rgba[edge] = [1.0, 0.42, 0.11, flicker * 0.4]
            except ImportError:
                pass
            self._ax_map.imshow(
                fire_rgba,
                origin="upper",
                extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
                zorder=3,
                interpolation="nearest",
            )

        if smoke_mask is not None and smoke_mask.any():
            drift = int(2 * math.sin(step * 0.3))
            shifted = np.roll(smoke_mask, drift, axis=1)
            smoke_alpha = 0.2 + 0.1 * math.sin(step * 0.2)
            smoke_rgba = np.zeros((*shifted.shape, 4), dtype=np.float32)
            smoke_rgba[shifted] = [0.47, 0.47, 0.47, smoke_alpha]
            self._ax_map.imshow(
                smoke_rgba,
                origin="upper",
                extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
                zorder=3,
                interpolation="bilinear",
            )

    # ─── Z4: Restricted zones ──────────────────────────────────────────

    def _draw_restricted_zones(
        self,
        nfz_mask: np.ndarray | None = None,
        traffic_closure_mask: np.ndarray | None = None,
    ) -> None:
        """Z4 — NFZ hatching + traffic closure stripes."""
        nfz = nfz_mask if nfz_mask is not None else self.tile.nfz_mask
        if nfz is not None and nfz.any():
            nfz_rgba = np.zeros((*nfz.shape, 4), dtype=np.float32)
            nfz_rgba[nfz] = [1.0, 0.0, 1.0, 0.25]
            self._ax_map.imshow(
                nfz_rgba,
                origin="upper",
                extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
                zorder=4,
                interpolation="nearest",
            )

        if traffic_closure_mask is not None and traffic_closure_mask.any():
            yy, xx = np.mgrid[0:self.H, 0:self.W]
            stripe = ((xx + yy) % 6) < 3
            tc_rgba = np.zeros((*traffic_closure_mask.shape, 4), dtype=np.float32)
            active = traffic_closure_mask & stripe
            tc_rgba[active] = [1.0, 1.0, 0.0, 0.3]
            self._ax_map.imshow(
                tc_rgba,
                origin="upper",
                extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
                zorder=4,
                interpolation="nearest",
            )

    # ─── Z5: Mission POIs (icons) ──────────────────────────────────────

    def _draw_pois(
        self,
        pois: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Z5 — stamp mission POI icons.

        Each POI dict: {"xy": (x, y), "icon": IconID, "label": str,
                        "status": "pending"|"active"|"completed"|"expired",
                        "color": optional str}
        """
        if not pois:
            return
        for poi in pois:
            xy = poi["xy"]
            icon_id = poi.get("icon", IconID.WAYPOINT)
            status = poi.get("status", "pending")
            label = poi.get("label", "")
            color = poi.get("color")

            # Status-dependent styling
            alpha = 1.0
            if status == "completed":
                icon_id = IconID.WAYPOINT_DONE
                alpha = 0.4
            elif status == "expired":
                alpha = 0.3
            elif status == "active":
                alpha = 1.0
                # Pulse effect — slightly larger
                self.icons.stamp(
                    icon_id, (xy[0], xy[1]), self._ax_map,
                    size=self.icon_size * 1.5, alpha=0.15,
                    color=self.profile.accent_color,
                    zorder=5,
                )

            self.icons.stamp(
                icon_id, (xy[0], xy[1]), self._ax_map,
                color=color, alpha=alpha,
                label=label,
                zorder=5,
            )

    # ─── Z6: Entity tracks ─────────────────────────────────────────────

    def _draw_entities(
        self,
        entity_positions: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Z6 — emergency vehicles, vessels, etc.

        Each dict: {"xy": (x, y), "type": "ambulance"|"police"|"fire"|"vessel",
                    "heading": float_deg, "trail": [(x,y), ...]}
        """
        if not entity_positions:
            return
        type_icons = {
            "ambulance": (IconID.ALERT, "#FF4444"),
            "police": (IconID.SHIELD, "#4488FF"),
            "fire": (IconID.FIRE, "#FF8800"),
            "vessel": (IconID.SHIP, "#4488FF"),
        }
        for ent in entity_positions:
            icon_id, color = type_icons.get(ent.get("type", ""), (IconID.WAYPOINT, "#888888"))
            xy = ent["xy"]
            heading = ent.get("heading", 0.0)
            trail = ent.get("trail", [])

            # Draw trail
            if trail and len(trail) > 1:
                tx = [p[0] for p in trail]
                ty = [p[1] for p in trail]
                self._ax_map.plot(
                    tx, ty,
                    color=color, alpha=0.3, linewidth=1, linestyle="--",
                    zorder=6,
                )

            self.icons.stamp(
                icon_id, (xy[0], xy[1]), self._ax_map,
                size=self.icon_size * 0.7,
                color=color,
                rotation_deg=heading,
                zorder=6,
            )

    # ─── Z7: Path ──────────────────────────────────────────────────────

    def _draw_path(
        self,
        trajectory: list[tuple[int, int]] | None = None,
        planned_path: list[tuple[int, int]] | None = None,
        completed_segments: int | None = None,
    ) -> None:
        """Z7 — completed trajectory + remaining planned path."""
        if trajectory and len(trajectory) > 1:
            tx = [p[0] for p in trajectory]
            ty = [p[1] for p in trajectory]
            n = len(trajectory)
            seg_end = completed_segments if completed_segments is not None else n

            # Completed portion (solid)
            if seg_end > 1:
                self._ax_map.plot(
                    tx[:seg_end], ty[:seg_end],
                    color=self.profile.accent_color, alpha=0.8,
                    linewidth=2.5, solid_capstyle="round",
                    zorder=7,
                )

        if planned_path and len(planned_path) > 1:
            px = [p[0] for p in planned_path]
            py = [p[1] for p in planned_path]
            self._ax_map.plot(
                px, py,
                color=self.profile.accent_color, alpha=0.35,
                linewidth=1.5, linestyle="--",
                zorder=7,
            )

    # ─── Z8: UAV icon ──────────────────────────────────────────────────

    def _draw_uav(
        self,
        drone_pos: tuple[int, int],
        heading_deg: float = 0.0,
        safety_radius: float = 0.0,
    ) -> None:
        """Z8 — UAV icon with heading and optional safety bubble."""
        x, y = drone_pos

        # Safety bubble
        if safety_radius > 0:
            circle = mpatches.Circle(
                (x, y), safety_radius,
                fill=False, edgecolor="#0088FF",
                linewidth=1, alpha=0.4, linestyle=":",
                zorder=8,
            )
            self._ax_map.add_patch(circle)

        self.icons.stamp(
            IconID.UAV, (x, y), self._ax_map,
            size=self.icon_size * 1.2,
            color="#0066FF",
            rotation_deg=heading_deg,
            zorder=8,
        )

    # ─── Z9: Event annotations ─────────────────────────────────────────

    def _draw_events(
        self,
        step: int,
        replan_flash: bool = False,
        replan_reason: str = "",
        invalidated_waypoint: tuple[int, int] | None = None,
        distress_position: tuple[int, int] | None = None,
        conflict_markers: Sequence[dict[str, Any]] | None = None,
        violation_flash: bool = False,
        replan_annotations: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Z9 — replan pulse, distress beacon, invalidated waypoint X,
        conflict markers, violation flash, replan annotations."""

        # Violation flash — red border pulse
        if violation_flash:
            for spine in self._ax_map.spines.values():
                spine.set_color("#FF0000")
                spine.set_linewidth(4)

        if replan_flash:
            self._ax_map.axhline(y=0, color="white", alpha=0.15, linewidth=self.W, zorder=9)
            if replan_reason:
                self._ax_map.text(
                    self.W * 0.5, 15, f"⟳ {replan_reason}",
                    fontsize=11, ha="center", va="top",
                    color="#FFFFFF", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FF4444CC", ec="none"),
                    zorder=9,
                )

        if invalidated_waypoint:
            ix, iy = invalidated_waypoint
            self._ax_map.plot(
                ix, iy, "x",
                color="#DD2222", markersize=14, markeredgewidth=3,
                zorder=9,
            )

        if distress_position:
            # Pulsing distress beacon
            dx, dy = distress_position
            pulse = 0.5 + 0.5 * math.sin(step * 0.8)
            self.icons.stamp(
                IconID.DISTRESS, (dx, dy), self._ax_map,
                size=self.icon_size * (1.0 + 0.5 * pulse),
                color="#FF0044",
                alpha=0.3 + 0.5 * pulse,
                zorder=9,
            )

        # Conflict markers — red Xs on detected conflicts
        if conflict_markers:
            for cm in conflict_markers:
                cx, cy = cm.get("position", (0, 0))
                sev = cm.get("severity", 1.0)
                self._ax_map.plot(
                    cx, cy, "x",
                    color="#FF2222", markersize=10 + 4 * sev,
                    markeredgewidth=2, alpha=0.6 + 0.4 * sev,
                    zorder=9,
                )
                # Collision-risk ring
                ring_radius = cm.get("radius", 8)
                ring = mpatches.Circle(
                    (cx, cy), ring_radius,
                    fill=False, edgecolor="#FF4444",
                    linewidth=1.5, alpha=0.4 * sev, linestyle=":",
                    zorder=9,
                )
                self._ax_map.add_patch(ring)

        # Replan annotations — callout arrows from replan points
        if replan_annotations:
            for ra in replan_annotations:
                rx, ry = ra.get("position", (0, 0))
                reason = ra.get("reason", "replan")
                trigger = ra.get("trigger", "")
                color = "#FFCC00" if trigger == "cadence" else "#FF6644"
                self._ax_map.annotate(
                    f"⟳ {trigger}",
                    xy=(rx, ry),
                    xytext=(rx + 12, ry - 12),
                    fontsize=6, fontweight="bold",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#000000AA", ec=color, lw=0.5),
                    zorder=9,
                )

    # ─── Z10: Cartographic overlay ─────────────────────────────────────

    def _draw_cartographic(self) -> None:
        """Z10 — scale bar, north arrow, coordinate readout, OSM attribution."""
        ax = self._ax_map
        res = self.tile.resolution_m

        # Scale bar (bottom-left)
        bar_cells = int(200.0 / res)  # 200m bar
        bar_label = "200 m"
        if bar_cells > self.W * 0.4:
            bar_cells = int(100.0 / res)
            bar_label = "100 m"

        bx = 10
        by = self.H - 15
        ax.plot(
            [bx, bx + bar_cells], [by, by],
            color="white", linewidth=3, solid_capstyle="butt", zorder=10,
        )
        ax.plot(
            [bx, bx + bar_cells], [by, by],
            color="black", linewidth=1.5, solid_capstyle="butt", zorder=10,
        )
        ax.text(
            bx + bar_cells / 2, by + 5, bar_label,
            fontsize=7, ha="center", va="top",
            color="white", fontweight="bold",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
            zorder=10,
        )

        # North arrow (top-left)
        nx, ny = 15, 20
        ax.annotate(
            "N", xy=(nx, ny - 8), fontsize=9, fontweight="bold",
            ha="center", va="bottom",
            color="white",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
            zorder=10,
        )
        ax.annotate(
            "", xy=(nx, ny - 10), xytext=(nx, ny),
            arrowprops=dict(arrowstyle="->", color="white", lw=2),
            zorder=10,
        )

        # Coordinate readout (bottom-right)
        lat, lon = self.tile.center_latlon
        coord_text = f"{lat:.4f}°N  {lon:.4f}°E"
        ax.text(
            self.W - 5, self.H - 5, coord_text,
            fontsize=6, ha="right", va="bottom",
            color="white",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
            zorder=10,
        )

        # OSM attribution (bottom-right, above coords)
        ax.text(
            self.W - 5, self.H - 15,
            "© OpenStreetMap contributors",
            fontsize=5, ha="right", va="bottom",
            color="#CCCCCC",
            fontstyle="italic",
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
            zorder=10,
        )

    # ─── Z11: HUD panels ──────────────────────────────────────────────

    def _draw_metrics_panel(
        self,
        step: int,
        max_steps: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Draw the right-side metrics panel."""
        ax = self._ax_metrics
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(STAKEHOLDER_PALETTE["panel_bg"][:7])
        for spine in ax.spines.values():
            spine.set_color(STAKEHOLDER_PALETTE["panel_border"])

        m = metrics or {}
        # Title
        ax.text(
            0.5, 0.95, "MISSION STATUS",
            fontsize=10, fontweight="bold", ha="center", va="top",
            color=self.profile.accent_color,
        )

        # KPI rows
        kpis = [
            ("Agency", self.profile.agency),
            ("Mission", self.profile.name),
            ("Planner", self.planner_name.upper()),
            ("Difficulty", self.difficulty.upper()),
            ("Step", f"{step}/{max_steps}"),
            ("Tasks Done", str(m.get("tasks_completed", 0))),
            ("Tasks Pending", str(m.get("tasks_pending", 0))),
            ("Replans", str(m.get("replans", m.get("replanning_count", 0)))),
            ("Violations", str(m.get("violations", m.get("violation_count", 0)))),
            ("Risk ∫", f"{m.get('risk_integral', 0.0):.2f}"),
            ("Energy", f"{m.get('energy_used', 0.0):.1f}%"),
            ("Score", f"{m.get('mission_score', 0.0):.2f}"),
        ]

        y_start = 0.85
        y_step = 0.065
        for i, (label, value) in enumerate(kpis):
            y = y_start - i * y_step
            ax.text(0.08, y, label, fontsize=7, color="#AAAAAA", va="center")
            ax.text(0.92, y, value, fontsize=7, fontweight="bold",
                    color=STAKEHOLDER_PALETTE["panel_text"],
                    ha="right", va="center")
            # Separator line
            ax.plot([0.05, 0.95], [y - 0.025, y - 0.025],
                    color="#333333", linewidth=0.5)

        # Progress bar
        progress = step / max(max_steps, 1)
        bar_y = 0.08
        ax.barh(bar_y, progress, height=0.04, left=0.05,
                color=self.profile.accent_color, alpha=0.8)
        ax.barh(bar_y, 1.0, height=0.04, left=0.05,
                color="#333333", alpha=0.3)
        ax.text(0.5, bar_y, f"{progress*100:.0f}%",
                fontsize=7, ha="center", va="center",
                color="white", fontweight="bold")

    def _draw_timeline(
        self,
        step: int,
        max_steps: int,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Draw the bottom timeline bar."""
        ax = self._ax_timeline
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(STAKEHOLDER_PALETTE["panel_bg"][:7])
        for spine in ax.spines.values():
            spine.set_color(STAKEHOLDER_PALETTE["panel_border"])

        # Progress track
        progress = step / max(max_steps, 1)
        ax.barh(0.5, progress, height=0.3, left=0.02,
                color=self.profile.accent_color, alpha=0.6)
        ax.barh(0.5, 0.96, height=0.3, left=0.02,
                color="#222222", alpha=0.3)

        # Event markers
        events = events or self._events
        for ev in events:
            t = ev.get("step", 0) / max(max_steps, 1)
            if t > 0.98:
                t = 0.98
            ev_type = ev.get("type", "")
            colors = {
                "replan": "#FF4444",
                "task_complete": "#00CC44",
                "task_injected": "#FFCC00",
                "nfz_expand": "#FF00FF",
                "distress": "#FF0044",
            }
            c = colors.get(ev_type, "#888888")
            ax.plot(0.02 + t * 0.96, 0.5, "|", color=c,
                    markersize=12, markeredgewidth=2)

        # Current position marker
        ax.plot(0.02 + progress * 0.96, 0.5, "v",
                color="white", markersize=8, markeredgewidth=1.5)

        # Labels
        ax.text(0.01, 0.15, "START", fontsize=5, color="#888888", va="center")
        ax.text(0.99, 0.15, "END", fontsize=5, color="#888888", ha="right", va="center")
        ax.text(0.5, 0.9, f"Step {step}/{max_steps}", fontsize=7,
                ha="center", va="center", color="white", fontweight="bold")

    def _draw_legend(self) -> None:
        """Draw the bottom-right legend with mission-specific icons."""
        ax = self._ax_legend
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(STAKEHOLDER_PALETTE["panel_bg"][:7])
        for spine in ax.spines.values():
            spine.set_color(STAKEHOLDER_PALETTE["panel_border"])

        ax.text(0.5, 0.92, "LEGEND", fontsize=8, fontweight="bold",
                ha="center", va="top", color=STAKEHOLDER_PALETTE["panel_accent"])

        # Legend entries — mission specific
        profile = self.profile
        entries = [
            (IconID.UAV, "UAV", "#0066FF"),
            (profile.icon_poi, profile.name.split()[0], profile.accent_color),
            (profile.icon_task, "Task", STAKEHOLDER_PALETTE["panel_ok"]),
            (IconID.WAYPOINT, "Waypoint", "#00CC44"),
            (IconID.NFZ, "No-Fly Zone", "#FF00FF"),
        ]

        y_start = 0.75
        y_step = 0.15
        for i, (icon_id, text, color) in enumerate(entries):
            y = y_start - i * y_step
            # Tiny icon using a colored marker
            ax.plot(0.15, y, "s", color=color, markersize=6, markeredgecolor="white",
                    markeredgewidth=0.5)
            ax.text(0.28, y, text, fontsize=6, va="center",
                    color=STAKEHOLDER_PALETTE["panel_text"])

    # ═══ MAIN RENDER ═══════════════════════════════════════════════════

    def render_frame(
        self,
        drone_pos: tuple[int, int],
        step: int,
        max_steps: int,
        *,
        # Dynamic overlays
        fire_mask: np.ndarray | None = None,
        smoke_mask: np.ndarray | None = None,
        nfz_mask: np.ndarray | None = None,
        traffic_closure_mask: np.ndarray | None = None,
        risk_map: np.ndarray | None = None,
        # Mission layer
        pois: Sequence[dict[str, Any]] | None = None,
        entity_positions: Sequence[dict[str, Any]] | None = None,
        # Path
        trajectory: list[tuple[int, int]] | None = None,
        planned_path: list[tuple[int, int]] | None = None,
        completed_segments: int | None = None,
        # UAV state
        heading_deg: float = 0.0,
        safety_radius: float = 8.0,
        # Events
        replan_flash: bool = False,
        replan_reason: str = "",
        invalidated_waypoint: tuple[int, int] | None = None,
        distress_position: tuple[int, int] | None = None,
        # V2 — conflict markers + annotations
        conflict_markers: Sequence[dict[str, Any]] | None = None,
        violation_flash: bool = False,
        replan_annotations: Sequence[dict[str, Any]] | None = None,
        # Dynamic obstacle mask overlay
        dynamic_obstacle_mask: np.ndarray | None = None,
        # Metrics
        metrics: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
        # Flags
        is_keyframe: bool = False,
    ) -> np.ndarray:
        """Render one frame of the 4-pane stakeholder visualization.

        Returns
        -------
        np.ndarray
            RGB image [H, W, 3] uint8.
        """
        # Clear the map axes (keep figure + panel structure)
        self._ax_map.clear()
        self._ax_map.set_xlim(-0.5, self.W - 0.5)
        self._ax_map.set_ylim(self.H - 0.5, -0.5)
        self._ax_map.set_aspect("equal")
        self._ax_map.set_xticks([])
        self._ax_map.set_yticks([])
        for spine in self._ax_map.spines.values():
            spine.set_color(self.profile.accent_color)
            spine.set_linewidth(2)

        # ── Draw layers in z-order ──
        self._draw_basemap()                                          # Z1
        self._draw_risk_overlay(risk_map)                             # Z2
        self._draw_fire_overlay(fire_mask, smoke_mask, step)          # Z3
        self._draw_restricted_zones(nfz_mask, traffic_closure_mask)   # Z4
        # Dynamic obstacle mask overlay (semi-transparent orange)
        if dynamic_obstacle_mask is not None and dynamic_obstacle_mask.any():
            dyn_rgba = np.zeros((*dynamic_obstacle_mask.shape, 4), dtype=np.float32)
            dyn_rgba[dynamic_obstacle_mask] = [1.0, 0.5, 0.0, 0.25]
            self._ax_map.imshow(
                dyn_rgba,
                origin="upper",
                extent=(-0.5, self.W - 0.5, self.H - 0.5, -0.5),
                zorder=4,
                interpolation="nearest",
            )
        self._draw_pois(pois)                                         # Z5
        self._draw_entities(entity_positions)                         # Z6
        self._draw_path(trajectory, planned_path, completed_segments) # Z7
        self._draw_uav(drone_pos, heading_deg, safety_radius)        # Z8
        self._draw_events(step, replan_flash, replan_reason,          # Z9
                          invalidated_waypoint, distress_position,
                          conflict_markers, violation_flash,
                          replan_annotations)
        self._draw_cartographic()                                     # Z10

        # ── HUD panels ──
        self._draw_metrics_panel(step, max_steps, metrics)            # Z11
        self._draw_timeline(step, max_steps, events)                  # Z11
        self._draw_legend()                                           # Z11

        # ── Rasterise ──
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()

        self._frames.append(frame)
        if is_keyframe:
            self._keyframe_indices.append(len(self._frames) - 1)

        return frame

    # ═══ EVENT LOGGING ═════════════════════════════════════════════════

    def log_event(self, step: int, event_type: str, detail: str = "") -> None:
        """Log a mission event for the timeline bar."""
        self._events.append({"step": step, "type": event_type, "detail": detail})

    # ═══ EXPORT ════════════════════════════════════════════════════════

    def save_frame(self, path: Path, frame: np.ndarray | None = None) -> None:
        """Save a single frame as PNG."""
        img = frame if frame is not None else (self._frames[-1] if self._frames else None)
        if img is None:
            return
        from PIL import Image
        Image.fromarray(img).save(str(path))

    def export_mp4(
        self,
        path: Path,
        fps: int = 10,
    ) -> None:
        """Export all frames as 1080p MP4."""
        if not self._frames:
            return

        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(self._frames):
                from PIL import Image
                Image.fromarray(frame).save(f"{tmpdir}/frame_{i:06d}.png")

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", f"{tmpdir}/frame_%06d.png",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "slow",
                str(path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

    def export_gif(self, path: Path, fps: int = 8, max_frames: int = 200) -> None:
        """Export frames as animated GIF."""
        if not self._frames:
            return
        from PIL import Image
        frames = self._frames[:max_frames]
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            str(path),
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    def export_keyframes(self, output_dir: Path) -> list[Path]:
        """Export keyframe PNGs to a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for idx in self._keyframe_indices:
            if idx < len(self._frames):
                p = output_dir / f"keyframe_{idx:04d}.png"
                self.save_frame(p, self._frames[idx])
                paths.append(p)
        return paths

    def export_metadata(self, path: Path, extra: dict | None = None) -> None:
        """Export episode metadata as JSON."""
        # Resolve git commit SHA for reproducibility
        git_commit_sha = "unknown"
        try:
            import subprocess
            git_commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass

        meta = {
            "mission_type": self.mission_type,
            "profile": {
                "name": self.profile.name,
                "name_el": self.profile.name_el,
                "agency": self.profile.agency,
                "agency_el": self.profile.agency_el,
                "tile_id": self.profile.tile_id,
            },
            "scenario_id": self.scenario_id,
            "planner": self.planner_name,
            "difficulty": self.difficulty,
            "tile": {
                "id": self.tile.tile_id,
                "center_latlon": list(self.tile.center_latlon),
                "resolution_m": self.tile.resolution_m,
                "grid_size": self.tile.grid_size,
                "crs": self.tile.crs,
            },
            "total_frames": len(self._frames),
            "keyframe_count": len(self._keyframe_indices),
            "events": self._events,
            "git_commit_sha": git_commit_sha,
            "attribution": "Map data © OpenStreetMap contributors (ODbL)",
        }
        if extra:
            meta.update(extra)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        """Close the matplotlib figure and free memory."""
        if hasattr(self, "_fig") and self._fig is not None:
            plt.close(self._fig)
            self._fig = None
