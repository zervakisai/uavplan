"""Styled basemap rendering — GIS-quality cartography for UAVBench.

Shared by both ``OperationalRenderer`` and ``StakeholderRenderer``.

Features
--------
- Landuse fills (ground / forest / urban / industrial / water)
- Road hierarchy via morphological erosion (primary / secondary / minor)
- Building fill + edge outline
- Analytical hillshade (pure numpy, optional)
- Water coastline detection (scipy optional)
- ODbL attribution text

All functions are pure numpy (+ optional scipy), no geopandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BasemapStyleConfig:
    """All visual knobs for the styled basemap."""

    # Landuse fills (hex colours)
    color_ground: str = "#F0EDE6"
    color_residential: str = "#E8D8C8"
    color_commercial: str = "#D8D0C8"
    color_vegetation: str = "#B8D8A8"
    color_water: str = "#A4C8E1"
    color_industrial: str = "#D5CFC5"

    # Buildings
    color_building_fill: str = "#5A5A5A"
    color_building_edge: str = "#3D3D3D"
    building_edge_width: int = 1  # erosion iterations for edge detection

    # Road tiers
    color_road_primary: str = "#C8C8C8"
    color_road_secondary: str = "#AAAAAA"
    color_road_minor: str = "#999999"

    # Water coastline
    color_coastline: str = "#6B9FC8"
    coastline_darken: float = 0.75

    # Hillshade
    hillshade_enabled: bool = True
    hillshade_azimuth: float = 315.0  # degrees, NW light
    hillshade_altitude: float = 45.0  # degrees above horizon
    hillshade_alpha: float = 0.25
    hillshade_z_factor: float = 2.0  # vertical exaggeration

    # Attribution
    attribution_text: str = "\u00a9 OpenStreetMap contributors (ODbL)"

    # Landuse code mapping: code -> palette key
    # 0=other, 1=residential, 2=commercial, 3=vegetation, 4=water
    landuse_codes: tuple[tuple[int, str], ...] = (
        (0, "ground"),
        (1, "residential"),
        (2, "commercial"),
        (3, "vegetation"),
        (4, "water"),
    )


DEFAULT_STYLE = BasemapStyleConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> np.ndarray:
    """Convert '#RRGGBB' or '#RRGGBBAA' → float32 [3]."""
    h = h.lstrip("#")
    if len(h) == 8:
        h = h[:6]
    return np.array(
        [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=np.float32,
    )


def _style_color(style: BasemapStyleConfig, landuse_key: str) -> np.ndarray:
    """Look up a landuse key → RGB float32 [3]."""
    mapping = {
        "ground": style.color_ground,
        "residential": style.color_residential,
        "commercial": style.color_commercial,
        "vegetation": style.color_vegetation,
        "water": style.color_water,
        "industrial": style.color_industrial,
    }
    return _hex_to_rgb(mapping.get(landuse_key, style.color_ground))


# ─────────────────────────────────────────────────────────────────────────────
# Hillshade
# ─────────────────────────────────────────────────────────────────────────────

def compute_hillshade(
    heightmap: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 2.0,
) -> np.ndarray:
    """Analytical hillshade from a DEM / building heightmap.

    Parameters
    ----------
    heightmap : ndarray [H, W] float
        Elevation values (building heights work fine).
    azimuth : float
        Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W).
    altitude : float
        Sun altitude in degrees above horizon.
    z_factor : float
        Vertical exaggeration.

    Returns
    -------
    ndarray [H, W] float32 in [0, 1]
        0 = fully shadowed, 1 = fully lit.
    """
    hm = heightmap.astype(np.float64) * z_factor

    # Terrain gradients
    dy, dx = np.gradient(hm)

    # Slope and aspect
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dy, dx)

    # Light direction
    az_rad = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)

    shaded = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )

    return np.clip(shaded, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Water edge detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_water_edges(
    landuse_map: np.ndarray,
    water_code: int = 4,
) -> np.ndarray:
    """Detect 1-2 px coastline around water bodies.

    Parameters
    ----------
    landuse_map : ndarray [H, W] int
        Landuse classification (water_code = water pixels).

    Returns
    -------
    ndarray [H, W] bool
        True at coastline pixels.
    """
    water = landuse_map == water_code
    if not water.any():
        return np.zeros_like(water, dtype=bool)

    try:
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(water, iterations=1)
        return water & ~eroded
    except ImportError:
        # Fallback: manual 4-connected edge detection
        edge = np.zeros_like(water, dtype=bool)
        edge[:-1, :] |= water[:-1, :] & ~water[1:, :]
        edge[1:, :] |= water[1:, :] & ~water[:-1, :]
        edge[:, :-1] |= water[:, :-1] & ~water[:, 1:]
        edge[:, 1:] |= water[:, 1:] & ~water[:, :-1]
        return edge


# ─────────────────────────────────────────────────────────────────────────────
# Road tier classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_road_tiers(
    roads_mask: np.ndarray,
    building_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Classify road pixels into primary / secondary / minor via erosion.

    Wide roads survive more erosion → primary.  Narrow survive less → minor.

    Parameters
    ----------
    roads_mask : ndarray [H, W] bool
    building_mask : ndarray [H, W] bool

    Returns
    -------
    dict with keys "primary", "secondary", "minor" → bool masks
    """
    road_only = roads_mask & ~building_mask

    try:
        from scipy.ndimage import binary_erosion

        eroded_1 = binary_erosion(road_only, iterations=1)
        eroded_2 = binary_erosion(road_only, iterations=2)

        primary = eroded_2
        secondary = eroded_1 & ~eroded_2
        minor = road_only & ~eroded_1
    except ImportError:
        # No scipy — single tier fallback
        primary = road_only
        secondary = np.zeros_like(road_only, dtype=bool)
        minor = np.zeros_like(road_only, dtype=bool)

    return {"primary": primary, "secondary": secondary, "minor": minor}


# ─────────────────────────────────────────────────────────────────────────────
# Main basemap builder
# ─────────────────────────────────────────────────────────────────────────────

def build_styled_basemap(
    heightmap: np.ndarray,
    roads_mask: Optional[np.ndarray] = None,
    *,
    landuse_map: Optional[np.ndarray] = None,
    style: Optional[BasemapStyleConfig] = None,
) -> np.ndarray:
    """Build a styled RGB basemap [H, W, 3] float32.

    Layer order: landuse fill → roads by tier → buildings (fill+edge)
                 → hillshade blend → water coastline.

    Parameters
    ----------
    heightmap : ndarray [H, W]
        Building heights / DEM.
    roads_mask : ndarray [H, W] bool, optional
        Road pixels.
    landuse_map : ndarray [H, W] int, optional
        Landuse classification codes.
    style : BasemapStyleConfig, optional
        Visual configuration.  Defaults to ``DEFAULT_STYLE``.

    Returns
    -------
    ndarray [H, W, 3] float32 in [0, 1]
    """
    if style is None:
        style = DEFAULT_STYLE

    H, W = heightmap.shape
    base = np.full((H, W, 3), _hex_to_rgb(style.color_ground), dtype=np.float32)

    # ── 1. Landuse fills ──────────────────────────────────────────────────
    if landuse_map is not None:
        for code, key in style.landuse_codes:
            mask = landuse_map == code
            if mask.any():
                base[mask] = _style_color(style, key)

    # ── 2. Roads by tier ──────────────────────────────────────────────────
    building = heightmap > 0
    if roads_mask is not None:
        tiers = classify_road_tiers(roads_mask, building)
        if tiers["primary"].any():
            base[tiers["primary"]] = _hex_to_rgb(style.color_road_primary)
        if tiers["secondary"].any():
            base[tiers["secondary"]] = _hex_to_rgb(style.color_road_secondary)
        if tiers["minor"].any():
            base[tiers["minor"]] = _hex_to_rgb(style.color_road_minor)

    # ── 3. Buildings: fill + edge ─────────────────────────────────────────
    base[building] = _hex_to_rgb(style.color_building_fill)
    try:
        from scipy.ndimage import binary_erosion

        interior = binary_erosion(building, iterations=style.building_edge_width)
        edge = building & ~interior
        base[edge] = _hex_to_rgb(style.color_building_edge)
    except ImportError:
        pass

    # ── 4. Hillshade blend ────────────────────────────────────────────────
    if style.hillshade_enabled and np.any(heightmap > 0):
        shade = compute_hillshade(
            heightmap,
            azimuth=style.hillshade_azimuth,
            altitude=style.hillshade_altitude,
            z_factor=style.hillshade_z_factor,
        )
        # Blend: darken shadowed areas, brighten lit areas
        alpha = style.hillshade_alpha
        shade_3d = shade[:, :, np.newaxis]
        base = base * (1.0 - alpha) + base * shade_3d * alpha
        base = np.clip(base, 0.0, 1.0)

    # ── 5. Water coastline ────────────────────────────────────────────────
    if landuse_map is not None:
        coastline = detect_water_edges(landuse_map, water_code=4)
        if coastline.any():
            base[coastline] = _hex_to_rgb(style.color_coastline)

    return base
