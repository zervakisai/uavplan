"""Vector icon library — matplotlib path-based pictograms.

Icons are defined as unit-square path vertices (0..1 × 0..1) and rendered
as matplotlib ``PathPatch`` artists.  This avoids any external image files,
scales to arbitrary resolution, and works with the Agg backend headlessly.

Every icon has:
    - path vertices (Nx2 float32, unit square)
    - path codes  (N int, matplotlib Path codes)
    - default fill / stroke colour (hex)

Supported icons (IconID enum):
    UAV, FIRE, SHIP, BUILDING, WAYPOINT, WAYPOINT_DONE, NFZ, PERSON,
    DISTRESS, ANCHOR, CAMERA, SHIELD, ALERT, HOME, WIND, RADIO
"""

from __future__ import annotations

import enum
from typing import Optional, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Icon IDs
# ─────────────────────────────────────────────────────────────────────────────

class IconID(str, enum.Enum):
    """Canonical icon identifiers."""
    UAV = "uav"
    FIRE = "fire"
    SHIP = "ship"
    BUILDING = "building"
    WAYPOINT = "waypoint"
    WAYPOINT_DONE = "waypoint_done"
    NFZ = "nfz"
    PERSON = "person"
    DISTRESS = "distress"
    ANCHOR = "anchor"
    CAMERA = "camera"
    SHIELD = "shield"
    ALERT = "alert"
    HOME = "home"
    WIND = "wind"
    RADIO = "radio"
    INSPECTION = "inspection"
    PATROL = "patrol"
    HAZARD = "hazard"
    CORRIDOR = "corridor"


# ─────────────────────────────────────────────────────────────────────────────
# Path definitions — unit square (0–1) coordinates
# ─────────────────────────────────────────────────────────────────────────────

MOVETO = mpath.Path.MOVETO if HAS_MPL else 1
LINETO = mpath.Path.LINETO if HAS_MPL else 2
CLOSEPOLY = mpath.Path.CLOSEPOLY if HAS_MPL else 79
CURVE3 = mpath.Path.CURVE3 if HAS_MPL else 3
CURVE4 = mpath.Path.CURVE4 if HAS_MPL else 4


def _triangle_up():
    """Upward-pointing triangle (UAV top-down)."""
    verts = [(0.5, 0.95), (0.15, 0.15), (0.5, 0.35), (0.85, 0.15), (0.5, 0.95), (0, 0)]
    codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _flame():
    """Flame icon — teardrop with inner flicker."""
    verts = [
        (0.5, 0.95),   # top
        (0.25, 0.55),  # left shoulder
        (0.15, 0.30),  # left base
        (0.35, 0.10),  # inner left
        (0.5, 0.25),   # inner dip
        (0.65, 0.10),  # inner right
        (0.85, 0.30),  # right base
        (0.75, 0.55),  # right shoulder
        (0.5, 0.95),   # back to top
        (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _ship():
    """Top-down ship/vessel silhouette."""
    verts = [
        (0.5, 0.92),   # bow
        (0.75, 0.65),  # starboard bow
        (0.78, 0.35),  # starboard mid
        (0.70, 0.10),  # starboard stern
        (0.30, 0.10),  # port stern
        (0.22, 0.35),  # port mid
        (0.25, 0.65),  # port bow
        (0.5, 0.92),   # back to bow
        (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _building_icon():
    """Building footprint — rectangle with roof line."""
    verts = [
        (0.20, 0.10),  # BL
        (0.20, 0.70),  # TL
        (0.50, 0.90),  # roof peak
        (0.80, 0.70),  # TR
        (0.80, 0.10),  # BR
        (0.20, 0.10),  # close
        (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _circle_marker():
    """Circle approximated as 12-gon for waypoint."""
    n = 12
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    cx, cy, r = 0.5, 0.5, 0.4
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    verts.append(verts[0])
    verts.append((0, 0))
    codes = [MOVETO] + [LINETO] * (n - 1) + [LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _checkmark():
    """Check mark inside circle (waypoint done)."""
    # circle
    n = 12
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    cx, cy, r = 0.5, 0.5, 0.4
    c_verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    c_verts.append(c_verts[0])
    c_verts.append((0, 0))
    c_codes = [MOVETO] + [LINETO] * (n - 1) + [LINETO, CLOSEPOLY]
    return np.array(c_verts, dtype=np.float32), c_codes


def _octagon():
    """Octagon for NFZ / stop-sign shape."""
    n = 8
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1] + np.pi / 8
    cx, cy, r = 0.5, 0.5, 0.45
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    verts.append(verts[0])
    verts.append((0, 0))
    codes = [MOVETO] + [LINETO] * (n - 1) + [LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _person():
    """Simple person silhouette."""
    verts = [
        # head (small circle-ish triangle)
        (0.45, 0.90), (0.55, 0.90), (0.50, 0.80), (0.45, 0.90), (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    # body (separate shape rendered as second patch)
    body_verts = [
        (0.50, 0.78),  # neck
        (0.30, 0.50),  # left arm tip
        (0.40, 0.48),  # left arm join
        (0.42, 0.30),  # left leg
        (0.50, 0.10),  # feet
        (0.58, 0.30),  # right leg
        (0.60, 0.48),  # right arm join
        (0.70, 0.50),  # right arm tip
        (0.50, 0.78),
        (0, 0),
    ]
    body_codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    # combine head + body into one path
    all_verts = list(verts[:-1]) + body_verts  # drop (0,0) close from head
    all_codes = codes[:-1] + body_codes
    # Actually simpler to just use body
    return np.array(body_verts, dtype=np.float32), body_codes


def _diamond():
    """Diamond shape for distress / SOS."""
    verts = [
        (0.5, 0.95), (0.85, 0.5), (0.5, 0.05), (0.15, 0.5), (0.5, 0.95), (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _anchor():
    """Anchor — simplified T shape with hook."""
    verts = [
        (0.45, 0.90), (0.55, 0.90),  # top crossbar
        (0.55, 0.80), (0.70, 0.80),   # right arm
        (0.70, 0.72), (0.55, 0.72),   # right arm bottom
        (0.55, 0.25),                  # shaft down
        (0.65, 0.15), (0.55, 0.05),   # right hook
        (0.45, 0.05), (0.35, 0.15),   # left hook
        (0.45, 0.25),                  # shaft up
        (0.45, 0.72), (0.30, 0.72),   # left arm bottom
        (0.30, 0.80), (0.45, 0.80),   # left arm top
        (0.45, 0.90), (0, 0),
    ]
    codes = [MOVETO] + [LINETO] * 16 + [CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _camera():
    """Camera icon — rectangle with lens circle."""
    verts = [
        (0.15, 0.25), (0.15, 0.65),
        (0.35, 0.75), (0.65, 0.75),
        (0.85, 0.65), (0.85, 0.25),
        (0.15, 0.25), (0, 0),
    ]
    codes = [MOVETO] + [LINETO] * 6 + [CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _shield():
    """Shield shape."""
    verts = [
        (0.5, 0.05),
        (0.15, 0.30), (0.15, 0.65),
        (0.5, 0.95),
        (0.85, 0.65), (0.85, 0.30),
        (0.5, 0.05), (0, 0),
    ]
    codes = [MOVETO] + [LINETO] * 6 + [CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _triangle_alert():
    """Warning triangle with ! inside."""
    verts = [
        (0.5, 0.92), (0.10, 0.10), (0.90, 0.10), (0.5, 0.92), (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _home():
    """House silhouette."""
    verts = [
        (0.50, 0.90),  # roof peak
        (0.15, 0.50),  # left roof edge
        (0.15, 0.10),  # BL
        (0.85, 0.10),  # BR
        (0.85, 0.50),  # right roof edge
        (0.50, 0.90),  # back to peak
        (0, 0),
    ]
    codes = [MOVETO, LINETO, LINETO, LINETO, LINETO, LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _arrow_right():
    """Right-pointing arrow for wind."""
    verts = [
        (0.10, 0.40), (0.60, 0.40),
        (0.60, 0.20), (0.90, 0.50),
        (0.60, 0.80), (0.60, 0.60),
        (0.10, 0.60), (0.10, 0.40), (0, 0),
    ]
    codes = [MOVETO] + [LINETO] * 7 + [CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _antenna():
    """Radio / antenna icon."""
    verts = [
        (0.45, 0.10), (0.55, 0.10),  # base
        (0.55, 0.60),  # shaft
        (0.70, 0.80), (0.75, 0.85),  # right signal
        (0.55, 0.65),  # back to shaft
        (0.55, 0.70),
        (0.30, 0.85), (0.25, 0.80),  # left signal
        (0.45, 0.65),  # back
        (0.45, 0.10), (0, 0),
    ]
    codes = [MOVETO] + [LINETO] * 10 + [CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _magnifier():
    """Magnifying glass for inspection."""
    n = 10
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    cx, cy, r = 0.45, 0.55, 0.30
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    # handle
    verts.append(verts[0])
    verts += [(0.68, 0.32), (0.85, 0.15), (0.78, 0.08), (0.62, 0.25)]
    verts.append(verts[0])
    verts.append((0, 0))
    codes = [MOVETO] + [LINETO] * (n - 1) + [LINETO] * 5 + [LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _crosshair():
    """Patrol crosshair / radar sweep."""
    # outer circle
    n = 12
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    cx, cy, r = 0.5, 0.5, 0.42
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    verts.append(verts[0])
    verts.append((0, 0))
    codes = [MOVETO] + [LINETO] * (n - 1) + [LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _hexagon():
    """Hazard hex."""
    n = 6
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1] + np.pi / 6
    cx, cy, r = 0.5, 0.5, 0.45
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    verts.append(verts[0])
    verts.append((0, 0))
    codes = [MOVETO] + [LINETO] * (n - 1) + [LINETO, CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


def _double_arrow():
    """Corridor — double-headed arrow."""
    verts = [
        (0.20, 0.50), (0.40, 0.80), (0.40, 0.60),
        (0.60, 0.60), (0.60, 0.80), (0.80, 0.50),
        (0.60, 0.20), (0.60, 0.40), (0.40, 0.40),
        (0.40, 0.20), (0.20, 0.50), (0, 0),
    ]
    codes = [MOVETO] + [LINETO] * 10 + [CLOSEPOLY]
    return np.array(verts, dtype=np.float32), codes


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_ICON_BUILDERS: dict[str, callable] = {
    IconID.UAV: _triangle_up,
    IconID.FIRE: _flame,
    IconID.SHIP: _ship,
    IconID.BUILDING: _building_icon,
    IconID.WAYPOINT: _circle_marker,
    IconID.WAYPOINT_DONE: _checkmark,
    IconID.NFZ: _octagon,
    IconID.PERSON: _person,
    IconID.DISTRESS: _diamond,
    IconID.ANCHOR: _anchor,
    IconID.CAMERA: _camera,
    IconID.SHIELD: _shield,
    IconID.ALERT: _triangle_alert,
    IconID.HOME: _home,
    IconID.WIND: _arrow_right,
    IconID.RADIO: _antenna,
    IconID.INSPECTION: _magnifier,
    IconID.PATROL: _crosshair,
    IconID.HAZARD: _hexagon,
    IconID.CORRIDOR: _double_arrow,
}

# Default colours per icon
_ICON_DEFAULTS: dict[str, dict] = {
    IconID.UAV:           {"fill": "#0066FF", "edge": "#003399", "alpha": 1.0},
    IconID.FIRE:          {"fill": "#FF2D00", "edge": "#CC2200", "alpha": 0.9},
    IconID.SHIP:          {"fill": "#4488FF", "edge": "#2255CC", "alpha": 0.9},
    IconID.BUILDING:      {"fill": "#5A5A5A", "edge": "#3D3D3D", "alpha": 0.85},
    IconID.WAYPOINT:      {"fill": "#00CC44", "edge": "#009933", "alpha": 0.85},
    IconID.WAYPOINT_DONE: {"fill": "#00CC44", "edge": "#006622", "alpha": 0.5},
    IconID.NFZ:           {"fill": "#FF00FF", "edge": "#CC00CC", "alpha": 0.7},
    IconID.PERSON:        {"fill": "#FFD700", "edge": "#CC9900", "alpha": 0.9},
    IconID.DISTRESS:      {"fill": "#FF0044", "edge": "#CC0033", "alpha": 0.95},
    IconID.ANCHOR:        {"fill": "#334155", "edge": "#1A2030", "alpha": 0.8},
    IconID.CAMERA:        {"fill": "#00DDFF", "edge": "#0099CC", "alpha": 0.85},
    IconID.SHIELD:        {"fill": "#3388FF", "edge": "#2266CC", "alpha": 0.8},
    IconID.ALERT:         {"fill": "#FF6B35", "edge": "#CC5500", "alpha": 0.9},
    IconID.HOME:          {"fill": "#00CC44", "edge": "#009933", "alpha": 0.8},
    IconID.WIND:          {"fill": "#888888", "edge": "#555555", "alpha": 0.6},
    IconID.RADIO:         {"fill": "#3388FF", "edge": "#2266CC", "alpha": 0.7},
    IconID.INSPECTION:    {"fill": "#00DDFF", "edge": "#0099BB", "alpha": 0.85},
    IconID.PATROL:        {"fill": "#4488FF", "edge": "#2255CC", "alpha": 0.8},
    IconID.HAZARD:        {"fill": "#FF6B35", "edge": "#CC5500", "alpha": 0.85},
    IconID.CORRIDOR:      {"fill": "#00E5FF", "edge": "#00AACC", "alpha": 0.7},
}


# ─────────────────────────────────────────────────────────────────────────────
# IconLibrary — stamp icons onto matplotlib Axes
# ─────────────────────────────────────────────────────────────────────────────

class IconLibrary:
    """Resolution-independent icon renderer for matplotlib Axes.

    Parameters
    ----------
    icon_size : float
        Size of icons in data-coordinate units (grid cells).
    """

    def __init__(self, icon_size: float = 8.0) -> None:
        if not HAS_MPL:
            raise ImportError("matplotlib is required for IconLibrary")
        self.icon_size = icon_size
        self._path_cache: dict[str, mpath.Path] = {}

    def _get_path(self, icon_id: str) -> mpath.Path:
        """Build and cache a matplotlib Path for the given icon."""
        if icon_id not in self._path_cache:
            builder = _ICON_BUILDERS.get(icon_id)
            if builder is None:
                raise KeyError(f"Unknown icon: {icon_id!r}.  "
                               f"Available: {sorted(_ICON_BUILDERS)}")
            verts, codes = builder()
            self._path_cache[icon_id] = mpath.Path(verts, codes)
        return self._path_cache[icon_id]

    def stamp(
        self,
        icon_id: str | IconID,
        center: tuple[float, float],
        ax: Axes,
        *,
        size: float | None = None,
        color: str | None = None,
        edge_color: str | None = None,
        alpha: float | None = None,
        rotation_deg: float = 0.0,
        zorder: int = 8,
        label: str | None = None,
        label_offset: tuple[float, float] = (0, -0.6),
        label_fontsize: float = 6.0,
    ) -> list[Artist]:
        """Stamp an icon onto the axes and return the artist(s).

        Parameters
        ----------
        icon_id : str or IconID
            Which icon to draw.
        center : (x, y)
            Centre of the icon in data coordinates.
        ax : matplotlib Axes
            Target axes.
        size : float, optional
            Override icon size (data-coord units).
        color : str, optional
            Fill colour (hex). Defaults to icon's canonical colour.
        edge_color : str, optional
            Edge colour. Defaults to icon's canonical.
        alpha : float, optional
            Opacity.
        rotation_deg : float
            Rotation in degrees (CCW).
        zorder : int
            matplotlib z-order.
        label : str, optional
            Text label placed below the icon.
        label_offset : (dx, dy)
            Offset of label from icon centre (units of icon_size).
        label_fontsize : float
            Label font size in points.

        Returns
        -------
        list[Artist]
            The PathPatch and optional Text artists added to *ax*.
        """
        icon_id = str(icon_id.value if isinstance(icon_id, IconID) else icon_id)
        defaults = _ICON_DEFAULTS.get(icon_id, {"fill": "#888888", "edge": "#444444", "alpha": 0.8})
        sz = size or self.icon_size
        fc = color or defaults["fill"]
        ec = edge_color or defaults["edge"]
        a = alpha if alpha is not None else defaults["alpha"]

        path = self._get_path(icon_id)

        # Transform: scale to `sz` and translate so centre is at `center`
        import matplotlib.transforms as mtransforms

        cx, cy = center
        t = (mtransforms.Affine2D()
             .translate(-0.5, -0.5)          # centre to origin
             .rotate_deg(rotation_deg)        # rotate
             .scale(sz)                       # scale
             .translate(cx, cy))              # move to position

        transformed = path.transformed(t)
        patch = mpatches.PathPatch(
            transformed,
            facecolor=fc,
            edgecolor=ec,
            linewidth=max(0.5, sz / 10),
            alpha=a,
            zorder=zorder,
        )
        ax.add_patch(patch)
        artists: list[Artist] = [patch]

        if label:
            lx = cx + label_offset[0] * sz
            ly = cy + label_offset[1] * sz
            txt = ax.text(
                lx, ly, label,
                fontsize=label_fontsize,
                ha="center", va="top",
                color=ec,
                fontweight="bold",
                zorder=zorder + 1,
                clip_on=True,
            )
            artists.append(txt)

        return artists

    def stamp_many(
        self,
        icon_id: str | IconID,
        positions: Sequence[tuple[float, float]],
        ax: Axes,
        **kwargs,
    ) -> list[list[Artist]]:
        """Stamp the same icon at multiple positions."""
        return [self.stamp(icon_id, pos, ax, **kwargs) for pos in positions]

    @staticmethod
    def available_icons() -> list[str]:
        """Return sorted list of available icon IDs."""
        return sorted(_ICON_BUILDERS.keys())
