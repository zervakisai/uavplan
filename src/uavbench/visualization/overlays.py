"""Mission-specific dynamic overlays for stakeholder visualization.

Provides overlay generators for the three mission types:
  - M1 Civil Protection: fire perimeter evolution, smoke drift, corridor
  - M2 Maritime Domain: vessel AIS tracks, distress pulse, search sectors
  - M3 Critical Infrastructure: inspection zones, restriction rings, patrol arcs

Each overlay function returns data structures consumed by
``StakeholderRenderer.render_frame()`` — no direct matplotlib calls.

Usage::

    from uavbench.visualization.overlays import (
        CivilProtectionOverlay,
        MaritimeDomainOverlay,
        CriticalInfraOverlay,
    )

    overlay = CivilProtectionOverlay(grid_shape=(500, 500))
    frame_data = overlay.compute(step=42, engine=engine)
    renderer.render_frame(drone_pos, step, max_steps, **frame_data)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from uavbench.visualization.icons import IconID


# ─────────────────────────────────────────────────────────────────────────────
# Base overlay
# ─────────────────────────────────────────────────────────────────────────────

class MissionOverlay:
    """Base class for mission-specific overlays.

    Subclasses implement ``compute()`` which returns a dict of keyword
    arguments for ``StakeholderRenderer.render_frame()``.
    """

    def __init__(self, grid_shape: tuple[int, int] = (500, 500)) -> None:
        self.H, self.W = grid_shape

    def compute(
        self,
        step: int,
        *,
        engine: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute overlay data for a single frame.

        Returns dict of kwargs for ``render_frame()``.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# M1: Civil Protection — Wildfire
# ─────────────────────────────────────────────────────────────────────────────

class CivilProtectionOverlay(MissionOverlay):
    """Fire perimeter, smoke drift, emergency corridor, and task POIs.

    Parameters
    ----------
    grid_shape : (H, W)
        Grid dimensions.
    fire_origins : list of (y, x)
        Initial fire ignition points.
    fire_spread_rate : float
        Cells per step the fire perimeter expands.
    wind_direction_deg : float
        Wind direction (degrees, 0=N, 90=E).
    wind_speed : float
        Wind factor (0–1).
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        fire_origins: Sequence[tuple[int, int]] | None = None,
        fire_spread_rate: float = 0.5,
        wind_direction_deg: float = 225.0,
        wind_speed: float = 0.4,
    ) -> None:
        super().__init__(grid_shape)
        self.fire_origins = list(fire_origins or [(self.H // 4, self.W // 4)])
        self.fire_spread_rate = fire_spread_rate
        self.wind_direction_deg = wind_direction_deg
        self.wind_speed = wind_speed
        # Internal state
        self._fire_mask = np.zeros((self.H, self.W), dtype=bool)
        self._smoke_mask = np.zeros((self.H, self.W), dtype=bool)
        for y, x in self.fire_origins:
            if 0 <= y < self.H and 0 <= x < self.W:
                self._fire_mask[y, x] = True

    def _expand_fire(self, step: int) -> None:
        """Expand fire perimeter with wind bias."""
        if not self._fire_mask.any():
            return

        try:
            from scipy.ndimage import binary_dilation
        except ImportError:
            return

        # Wind-biased structuring element
        rad = math.radians(self.wind_direction_deg)
        dy, dx = -math.cos(rad), math.sin(rad)  # N=0

        struct = np.ones((3, 3), dtype=bool)
        # Expand every N steps based on spread rate
        if step % max(1, int(1.0 / self.fire_spread_rate)) == 0:
            new_fire = binary_dilation(self._fire_mask, structure=struct)
            # Wind bias: shift expansion
            shift_y = int(round(dy * self.wind_speed * 2))
            shift_x = int(round(dx * self.wind_speed * 2))
            wind_fire = np.roll(np.roll(new_fire, shift_y, axis=0), shift_x, axis=1)
            self._fire_mask = new_fire | wind_fire

        # Smoke: dilated fire shifted downwind
        smoke_shift_y = int(5 * dy * (1 + 0.5 * math.sin(step * 0.1)))
        smoke_shift_x = int(5 * dx * (1 + 0.5 * math.sin(step * 0.1)))
        dilated = binary_dilation(self._fire_mask, iterations=3)
        self._smoke_mask = np.roll(
            np.roll(dilated, smoke_shift_y, axis=0),
            smoke_shift_x, axis=1,
        ) & ~self._fire_mask

    def compute(
        self,
        step: int,
        *,
        engine: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._expand_fire(step)

        # Build POI list from engine if available
        pois = []
        if engine is not None:
            for t in getattr(engine, "tasks", []):
                spec = t.spec if hasattr(t, "spec") else t
                status = "pending"
                if hasattr(t, "status"):
                    status = t.status.value if hasattr(t.status, "value") else str(t.status)
                category = getattr(spec, "category", "")
                icon = IconID.FIRE if "perimeter" in category else IconID.CAMERA
                pois.append({
                    "xy": spec.xy,
                    "icon": icon,
                    "label": getattr(spec, "task_id", ""),
                    "status": status,
                })

        # Emergency vehicles from extra
        entities = []
        if extra and "vehicles" in extra:
            for v in extra["vehicles"]:
                entities.append({
                    "xy": v["xy"],
                    "type": v.get("type", "fire"),
                    "heading": v.get("heading", 0.0),
                    "trail": v.get("trail", []),
                })

        return {
            "fire_mask": self._fire_mask.copy(),
            "smoke_mask": self._smoke_mask.copy(),
            "pois": pois,
            "entity_positions": entities,
        }


# ─────────────────────────────────────────────────────────────────────────────
# M2: Maritime Domain — SAR
# ─────────────────────────────────────────────────────────────────────────────

class MaritimeDomainOverlay(MissionOverlay):
    """Vessel tracks, distress beacons, search sectors, hazard zones.

    Parameters
    ----------
    grid_shape : (H, W)
        Grid dimensions.
    patrol_center : (y, x)
        Centre of the patrol area.
    patrol_radius : float
        Radius of the circular patrol route (cells).
    num_vessels : int
        Number of simulated vessels.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        patrol_center: tuple[int, int] | None = None,
        patrol_radius: float = 150.0,
        num_vessels: int = 3,
    ) -> None:
        super().__init__(grid_shape)
        self.patrol_center = patrol_center or (self.H // 2, self.W // 2)
        self.patrol_radius = patrol_radius
        self.num_vessels = num_vessels
        # Vessel state: position, heading, trail
        self._vessels: list[dict] = []
        self._init_vessels()
        # Distress state
        self._distress_active = False
        self._distress_pos: tuple[int, int] | None = None
        self._distress_step: int = 0

    def _init_vessels(self) -> None:
        """Initialise vessel positions around patrol area."""
        cy, cx = self.patrol_center
        for i in range(self.num_vessels):
            angle = 2 * math.pi * i / self.num_vessels
            vy = cy + self.patrol_radius * 0.6 * math.sin(angle)
            vx = cx + self.patrol_radius * 0.6 * math.cos(angle)
            self._vessels.append({
                "xy": (int(vx), int(vy)),
                "heading": math.degrees(angle),
                "speed": 0.3 + 0.2 * (i % 3),
                "trail": [(int(vx), int(vy))],
                "type": "vessel",
            })

    def _update_vessels(self, step: int) -> None:
        """Move vessels along simple patrol paths."""
        cy, cx = self.patrol_center
        for i, v in enumerate(self._vessels):
            angle = 2 * math.pi * i / self.num_vessels + step * v["speed"] * 0.01
            vy = cy + self.patrol_radius * 0.6 * math.sin(angle)
            vx = cx + self.patrol_radius * 0.6 * math.cos(angle)
            vx = int(np.clip(vx, 5, self.W - 5))
            vy = int(np.clip(vy, 5, self.H - 5))
            v["xy"] = (vx, vy)
            v["heading"] = math.degrees(angle) + 90
            v["trail"].append((vx, vy))
            if len(v["trail"]) > 30:
                v["trail"] = v["trail"][-30:]

    def inject_distress(self, step: int, position: tuple[int, int]) -> None:
        """Trigger a distress event at the given position."""
        self._distress_active = True
        self._distress_pos = position
        self._distress_step = step

    def compute(
        self,
        step: int,
        *,
        engine: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._update_vessels(step)

        # POIs from engine
        pois = []
        if engine is not None:
            for t in getattr(engine, "tasks", []):
                spec = t.spec if hasattr(t, "spec") else t
                status = "pending"
                if hasattr(t, "status"):
                    status = t.status.value if hasattr(t.status, "value") else str(t.status)
                category = getattr(spec, "category", "")
                icon = IconID.ANCHOR if "patrol" in category else IconID.SHIP
                if "distress" in category:
                    icon = IconID.DISTRESS
                pois.append({
                    "xy": spec.xy,
                    "icon": icon,
                    "label": getattr(spec, "task_id", ""),
                    "status": status,
                })

        # Distress overlay
        distress_pos = None
        if self._distress_active and self._distress_pos:
            distress_pos = self._distress_pos

        # Check extra for injected distress
        if extra and "distress" in extra:
            d = extra["distress"]
            self.inject_distress(step, d["position"])
            distress_pos = d["position"]

        return {
            "pois": pois,
            "entity_positions": [dict(v) for v in self._vessels],
            "distress_position": distress_pos,
        }


# ─────────────────────────────────────────────────────────────────────────────
# M3: Critical Infrastructure — Inspection
# ─────────────────────────────────────────────────────────────────────────────

class CriticalInfraOverlay(MissionOverlay):
    """Inspection sites, restriction zones, monitoring arcs.

    Parameters
    ----------
    grid_shape : (H, W)
        Grid dimensions.
    inspection_sites : list of dict
        Each dict: {"xy": (x, y), "label": str, "radius": float}
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        inspection_sites: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(grid_shape)
        self.inspection_sites = list(inspection_sites or [])
        self._restriction_zones: list[dict] = []

    def add_restriction_zone(
        self,
        center: tuple[int, int],
        radius: float,
        step_activated: int,
    ) -> None:
        """Add a dynamic restriction zone."""
        self._restriction_zones.append({
            "center": center,
            "radius": radius,
            "step_activated": step_activated,
        })

    def compute(
        self,
        step: int,
        *,
        engine: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Build POI list
        pois = []

        # From engine
        if engine is not None:
            for t in getattr(engine, "tasks", []):
                spec = t.spec if hasattr(t, "spec") else t
                status = "pending"
                if hasattr(t, "status"):
                    status = t.status.value if hasattr(t.status, "value") else str(t.status)
                category = getattr(spec, "category", "")
                icon = IconID.INSPECTION if "inspection" in category else IconID.BUILDING
                pois.append({
                    "xy": spec.xy,
                    "icon": icon,
                    "label": getattr(spec, "task_id", ""),
                    "status": status,
                })

        # Static inspection sites
        for site in self.inspection_sites:
            pois.append({
                "xy": site["xy"],
                "icon": IconID.BUILDING,
                "label": site.get("label", ""),
                "status": "pending",
                "color": "#00CC88",
            })

        # Build dynamic NFZ from restriction zones
        nfz_mask = np.zeros((self.H, self.W), dtype=bool)
        for rz in self._restriction_zones:
            if step >= rz["step_activated"]:
                cy, cx = rz["center"]
                r = rz["radius"]
                yy, xx = np.ogrid[0:self.H, 0:self.W]
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                nfz_mask |= dist <= r

        # Check extra for injected restrictions
        if extra and "restrictions" in extra:
            for rz in extra["restrictions"]:
                self.add_restriction_zone(
                    rz["center"], rz["radius"], rz.get("step", step),
                )

        return {
            "pois": pois,
            "nfz_mask": nfz_mask if nfz_mask.any() else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_overlay(
    mission_type: str,
    grid_shape: tuple[int, int] = (500, 500),
    **kwargs,
) -> MissionOverlay:
    """Create a mission-specific overlay by type name.

    Parameters
    ----------
    mission_type : str
        One of "civil_protection", "maritime_domain", "critical_infrastructure".
    grid_shape : (H, W)
    **kwargs
        Forwarded to the overlay constructor.
    """
    registry = {
        "civil_protection": CivilProtectionOverlay,
        "maritime_domain": MaritimeDomainOverlay,
        "critical_infrastructure": CriticalInfraOverlay,
    }
    cls = registry.get(mission_type)
    if cls is None:
        raise ValueError(
            f"Unknown mission type: {mission_type!r}.  "
            f"Available: {sorted(registry)}"
        )
    return cls(grid_shape=grid_shape, **kwargs)
