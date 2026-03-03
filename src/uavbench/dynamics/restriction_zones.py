"""Mission-aware restriction zones (dynamic NFZ).

Zones activate on a staggered schedule between event_t1 and midpoint(t1, t2).
Coverage is capped at max_coverage fraction of the map.
All randomness flows through caller-supplied rng (DC-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RestrictionZone:
    """A single dynamic no-fly zone."""

    zone_id: str
    zone_type: str
    center: tuple[int, int]  # (x, y)
    radius: int
    activation_step: int
    active: bool = False
    severity: float = 1.0
    label: str = ""
    expires_step: int | None = None
    _mask: np.ndarray | None = field(default=None, repr=False)


class RestrictionZoneModel:
    """Manages dynamic no-fly restriction zones.

    Zones activate on schedule, can expand, and are subject to a
    max_coverage cap. Provides NFZ mask for blocking layer.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
        num_zones: int = 3,
        max_coverage: float = 0.30,
        event_t1: int = 30,
        event_t2: int = 80,
    ) -> None:
        self._rng = rng
        self._H, self._W = map_shape
        self._max_coverage = max_coverage
        self._step_count = 0
        self._zones: list[RestrictionZone] = []
        self._nfz_mask = np.zeros((self._H, self._W), dtype=bool)

        # Create zones with staggered activation
        mid = (event_t1 + event_t2) // 2
        for i in range(num_zones):
            # Activation evenly spaced between t1 and midpoint
            if num_zones > 1:
                act_step = event_t1 + i * (mid - event_t1) // (num_zones - 1)
            else:
                act_step = event_t1

            # Random center
            cx = int(rng.integers(self._W))
            cy = int(rng.integers(self._H))
            radius = int(rng.integers(8, 20))

            self._zones.append(
                RestrictionZone(
                    zone_id=f"NFZ-{i}",
                    zone_type="restriction",
                    center=(cx, cy),
                    radius=radius,
                    activation_step=act_step,
                    label=f"Restriction Zone {i}",
                )
            )

    # -- Public properties --

    @property
    def active_zones(self) -> int:
        return sum(1 for z in self._zones if z.active)

    def get_nfz_mask(self) -> np.ndarray:
        """bool[H, W]: union of all active zone masks."""
        return self._nfz_mask.copy()

    def get_zones(self) -> list[RestrictionZone]:
        return list(self._zones)

    @property
    def peak_coverage(self) -> float:
        return float(self._nfz_mask.sum()) / (self._H * self._W)

    # -- Step --

    def step(self, dt: float = 1.0, fire_mask: np.ndarray | None = None) -> None:
        """Advance restriction zones by one step.

        Activates zones on schedule, deactivates expired ones,
        rebuilds combined mask, enforces coverage cap.
        """
        self._step_count += 1

        for zone in self._zones:
            # Activate on schedule
            if not zone.active and self._step_count >= zone.activation_step:
                zone.active = True
                zone._mask = self._rasterize_circle(zone.center, zone.radius)

            # Deactivate expired
            if zone.active and zone.expires_step is not None:
                if self._step_count >= zone.expires_step:
                    zone.active = False
                    zone._mask = None

        self._rebuild_mask()

    def relax_zones(self, shrink_px: int = 2) -> int:
        """Erode active zone masks. Returns cells freed. For guardrail D2."""
        cells_freed = 0
        for zone in self._zones:
            if not zone.active or zone._mask is None:
                continue
            before = int(zone._mask.sum())
            # Simple erosion: shrink radius
            zone.radius = max(0, zone.radius - shrink_px)
            if zone.radius == 0:
                zone.active = False
                zone._mask = None
                cells_freed += before
            else:
                zone._mask = self._rasterize_circle(zone.center, zone.radius)
                cells_freed += before - int(zone._mask.sum())

        self._rebuild_mask()
        return cells_freed

    # -- Internal --

    def _rasterize_circle(
        self, center: tuple[int, int], radius: int
    ) -> np.ndarray:
        """Create bool mask for a Manhattan-distance circle."""
        mask = np.zeros((self._H, self._W), dtype=bool)
        cx, cy = center
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) <= radius:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self._H and 0 <= nx < self._W:
                        mask[ny, nx] = True
        return mask

    def _rebuild_mask(self) -> None:
        """Rebuild combined NFZ mask, enforcing coverage cap.

        Zones that exceed the cap are excluded from the mask this step
        but NOT permanently deactivated — they can be included in future
        steps if earlier zones expire or are relaxed.
        """
        self._nfz_mask = np.zeros((self._H, self._W), dtype=bool)
        total_cells = self._H * self._W

        # Add zones from oldest to newest
        for zone in self._zones:
            if zone.active and zone._mask is not None:
                candidate = self._nfz_mask | zone._mask
                coverage = candidate.sum() / total_cells
                if coverage <= self._max_coverage:
                    self._nfz_mask = candidate
                # else: skip this zone this step (cap exceeded) but keep active
