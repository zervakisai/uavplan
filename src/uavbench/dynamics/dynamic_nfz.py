"""Dynamic no-fly zones that expand and contract — forces replanning.

.. deprecated:: 1.0.0
    This module is superseded by :class:`uavbench.dynamics.restriction_zones.MissionRestrictionModel`,
    which provides mission-grounded restriction zones with structured lifecycle events.
    ``DynamicNFZModel`` is retained only for backward compatibility with external scripts
    that may reference it directly.  It is NOT imported by any production module in
    UAVBench and will be removed in a future release.

Zones are placed along the UAV's straight-line path so the initial
static plan becomes invalid within the first ~30-50 steps.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

warnings.warn(
    "DynamicNFZModel is deprecated; use MissionRestrictionModel from "
    "uavbench.dynamics.restriction_zones instead.",
    DeprecationWarning,
    stacklevel=2,
)

GridPos = Tuple[int, int]


class DynamicNFZModel:
    """Expanding / contracting circular no-fly zones.

    Args:
        map_shape: (H, W)
        uav_start: (x, y) — used to place zones on the likely path
        uav_goal: (x, y)
        num_zones: how many independent zones
        expansion_rate: pixels / step growth
        max_radius: cap
        rng: deterministic RNG
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        uav_start: GridPos,
        uav_goal: GridPos,
        num_zones: int = 3,
        expansion_rate: float = 0.8,
        max_radius: int = 35,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.height, self.width = map_shape
        self.num_zones = num_zones
        self.expansion_rate = expansion_rate
        self.max_radius = max_radius
        self._rng = rng if rng is not None else np.random.default_rng()

        # Place centres along start → goal line
        self.centres = self._place_along_path(uav_start, uav_goal)
        self.radii = np.full(num_zones, 3.0, dtype=np.float32)
        self.directions = np.ones(num_zones, dtype=np.int8)   # +1 expand, -1 contract
        # Stagger activation so zones appear over time
        self.activation_step = self._rng.integers(10, 40, size=num_zones)
        self.step_count = 0

    # ------------------------------------------------------------------

    def _place_along_path(self, start: GridPos, goal: GridPos) -> list[GridPos]:
        """Place zone centres at even fractions along start→goal, with jitter."""
        centres: list[GridPos] = []
        for i in range(1, self.num_zones + 1):
            frac = i / (self.num_zones + 1)
            cx = int(start[0] + (goal[0] - start[0]) * frac)
            cy = int(start[1] + (goal[1] - start[1]) * frac)
            # Small jitter
            cx += int(self._rng.integers(-15, 16))
            cy += int(self._rng.integers(-15, 16))
            cx = int(np.clip(cx, 30, self.width - 30))
            cy = int(np.clip(cy, 30, self.height - 30))
            centres.append((cx, cy))
        return centres

    # ------------------------------------------------------------------

    def step(self, dt: float = 1.0) -> None:
        """Expand / contract active zones."""
        self.step_count += 1

        for i in range(self.num_zones):
            if self.step_count < self.activation_step[i]:
                continue
            self.radii[i] += self.directions[i] * self.expansion_rate * dt
            if self.radii[i] >= self.max_radius:
                self.radii[i] = float(self.max_radius)
                self.directions[i] = -1
            elif self.radii[i] <= 5.0:
                self.radii[i] = 5.0
                self.directions[i] = 1

    def get_nfz_mask(self) -> np.ndarray:
        """[H, W] bool — current dynamic NFZ footprint."""
        mask = np.zeros((self.height, self.width), dtype=bool)

        for i in range(self.num_zones):
            if self.step_count < self.activation_step[i]:
                continue
            cx, cy = self.centres[i]
            r = int(self.radii[i])
            # Efficient circle via meshgrid
            ys = np.arange(max(0, cy - r), min(self.height, cy + r + 1))
            xs = np.arange(max(0, cx - r), min(self.width, cx + r + 1))
            if len(ys) == 0 or len(xs) == 0:
                continue
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
            mask[yy[dist_sq <= r * r], xx[dist_sq <= r * r]] = True

        return mask

    @property
    def active_zones(self) -> int:
        return int((self.step_count >= self.activation_step).sum())
