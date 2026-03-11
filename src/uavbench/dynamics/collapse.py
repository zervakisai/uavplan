"""Structural collapse model — fire-triggered building debris.

Buildings that burn for collapse_delay steps collapse, spawning permanent
debris in adjacent cells. Debris is a 3rd dynamics layer alongside fire
and traffic, blocking movement permanently.

Deterministic: all randomness via caller-supplied rng (DC-1).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

# Manhattan radius for debris scatter around collapsed building
_DEBRIS_RADIUS = 2

# Dilation structure for Manhattan distance-1 (cross)
_CROSS = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]], dtype=bool,
)


class CollapseModel:
    """Fire-triggered structural collapse with debris scatter.

    Buildings (heightmap > 0) that are exposed to fire for ≥ collapse_delay
    steps collapse. Collapsed buildings spawn debris in cells within
    Manhattan distance ≤ _DEBRIS_RADIUS, with probability debris_prob
    per candidate cell. Debris is PERMANENT and blocks movement.

    All randomness flows through rng (DC-1).
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        rng: np.random.Generator,
        collapse_delay: int = 80,
        debris_prob: float = 0.6,
    ) -> None:
        self._rng = rng
        self._H, self._W = heightmap.shape

        # Static building mask (> 0 = building)
        self._buildings = heightmap > 0

        self._collapse_delay = collapse_delay
        self._debris_prob = debris_prob

        # Per-cell fire exposure counter (only for building cells)
        self._fire_exposure = np.zeros((self._H, self._W), dtype=np.int32)

        # Collapsed building mask (buildings that have structurally failed)
        self._collapsed = np.zeros((self._H, self._W), dtype=bool)

        # Debris mask — permanent, accumulates over episode
        self._debris = np.zeros((self._H, self._W), dtype=bool)

        # Events log
        self._events: list[dict] = []

    @property
    def debris_mask(self) -> np.ndarray:
        """bool[H, W]: True where debris blocks movement (permanent)."""
        return self._debris.copy()

    @property
    def collapsed_mask(self) -> np.ndarray:
        """bool[H, W]: True where buildings have collapsed."""
        return self._collapsed.copy()

    def pop_events(self) -> list[dict]:
        """Return and clear pending collapse events."""
        events = self._events
        self._events = []
        return events

    def step(self, fire_mask: np.ndarray | None, step_idx: int = 0) -> None:
        """Advance collapse by one timestep.

        1. Increment fire_exposure for building cells currently on fire.
        2. Detect buildings exceeding collapse_delay → mark collapsed.
        3. Scatter debris around newly collapsed buildings.
        """
        if fire_mask is None:
            return

        # 1. Increment exposure for buildings under fire
        burning_buildings = self._buildings & fire_mask & ~self._collapsed
        self._fire_exposure[burning_buildings] += 1

        # 2. Detect newly collapsing buildings
        newly_collapsed = (
            burning_buildings
            & (self._fire_exposure >= self._collapse_delay)
            & ~self._collapsed
        )

        if not newly_collapsed.any():
            return

        self._collapsed |= newly_collapsed

        # 3. Scatter debris around each newly collapsed cell
        # Dilate newly_collapsed by Manhattan radius to get candidate area
        debris_zone = newly_collapsed.copy()
        for _ in range(_DEBRIS_RADIUS):
            debris_zone = binary_dilation(debris_zone, structure=_CROSS)

        # Candidate cells: in debris zone, not already debris, not buildings
        candidates = debris_zone & ~self._debris & ~self._buildings
        if not candidates.any():
            return

        # Roll for debris probability (DC-1: uses rng)
        cand_ys, cand_xs = np.where(candidates)
        rolls = self._rng.random(len(cand_ys))
        debris_hits = rolls < self._debris_prob
        new_debris_ys = cand_ys[debris_hits]
        new_debris_xs = cand_xs[debris_hits]

        if len(new_debris_ys) > 0:
            self._debris[new_debris_ys, new_debris_xs] = True
            # Log collapse events
            col_ys, col_xs = np.where(newly_collapsed)
            for y, x in zip(col_ys, col_xs):
                self._events.append({
                    "type": "building_collapse",
                    "x": int(x),
                    "y": int(y),
                    "step": step_idx,
                    "debris_cells": int(debris_hits.sum()),
                })
