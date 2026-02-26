"""Interaction engine — couples fire, traffic, and NFZ dynamics.

Manages cross-layer effects: fire closes roads, traffic diverts around fire,
NFZ expansion. Produces traffic_closure_mask for blocking layer.
Fresh instance per episode (no accumulated state leak).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class InteractionEngine:
    """Couples fire, traffic, and restriction zone dynamics.

    Must be instantiated fresh per episode.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        roads_mask: np.ndarray | None = None,
        coupling_strength: float = 1.0,
    ) -> None:
        self._H, self._W = map_shape
        self._roads = (
            roads_mask.astype(bool) if roads_mask is not None
            else np.zeros((self._H, self._W), dtype=bool)
        )
        self._coupling = coupling_strength
        self._traffic_closure_mask = np.zeros(
            (self._H, self._W), dtype=bool
        )

    @property
    def traffic_closure_mask(self) -> np.ndarray:
        """bool[H, W]: road cells closed due to fire proximity."""
        return self._traffic_closure_mask.copy()

    def update(
        self,
        *,
        fire_mask: np.ndarray | None = None,
        traffic_positions: np.ndarray | None = None,
        nfz_mask: np.ndarray | None = None,
    ) -> None:
        """Update cross-layer interactions.

        Computes traffic closure mask from fire-road overlap.
        """
        self._traffic_closure_mask = np.zeros((self._H, self._W), dtype=bool)

        if fire_mask is None or not self._roads.any():
            return

        # Fire-to-roads: dilate fire mask and intersect with roads
        dilation = max(1, int(self._coupling * 2))
        dilated = self._dilate(fire_mask, dilation)
        self._traffic_closure_mask = dilated & self._roads

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Manhattan-distance dilation of a boolean mask."""
        if not mask.any():
            return np.zeros_like(mask)

        result = mask.copy()
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) + abs(dx) <= radius:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self._H and 0 <= nx < self._W:
                            result[ny, nx] = True
        return result
