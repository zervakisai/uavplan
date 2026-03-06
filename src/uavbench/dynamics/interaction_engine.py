"""Interaction engine — couples fire, traffic, and NFZ dynamics.

Manages cross-layer effects: fire closes roads, traffic diverts around fire,
NFZ expansion. Produces traffic_closure_mask for blocking layer.
Fresh instance per episode (no accumulated state leak).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


# Pre-computed cross struct for Manhattan dilation
_CROSS_STRUCT = generate_binary_structure(2, 1)


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

        if not fire_mask.any():
            return

        # Fire-to-roads: dilate fire mask and intersect with roads
        dilation = max(1, int(self._coupling * 2))
        dilated = binary_dilation(
            fire_mask, structure=_CROSS_STRUCT, iterations=dilation,
        )
        self._traffic_closure_mask = dilated & self._roads
