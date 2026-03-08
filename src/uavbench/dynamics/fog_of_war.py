"""Fog of War — partial observability filter (FG-1).

Sits between env and planner in runner.py. Agent sees true dynamic state
only within sensor_radius; outside = stale (last known).

FG-1: Fog filter is planner-agnostic. Same (pos, state, radius) → same
observation. All planners receive identical fog-filtered state (FC-2).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FogOfWar:
    """Partial observability filter for dynamic state.

    Maintains a memory of last-seen state per cell. Cells within
    sensor_radius of the agent are updated to true state; cells
    outside retain their stale (last-known) values.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        sensor_radius: int = 50,
    ) -> None:
        self._H, self._W = map_shape
        self._radius = sensor_radius
        self._memory: dict[str, np.ndarray] = {}

    def observe(
        self,
        agent_xy: tuple[int, int],
        true_state: dict[str, Any],
        step_idx: int,
    ) -> dict[str, Any]:
        """Return fog-filtered dynamic state.

        Visible cells (within sensor_radius of agent) = true values.
        Other cells = stale (last-known) values from memory.
        Deterministic: same (pos, state) → same observation (FG-1).
        """
        vis_mask = self._circle_mask(agent_xy)
        filtered: dict[str, Any] = {}

        # Keys that are spatial arrays requiring fog filtering
        _SPATIAL_KEYS = (
            "fire_mask", "smoke_mask", "traffic_occupancy_mask",
            "traffic_closure_mask", "dynamic_nfz_mask",
        )

        for key, val in true_state.items():
            if val is None:
                filtered[key] = None
                continue

            if key in _SPATIAL_KEYS and isinstance(val, np.ndarray):
                if key not in self._memory:
                    self._memory[key] = np.zeros_like(val)
                # Update visible region with true state
                self._memory[key][vis_mask] = val[vis_mask]
                filtered[key] = self._memory[key].copy()
            else:
                # Non-spatial data passes through unchanged
                filtered[key] = val

        return filtered

    def _circle_mask(self, agent_xy: tuple[int, int]) -> np.ndarray:
        """Compute boolean mask of cells within sensor_radius (L2 norm)."""
        ax, ay = agent_xy
        ys, xs = np.ogrid[0:self._H, 0:self._W]
        dist_sq = (xs - ax) ** 2 + (ys - ay) ** 2
        return dist_sq <= self._radius ** 2
