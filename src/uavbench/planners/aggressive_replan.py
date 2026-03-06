"""Aggressive replan planner (wraps A*).

Replans whenever the blocking mask changes, with path-progress
tracking for RS-1 compliance.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from uavbench.blocking import compute_blocking_mask
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import PlannerBase, PlanResult


class AggressiveReplanPlanner(PlannerBase):
    """Replans on any dynamic change, with RS-1 storm prevention."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)
        self._inner = AStarPlanner(heightmap, no_fly, config)
        self._cooldown = 3
        # Pre-compute static mask once (avoids recomputing heightmap > 0 every call)
        self._static_mask = (heightmap > 0) | no_fly

        # Tracking
        self._last_mask_hash: str = ""
        self._last_replan_step = -self._cooldown
        self._last_replan_pos: tuple[int, int] | None = None
        self._dyn_state: dict[str, Any] | None = None
        self._cached_mask: np.ndarray | None = None

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Plan using A* with dynamic obstacle awareness.

        Uses cached blocking mask from update() to avoid recomputation.
        """
        if self._cached_mask is not None:
            effective_height = self._heightmap.copy()
            dynamic_blocked = self._cached_mask & ~self._static_mask
            effective_height[dynamic_blocked] = 999.0
            inner = AStarPlanner(effective_height, self._no_fly, self._config)
            return inner.plan(start, goal, cost_map)
        return self._inner.plan(start, goal, cost_map)

    def update(self, dyn_state: dict[str, Any]) -> None:
        """Store dynamic state; mask computed in should_replan() and cached for plan()."""
        self._dyn_state = dyn_state
        self._cached_mask = None  # invalidate; recompute in should_replan()

    def should_replan(
        self,
        current_pos: tuple[int, int],
        current_path: list[tuple[int, int]],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        # Cooldown (RS-1)
        if step - self._last_replan_step < self._cooldown:
            return (False, "cooldown")

        # Compute and cache mask (reused by plan() if replan triggers)
        self._cached_mask = compute_blocking_mask(
            self._heightmap, self._no_fly,
            self._config, dyn_state,
        )
        mask_hash = hashlib.sha256(self._cached_mask.tobytes()).hexdigest()

        # Calibration: first call records baseline (no replan)
        if self._last_mask_hash == "":
            self._last_mask_hash = mask_hash
            self._last_replan_pos = current_pos
            return (False, "calibration")

        if mask_hash == self._last_mask_hash:
            return (False, "no_change")

        # Mask changed — check if path is actually affected
        path_blocked = False
        for px, py in current_path:
            if 0 <= py < self._cached_mask.shape[0] and 0 <= px < self._cached_mask.shape[1]:
                if self._cached_mask[py, px]:
                    path_blocked = True
                    break

        # RS-1: skip if same position AND path still clear
        if current_pos == self._last_replan_pos and not path_blocked:
            self._last_mask_hash = mask_hash
            return (False, "naive_skip")

        self._last_replan_step = step
        self._last_replan_pos = current_pos
        self._last_mask_hash = mask_hash
        return (True, "mask_changed")
