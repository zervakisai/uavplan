"""Aggressive replan planner (wraps A*).

Replans whenever the blocking mask changes, with path-progress
tracking for RS-1 compliance.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from uavbench2.blocking import compute_blocking_mask
from uavbench2.planners.astar import AStarPlanner
from uavbench2.planners.base import PlannerBase, PlanResult


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

        # Tracking
        self._last_mask_hash: str = ""
        self._last_replan_step = -self._cooldown
        self._last_replan_pos: tuple[int, int] | None = None

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        return self._inner.plan(start, goal, cost_map)

    def update(self, dyn_state: dict[str, Any]) -> None:
        pass

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

        # Check if mask changed
        mask = compute_blocking_mask(
            self._heightmap, self._no_fly,
            self._config, dyn_state,
        )
        mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()

        if mask_hash == self._last_mask_hash:
            return (False, "no_change")

        # Path-progress: skip if same position + same mask
        if (
            current_pos == self._last_replan_pos
            and mask_hash == self._last_mask_hash
        ):
            return (False, "naive_skip")

        self._last_replan_step = step
        self._last_replan_pos = current_pos
        self._last_mask_hash = mask_hash
        return (True, "mask_changed")
