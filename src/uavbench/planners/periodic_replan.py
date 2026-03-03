"""Periodic replan planner (wraps A*).

Replans every N steps with path-progress tracking to prevent
replan storms (RS-1).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from uavbench.blocking import compute_blocking_mask
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import PlannerBase, PlanResult


class PeriodicReplanPlanner(PlannerBase):
    """Replans every replan_every_steps with path-progress tracking.

    RS-1 compliance: skips replans when at same position with same
    blocking mask (naive replan). Enforces cooldown of 3 steps.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)
        self._inner = AStarPlanner(heightmap, no_fly, config)
        self._replan_every = getattr(config, "replan_every_steps", 6)
        self._cooldown = 3

        # Path-progress tracking (RS-1)
        self._last_replan_step = -self._cooldown  # allow first replan
        self._last_replan_pos: tuple[int, int] | None = None
        self._last_mask_hash: str = ""
        self._dyn_state: dict[str, Any] | None = None

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Plan using A* with dynamic obstacle awareness.

        When dynamic state is available (after update()), builds a
        blocking mask that includes fire, smoke, traffic, etc. and
        routes around them.
        """
        if self._dyn_state is not None and self._config is not None:
            mask = compute_blocking_mask(
                self._heightmap, self._no_fly,
                self._config, self._dyn_state,
            )
            effective_height = self._heightmap.copy()
            dynamic_blocked = mask & ~((self._heightmap > 0) | self._no_fly)
            effective_height[dynamic_blocked] = 999.0
            inner = AStarPlanner(effective_height, self._no_fly, self._config)
            return inner.plan(start, goal, cost_map)
        return self._inner.plan(start, goal, cost_map)

    def update(self, dyn_state: dict[str, Any]) -> None:
        self._dyn_state = dyn_state

    def should_replan(
        self,
        current_pos: tuple[int, int],
        current_path: list[tuple[int, int]],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        # Cooldown check (RS-1)
        if step - self._last_replan_step < self._cooldown:
            return (False, "cooldown")

        # Periodic trigger
        if step % self._replan_every != 0:
            return (False, "not_periodic")

        # Path-progress tracking: skip naive replans (RS-1)
        mask = compute_blocking_mask(
            self._heightmap, self._no_fly,
            self._config, dyn_state,
        )
        mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()

        if (
            current_pos == self._last_replan_pos
            and mask_hash == self._last_mask_hash
        ):
            return (False, "naive_skip")

        self._last_replan_step = step
        self._last_replan_pos = current_pos
        self._last_mask_hash = mask_hash
        return (True, "periodic")
