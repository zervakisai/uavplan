"""D* Lite incremental planner (PL-5).

Supports incremental replanning when obstacles change.
Uses A* internally with obstacle-change detection for should_replan().
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from uavbench.blocking import compute_blocking_mask
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import PlannerBase, PlanResult


class DStarLitePlanner(PlannerBase):
    """D* Lite incremental planner.

    Tracks obstacle changes and triggers replanning when the blocking
    mask changes along the current path. Uses A* for actual path
    computation with incremental obstacle tracking (PL-5).
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)
        self._inner = AStarPlanner(heightmap, no_fly, config)
        self._cooldown = 3

        # Incremental state
        self._last_mask: np.ndarray | None = None
        self._last_mask_hash: str = ""
        self._last_replan_step = -self._cooldown
        self._last_replan_pos: tuple[int, int] | None = None
        self._dyn_state: dict[str, Any] | None = None
        self._current_path: list[tuple[int, int]] = []

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Plan using A* with cost map that includes dynamic obstacles."""
        t0 = time.perf_counter()

        # Build cost map from current dynamic state
        if self._dyn_state is not None and self._config is not None:
            mask = compute_blocking_mask(
                self._heightmap, self._no_fly,
                self._config, self._dyn_state,
            )
            # Temporarily update heightmap view for inner planner
            # by using blocked cells as high-cost
            effective_height = self._heightmap.copy()
            dynamic_blocked = mask & ~((self._heightmap > 0) | self._no_fly)
            effective_height[dynamic_blocked] = 999.0

            inner = AStarPlanner(effective_height, self._no_fly, self._config)
            result = inner.plan(start, goal, cost_map)
        else:
            result = self._inner.plan(start, goal, cost_map)

        if result.success:
            self._current_path = list(result.path)

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            path=result.path,
            success=result.success,
            compute_time_ms=elapsed,
            expansions=result.expansions,
            reason=result.reason,
        )

    def update(self, dyn_state: dict[str, Any]) -> None:
        """Accept dynamic state for incremental obstacle tracking (PL-5)."""
        self._dyn_state = dyn_state

    def _build_dynamic_mask(
        self, dyn_state: dict[str, Any],
    ) -> np.ndarray:
        """Build blocking mask from dynamic state.

        Uses compute_blocking_mask when config is available,
        otherwise merges all non-None bool masks from dyn_state.
        """
        if self._config is not None:
            return compute_blocking_mask(
                self._heightmap, self._no_fly,
                self._config, dyn_state,
            )
        # Fallback: merge all provided masks directly
        mask = (self._heightmap > 0) | self._no_fly
        for key in (
            "fire_mask", "forced_block_mask", "traffic_closure_mask",
            "traffic_occupancy_mask", "dynamic_nfz_mask",
        ):
            val = dyn_state.get(key)
            if val is not None:
                mask = mask | val.astype(bool)
        smoke = dyn_state.get("smoke_mask")
        if smoke is not None:
            mask = mask | (smoke >= 0.5)
        return mask

    def should_replan(
        self,
        current_pos: tuple[int, int],
        current_path: list[tuple[int, int]],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        """Check if obstacles changed along current path."""
        # Cooldown (RS-1)
        if step - self._last_replan_step < self._cooldown:
            return (False, "cooldown")

        # Compute current mask
        mask = self._build_dynamic_mask(dyn_state)
        mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()

        # First call: calibrate baseline hash without triggering replan
        if self._last_mask_hash == "":
            self._last_mask_hash = mask_hash
            self._last_replan_pos = current_pos
            return (False, "calibration")

        # No change in mask → no replan
        if mask_hash == self._last_mask_hash:
            return (False, "no_change")

        # Path-progress: skip if same position + same mask (RS-1)
        if (
            current_pos == self._last_replan_pos
            and mask_hash == self._last_mask_hash
        ):
            return (False, "naive_skip")

        # Check if any path cell is now blocked
        path_blocked = False
        for px, py in current_path:
            if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                if mask[py, px]:
                    path_blocked = True
                    break

        if not path_blocked:
            # Mask changed but path is still clear — record and skip
            self._last_mask_hash = mask_hash
            return (False, "path_clear")

        self._last_replan_step = step
        self._last_replan_pos = current_pos
        self._last_mask_hash = mask_hash
        return (True, "obstacle_changed")
