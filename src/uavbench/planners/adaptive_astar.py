"""Adaptive A* with dynamic replanning and obstacle avoidance.

Replans at adaptive intervals based on obstacle proximity, and
immediately when the current path is blocked by fire or traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .astar import AStarPlanner

GridPos = Tuple[int, int]


@dataclass(frozen=True)
class AdaptiveAStarConfig:
    base_interval: int = 10        # Steps between scheduled replans
    lookahead_steps: int = 5       # How far ahead to check for blocks
    fire_close_threshold: int = 8  # L1 dist for increased replan frequency
    traffic_buffer: int = 5        # Buffer radius around vehicles (Manhattan)
    smoke_threshold: float = 0.3   # Smoke intensity to treat as obstacle


class AdaptiveAStarPlanner:
    """A* with adaptive replanning for dynamic environments.

    Same __init__(heightmap, no_fly) interface as AStarPlanner
    for registry compatibility.  Additional methods:
    - should_replan()  — check if replanning is needed
    - replan()         — execute replanning from current position
    - get_replan_metrics() — return replanning statistics
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[AdaptiveAStarConfig] = None,
    ):
        self.heightmap = heightmap
        self.no_fly = no_fly
        self.cfg = config or AdaptiveAStarConfig()

        # State
        self._current_path: List[GridPos] = []
        self._goal: GridPos = (0, 0)
        self._steps_since_replan: int = 0
        self._replan_events: List[dict[str, Any]] = []
        self._total_replans: int = 0

    def plan(self, start: GridPos, goal: GridPos) -> List[GridPos]:
        """Initial plan (same interface as AStarPlanner)."""
        self._goal = goal
        self._steps_since_replan = 0
        self._replan_events = []
        self._total_replans = 0

        planner = AStarPlanner(self.heightmap, self.no_fly)
        self._current_path = planner.plan(start, goal)
        return list(self._current_path)

    def should_replan(
        self,
        current_pos: Tuple[int, int, int],
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """Determine if replanning is needed.

        Args:
            extra_obstacles: optional [H,W] bool mask merging any additional
                dynamic layers (moving target buffer, intruder buffer, dynamic NFZ).

        Returns (should_replan, reason).
        """
        self._steps_since_replan += 1

        # 1. Check if upcoming path is blocked
        if self._is_path_blocked(fire_mask, traffic_positions, smoke_mask,
                                 extra_obstacles):
            return True, "path_blocked"

        # 2. Check adaptive interval
        interval = self._adaptive_interval(current_pos, fire_mask, traffic_positions)
        if self._steps_since_replan >= interval:
            return True, "scheduled"

        return False, ""

    def replan(
        self,
        current_pos: Tuple[int, int, int],
        goal: GridPos,
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        reason: str = "unknown",
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
    ) -> List[GridPos]:
        """Execute replanning from current position.

        Merges all dynamic obstacles into a temp obstacle mask and plans
        around them.  ``extra_obstacles`` is a pre-merged bool mask from the
        benchmark for moving-target / intruder / dynamic-NFZ layers.
        """
        obstacles = self.no_fly.copy()

        # Merge fire into obstacles
        if fire_mask is not None:
            obstacles = obstacles | fire_mask

        # Merge heavy smoke into obstacles
        if smoke_mask is not None:
            obstacles = obstacles | (smoke_mask > self.cfg.smoke_threshold)

        # Merge traffic buffer into obstacles (Manhattan distance)
        if traffic_positions is not None and len(traffic_positions) > 0:
            buf = self.cfg.traffic_buffer
            H, W = obstacles.shape
            for vy, vx in traffic_positions:
                iy, ix = int(vy), int(vx)
                for dy in range(-buf, buf + 1):
                    for dx in range(-buf, buf + 1):
                        if abs(dy) + abs(dx) <= buf:
                            ny, nx = iy + dy, ix + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                obstacles[ny, nx] = True

        # Merge extra dynamic layers (target / intruders / NFZ)
        if extra_obstacles is not None:
            obstacles = obstacles | extra_obstacles

        start_2d: GridPos = (int(current_pos[0]), int(current_pos[1]))
        planner = AStarPlanner(self.heightmap, obstacles)
        new_path = planner.plan(start_2d, goal)

        # Log event
        self._replan_events.append({
            "step": self._steps_since_replan,
            "position": (int(current_pos[0]), int(current_pos[1]), int(current_pos[2])),
            "reason": reason,
            "new_path_length": len(new_path),
        })
        self._total_replans += 1
        self._steps_since_replan = 0

        if new_path:
            self._current_path = new_path

        return new_path

    def get_replan_metrics(self) -> dict[str, Any]:
        """Return replanning statistics."""
        return {
            "total_replans": self._total_replans,
            "replan_events": list(self._replan_events),
        }

    # ---- Internal helpers ----

    def _is_path_blocked(
        self,
        fire_mask: Optional[np.ndarray],
        traffic_positions: Optional[np.ndarray],
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
    ) -> bool:
        """Check if upcoming path segment intersects obstacles."""
        if not self._current_path:
            return False

        end_idx = min(self.cfg.lookahead_steps, len(self._current_path))
        upcoming = self._current_path[:end_idx]

        for x, y in upcoming:
            if fire_mask is not None and fire_mask[y, x]:
                return True
            if smoke_mask is not None and smoke_mask[y, x] > self.cfg.smoke_threshold:
                return True
            if traffic_positions is not None and len(traffic_positions) > 0:
                dists = np.abs(traffic_positions[:, 0] - y) + np.abs(traffic_positions[:, 1] - x)
                if dists.min() <= self.cfg.traffic_buffer:
                    return True
            if extra_obstacles is not None and extra_obstacles[y, x]:
                return True

        return False

    def _adaptive_interval(
        self,
        current_pos: Tuple[int, int, int],
        fire_mask: Optional[np.ndarray],
        traffic_positions: Optional[np.ndarray],
    ) -> int:
        """Compute replanning interval based on obstacle proximity."""
        cx, cy = int(current_pos[0]), int(current_pos[1])
        threshold = self.cfg.fire_close_threshold
        base = self.cfg.base_interval

        # Distance to nearest fire
        if fire_mask is not None:
            fire_coords = np.argwhere(fire_mask)
            if len(fire_coords) > 0:
                dists = np.abs(fire_coords[:, 0] - cy) + np.abs(fire_coords[:, 1] - cx)
                min_fire = int(dists.min())
                if min_fire < threshold:
                    return max(2, base // 4)
                if min_fire < threshold * 2:
                    return max(4, base // 2)

        # Distance to nearest vehicle
        if traffic_positions is not None and len(traffic_positions) > 0:
            dists = np.abs(traffic_positions[:, 0] - cy) + np.abs(traffic_positions[:, 1] - cx)
            if int(dists.min()) < self.cfg.traffic_buffer * 2:
                return max(2, base // 4)

        return base
