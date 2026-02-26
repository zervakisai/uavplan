"""A* planner (4-connected grid).

Minimal Phase 2 implementation. Finds optimal shortest path on a grid
with building obstacles.
"""

from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np

from uavbench2.planners.base import PlannerBase, PlanResult


class AStarPlanner(PlannerBase):
    """A* path planner on 4-connected grid."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)
        self._max_expansions = 200_000

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Run A* search from start to goal.

        Returns PlanResult with path as list[(x,y)] (PL-6).
        """
        t0 = time.perf_counter()

        blocked = (self._heightmap > 0) | self._no_fly
        H, W = blocked.shape

        sx, sy = start
        gx, gy = goal

        # Heuristic: Manhattan distance
        def h(x: int, y: int) -> int:
            return abs(x - gx) + abs(y - gy)

        # Priority queue: (f, g, x, y)
        open_set: list[tuple[int, int, int, int]] = []
        heapq.heappush(open_set, (h(sx, sy), 0, sx, sy))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], int] = {(sx, sy): 0}
        expansions = 0

        # 4-connected neighbors: dx, dy
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while open_set and expansions < self._max_expansions:
            _, g, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # Reconstruct path
                path = [(gx, gy)]
                node = (gx, gy)
                while node in came_from:
                    node = came_from[node]
                    path.append(node)
                path.reverse()

                elapsed = (time.perf_counter() - t0) * 1000.0
                return PlanResult(
                    path=path,
                    success=True,
                    compute_time_ms=elapsed,
                    expansions=expansions,
                )

            if g > g_score.get((cx, cy), float("inf")):
                continue

            expansions += 1

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if blocked[ny, nx]:
                    continue

                new_g = g + 1
                if new_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = new_g
                    came_from[(nx, ny)] = (cx, cy)
                    f = new_g + h(nx, ny)
                    heapq.heappush(open_set, (f, new_g, nx, ny))

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            path=[start],
            success=False,
            compute_time_ms=elapsed,
            expansions=expansions,
            reason="no_path_found",
        )
