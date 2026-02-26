"""Theta* any-angle planner with Bresenham LOS.

Plans any-angle paths, then expands to 4-connected grid steps
for execution (PL-6).
"""

from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np

from uavbench2.planners.base import PlannerBase, PlanResult


def _bresenham_los(
    blocked: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
) -> bool:
    """Bresenham line-of-sight check. Returns True if clear."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    H, W = blocked.shape

    cx, cy = x0, y0
    while True:
        if not (0 <= cx < W and 0 <= cy < H):
            return False
        if blocked[cy, cx]:
            return False
        if cx == x1 and cy == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy


def _expand_to_grid(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Expand any-angle waypoints to 4-connected grid steps."""
    if len(path) <= 1:
        return list(path)
    result = [path[0]]
    for i in range(1, len(path)):
        tx, ty = path[i]
        cx, cy = result[-1]
        while (cx, cy) != (tx, ty):
            dx = tx - cx
            dy = ty - cy
            if abs(dx) >= abs(dy):
                cx += 1 if dx > 0 else -1
            else:
                cy += 1 if dy > 0 else -1
            result.append((cx, cy))
    return result


class ThetaStarPlanner(PlannerBase):
    """Theta* any-angle planner with grid expansion."""

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
        t0 = time.perf_counter()
        blocked = (self._heightmap > 0) | self._no_fly
        H, W = blocked.shape
        sx, sy = start
        gx, gy = goal

        def h(x: int, y: int) -> float:
            return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5

        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (h(sx, sy), 0.0, sx, sy))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {(sx, sy): 0.0}
        expansions = 0
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while open_set and expansions < self._max_expansions:
            _, g, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                waypoints = [(gx, gy)]
                node = (gx, gy)
                while node in came_from:
                    node = came_from[node]
                    waypoints.append(node)
                waypoints.reverse()
                path = _expand_to_grid(waypoints)
                elapsed = (time.perf_counter() - t0) * 1000.0
                return PlanResult(
                    path=path, success=True,
                    compute_time_ms=elapsed, expansions=expansions,
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

                # Theta* line-of-sight: try parent→neighbor
                parent = came_from.get((cx, cy))
                if parent is not None:
                    px, py = parent
                    if _bresenham_los(blocked, px, py, nx, ny):
                        pg = g_score[parent]
                        dist = ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5
                        new_g = pg + dist
                        if new_g < g_score.get((nx, ny), float("inf")):
                            g_score[(nx, ny)] = new_g
                            came_from[(nx, ny)] = parent
                            heapq.heappush(
                                open_set, (new_g + h(nx, ny), new_g, nx, ny)
                            )
                        continue

                new_g = g + 1.0
                if new_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = new_g
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(
                        open_set, (new_g + h(nx, ny), new_g, nx, ny)
                    )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            path=[start], success=False,
            compute_time_ms=elapsed, expansions=expansions,
            reason="no_path_found",
        )
