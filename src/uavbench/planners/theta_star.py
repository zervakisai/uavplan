"""Theta* any-angle planner with Bresenham LOS.

Plans any-angle paths, then expands to 4-connected grid steps
for execution (PL-6).
"""

from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np

from uavbench.planners.base import PlannerBase, PlanResult


def _bresenham_los(
    blocked: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
) -> bool:
    """Strict Bresenham LOS check for 4-connected grid expansion.

    Returns True if the line is clear AND all diagonal steps have
    at least one unblocked 4-connected intermediate.  This ensures
    the path can be expanded to 4-connected grid steps without
    traversing blocked cells.
    """
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
        move_x = e2 > -dy
        move_y = e2 < dx

        if move_x and move_y:
            # Diagonal step — at least one 4-connected intermediate
            # must be unblocked for grid expansion to work.
            mid_a_x, mid_a_y = cx + sx, cy   # x-first
            mid_b_x, mid_b_y = cx, cy + sy   # y-first
            a_ok = (
                0 <= mid_a_x < W and 0 <= mid_a_y < H
                and not blocked[mid_a_y, mid_a_x]
            )
            b_ok = (
                0 <= mid_b_x < W and 0 <= mid_b_y < H
                and not blocked[mid_b_y, mid_b_x]
            )
            if not a_ok and not b_ok:
                return False

        if move_x:
            err -= dy
            cx += sx
        if move_y:
            err += dx
            cy += sy


def _expand_to_grid(
    path: list[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]]:
    """Expand any-angle waypoints to 4-connected grid steps.

    Uses Bresenham tracing (same cells verified by LOS check) and
    converts diagonal moves to 4-connected by inserting validated
    intermediate cells. Falls back to staircase with blocked-cell
    avoidance if needed.
    """
    if len(path) <= 1:
        return list(path)
    result = [path[0]]
    H, W = blocked.shape
    for i in range(1, len(path)):
        segment = _bresenham_to_4connected(
            result[-1], path[i], blocked, H, W,
        )
        result.extend(segment)
    return result


def _bresenham_to_4connected(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace Bresenham line from start to end, converting diagonal
    moves to validated 4-connected steps."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    cx, cy = x0, y0
    cells: list[tuple[int, int]] = []

    while (cx, cy) != (x1, y1):
        prev_x, prev_y = cx, cy
        e2 = 2 * err
        moved_x = False
        moved_y = False
        if e2 > -dy:
            err -= dy
            cx += sx
            moved_x = True
        if e2 < dx:
            err += dx
            cy += sy
            moved_y = True

        if moved_x and moved_y:
            # Diagonal — insert a 4-connected intermediate
            mid_a = (prev_x + sx, prev_y)  # x-first
            mid_b = (prev_x, prev_y + sy)  # y-first
            ax, ay = mid_a
            bx, by = mid_b
            if 0 <= ay < H and 0 <= ax < W and not blocked[ay, ax]:
                cells.append(mid_a)
            elif 0 <= by < H and 0 <= bx < W and not blocked[by, bx]:
                cells.append(mid_b)
            # else: both intermediates blocked — skip (shouldn't happen
            # with valid LOS, destination cell is still added below)

        cells.append((cx, cy))

    return cells


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
                path = _expand_to_grid(waypoints, blocked)
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
