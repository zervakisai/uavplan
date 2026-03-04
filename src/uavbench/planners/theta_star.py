"""Theta* any-angle path planner.

Extends A* by allowing line-of-sight shortcuts between non-adjacent cells.
Produces shorter, smoother paths than grid-constrained A*.

The any-angle path is expanded to 4-connected grid steps for execution
(PC-1 compliance). This is a static planner — it never replans.

References:
    Nash et al., "Theta*: Any-Angle Path Planning on Grids", AAAI 2007.
"""

from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np

from uavbench.planners.base import PlannerBase, PlanResult


class ThetaStarPlanner(PlannerBase):
    """Theta* any-angle planner with grid-step expansion (PC-1)."""

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
        """Run Theta* search, then expand to grid steps (PC-1)."""
        t0 = time.perf_counter()

        blocked = (self._heightmap > 0) | self._no_fly
        H, W = blocked.shape
        sx, sy = start
        gx, gy = goal

        def h(x: int, y: int) -> float:
            return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5

        # Priority queue: (f, g, x, y)
        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (h(sx, sy), 0.0, sx, sy))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {(sx, sy): 0.0}
        expansions = 0

        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set and expansions < self._max_expansions:
            f, g, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # Reconstruct any-angle path
                waypoints = self._reconstruct(came_from, (gx, gy))
                # Expand to grid steps (PC-1)
                grid_path = self._expand_to_grid(waypoints, blocked)
                elapsed = (time.perf_counter() - t0) * 1000.0
                return PlanResult(
                    path=grid_path,
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

                # Diagonal: check both intermediate cells are free
                if dx != 0 and dy != 0:
                    if blocked[cy, cx + dx] or blocked[cy + dy, cx]:
                        continue

                # --- Theta* line-of-sight check ---
                # Try to shortcut from parent of current to neighbor
                parent = came_from.get((cx, cy), (cx, cy))
                px, py = parent

                step_cost = (dx ** 2 + dy ** 2) ** 0.5

                if _line_of_sight(px, py, nx, ny, blocked):
                    # Shortcut: connect parent directly to neighbor
                    new_g = g_score[parent] + ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5
                    if new_g < g_score.get((nx, ny), float("inf")):
                        g_score[(nx, ny)] = new_g
                        came_from[(nx, ny)] = parent
                        heapq.heappush(open_set, (new_g + h(nx, ny), new_g, nx, ny))
                else:
                    # Standard A* update
                    new_g = g + step_cost
                    if new_g < g_score.get((nx, ny), float("inf")):
                        g_score[(nx, ny)] = new_g
                        came_from[(nx, ny)] = (cx, cy)
                        heapq.heappush(open_set, (new_g + h(nx, ny), new_g, nx, ny))

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            path=[start],
            success=False,
            compute_time_ms=elapsed,
            expansions=expansions,
            reason="no_path_found",
        )

    # -- Internal --

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Reconstruct waypoint path from came_from dict."""
        path = [goal]
        node = goal
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path

    @staticmethod
    def _expand_to_grid(
        waypoints: list[tuple[int, int]],
        blocked: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Expand any-angle waypoints to 4-connected grid steps (PC-1).

        Uses greedy stepping toward the next waypoint, choosing the
        4-connected neighbor that makes most progress. When the preferred
        directions are blocked, tries all 4 directions and picks the one
        closest to the target waypoint to route around obstacles.
        """
        if len(waypoints) <= 1:
            return waypoints

        H, W = blocked.shape
        grid_path = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            bx, by = waypoints[i + 1]

            cx, cy = grid_path[-1]
            max_steps = 2 * (abs(bx - cx) + abs(by - cy)) + 20
            visited_seg: set[tuple[int, int]] = {(cx, cy)}

            for _ in range(max_steps):
                if (cx, cy) == (bx, by):
                    break

                dx = bx - cx
                dy = by - cy

                # Primary direction: larger delta
                if abs(dx) >= abs(dy):
                    primary = (cx + (1 if dx > 0 else -1), cy)
                    secondary = (cx, cy + (1 if dy > 0 else -1)) if dy != 0 else None
                else:
                    primary = (cx, cy + (1 if dy > 0 else -1))
                    secondary = (cx + (1 if dx > 0 else -1), cy) if dx != 0 else None

                moved = False
                # Try primary first
                px, py = primary
                if 0 <= px < W and 0 <= py < H and not blocked[py, px] and (px, py) not in visited_seg:
                    cx, cy = px, py
                    moved = True
                elif secondary is not None:
                    sx, sy = secondary
                    if 0 <= sx < W and 0 <= sy < H and not blocked[sy, sx] and (sx, sy) not in visited_seg:
                        cx, cy = sx, sy
                        moved = True

                # Both preferred directions blocked/visited — try all 4
                if not moved:
                    best_d = float("inf")
                    best_n = None
                    for adx, ady in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = cx + adx, cy + ady
                        if not (0 <= nx < W and 0 <= ny < H):
                            continue
                        if blocked[ny, nx] or (nx, ny) in visited_seg:
                            continue
                        d = abs(nx - bx) + abs(ny - by)
                        if d < best_d:
                            best_d = d
                            best_n = (nx, ny)
                    if best_n is None:
                        break  # Truly stuck — all neighbors blocked or visited
                    cx, cy = best_n

                visited_seg.add((cx, cy))
                if (cx, cy) != grid_path[-1]:
                    grid_path.append((cx, cy))

        return grid_path


def _line_of_sight(
    x0: int, y0: int, x1: int, y1: int,
    blocked: np.ndarray,
) -> bool:
    """Bresenham line-of-sight check.

    Returns True if no blocked cell lies on the line from (x0,y0) to (x1,y1).
    Uses the Bresenham line algorithm to enumerate intermediate cells.
    """
    H, W = blocked.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

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
