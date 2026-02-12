"""Theta* planner: any-angle pathfinding for smoother paths.

Theta* allows straight-line paths between non-adjacent nodes, producing
smoother trajectories than grid-aligned 4/8-connected planners.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from uavbench.planners.base import BasePlanner, PlannerConfig, PlanResult

GridPos = Tuple[int, int]  # (x, y)


@dataclass
class ThetaStarConfig(PlannerConfig):
    """Theta* specific config."""
    max_expansions: int = 200_000


class ThetaStarPlanner(BasePlanner):
    """Theta* planner: any-angle pathfinding.
    
    Allows diagonal movement with line-of-sight shortcuts for smoother paths.
    Uses the same graph search as A* but with different cost calculations.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[ThetaStarConfig] = None,
    ):
        super().__init__(heightmap, no_fly, config or ThetaStarConfig())
        self.cfg: ThetaStarConfig = self.cfg

    def plan(
        self,
        start: GridPos,
        goal: GridPos,
        cost_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """Plan a path using Theta* (any-angle)."""
        start_time = time.monotonic()

        self._validate_pos(start, "start")
        self._validate_pos(goal, "goal")

        if start == goal:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return PlanResult(
                path=[start],
                success=True,
                compute_time_ms=elapsed_ms,
                expansions=0,
                reason="start == goal"
            )

        if self._is_blocked(start):
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return PlanResult(
                path=[],
                success=False,
                compute_time_ms=elapsed_ms,
                expansions=0,
                reason="start position blocked"
            )
        if self._is_blocked(goal):
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return PlanResult(
                path=[],
                success=False,
                compute_time_ms=elapsed_ms,
                expansions=0,
                reason="goal position blocked"
            )

        # Main Theta* search
        open_heap: List[Tuple[float, int, GridPos]] = []
        tie = 0

        g_score: Dict[GridPos, float] = {start: 0.0}
        came_from: Dict[GridPos, GridPos] = {}

        f0 = self._heuristic(start, goal)
        heappush(open_heap, (f0, tie, start))

        closed: set[GridPos] = set()
        expansions = 0

        while open_heap:
            # Timeout check
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if elapsed_ms > self.cfg.max_planning_time_ms:
                return PlanResult(
                    path=[],
                    success=False,
                    compute_time_ms=elapsed_ms,
                    expansions=expansions,
                    reason=f"timeout ({self.cfg.max_planning_time_ms}ms exceeded)"
                )

            _, _, current = heappop(open_heap)

            if current in closed:
                continue

            if current == goal:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                path = self._reconstruct_path(came_from, current)
                return PlanResult(
                    path=path,
                    success=True,
                    compute_time_ms=elapsed_ms,
                    expansions=expansions,
                    reason="goal found"
                )

            closed.add(current)
            expansions += 1
            if expansions > self.cfg.max_expansions:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                return PlanResult(
                    path=[],
                    success=False,
                    compute_time_ms=elapsed_ms,
                    expansions=expansions,
                    reason=f"expansion limit ({self.cfg.max_expansions}) exceeded"
                )

            # Get neighbors (8-connected for Theta*)
            x, y = current
            candidates = [
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y + 1),
            ]

            for nbr in candidates:
                if nbr in closed:
                    continue
                if self._is_blocked(nbr):
                    continue

                # Theta* key insight: try line-of-sight to parent of current
                parent = came_from.get(current)
                if parent is not None and self._line_of_sight(parent, nbr):
                    # Direct path from parent to neighbor
                    tentative_g = g_score[parent] + self._euclidean_dist(parent, nbr)
                    if tentative_g < g_score.get(nbr, float("inf")):
                        came_from[nbr] = parent
                        g_score[nbr] = tentative_g
                else:
                    # Regular grid move
                    tentative_g = g_score[current] + self._euclidean_dist(current, nbr)
                    if tentative_g < g_score.get(nbr, float("inf")):
                        came_from[nbr] = current
                        g_score[nbr] = tentative_g

                # Re-open if better path found
                f = g_score[nbr] + self._heuristic(nbr, goal)
                tie += 1
                heappush(open_heap, (f, tie, nbr))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        return PlanResult(
            path=[],
            success=False,
            compute_time_ms=elapsed_ms,
            expansions=expansions,
            reason="no path found"
        )

    # ========== Theta* helpers ==========

    def _euclidean_dist(self, a: GridPos, b: GridPos) -> float:
        """Euclidean distance between two grid points."""
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        return float((dx**2 + dy**2) ** 0.5)

    def _line_of_sight(self, a: GridPos, b: GridPos) -> bool:
        """Check if there's a line-of-sight (straight line) between a and b.
        
        Uses Bresenham's line algorithm to check all grid cells on the line.
        """
        ax, ay = a
        bx, by = b

        dx = abs(bx - ax)
        dy = abs(by - ay)
        sx = 1 if bx > ax else -1
        sy = 1 if by > ay else -1

        if dx > dy:
            # X-major line
            err = dx / 2
            y = ay
            for x in range(ax, bx + sx, sx):
                if self._is_blocked((x, y)):
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    if self._is_blocked((x, y)):
                        return False
                    err += dx
        else:
            # Y-major line
            err = dy / 2
            x = ax
            for y in range(ay, by + sy, sy):
                if self._is_blocked((x, y)):
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    if self._is_blocked((x, y)):
                        return False
                    err += dy

        return True

    def _reconstruct_path(
        self,
        came_from: Dict[GridPos, GridPos],
        current: GridPos,
    ) -> List[GridPos]:
        """Reconstruct path from start to current (may include non-grid-aligned points)."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
