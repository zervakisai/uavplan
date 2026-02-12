from __future__ import annotations

import time
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from uavbench.planners.base import BasePlanner, PlannerConfig, PlanResult

GridPos = Tuple[int, int]  # (x, y)


@dataclass
class AStarConfig(PlannerConfig):
    """A* specific config."""
    max_expansions: int = 200_000


class AStarPlanner(BasePlanner):
    """A* grid planner on (x,y).

    Uses:
      - no_fly mask (True = blocked)
      - optional building blocking (heightmap>0 = blocked)
    """

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[AStarConfig] = None):
        super().__init__(heightmap, no_fly, config or AStarConfig())
        self.cfg: AStarConfig = self.cfg  # Type hint

    # ---------- Public API ----------

    def plan(
        self,
        start: GridPos,
        goal: GridPos,
        cost_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """Plan a path from start to goal.
        
        Returns PlanResult with path, success, compute_time_ms, and expansions count.
        """
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

        # open set: heap of (f, tie, node)
        open_heap: List[Tuple[float, int, GridPos]] = []
        tie = 0

        g_score: Dict[GridPos, float] = {start: 0.0}
        came_from: Dict[GridPos, GridPos] = {}

        f0 = self._heuristic(start, goal)
        heappush(open_heap, (f0, tie, start))

        closed: set[GridPos] = set()
        expansions = 0

        while open_heap:
            # Early timeout check
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

            for nbr in self._neighbors(current):
                if nbr in closed:
                    continue
                if self._is_blocked(nbr):
                    continue

                # Use cost_map if provided, otherwise uniform cost
                if cost_map is not None and 0 <= nbr[0] < self.W and 0 <= nbr[1] < self.H:
                    nbr_cost = float(cost_map[nbr[1], nbr[0]])
                else:
                    nbr_cost = self._move_cost(current, nbr)

                tentative_g = g_score[current] + nbr_cost

                # If new path to nbr is better, record it
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f = tentative_g + self._heuristic(nbr, goal)
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

    # ---------- Core helpers ----------

    def _move_cost(self, a: GridPos, b: GridPos) -> float:
        """Cost to move from a to b."""
        ax, ay = a
        bx, by = b
        # 4-neighborhood -> cost 1
        # diagonal -> sqrt(2) (if enabled)
        if ax != bx and ay != by:
            return 2 ** 0.5
        return 1.0

    def _reconstruct_path(self, came_from: Dict[GridPos, GridPos], current: GridPos) -> List[GridPos]:
        """Reconstruct path from start to current."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

