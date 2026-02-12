"""Jump Point Search (JPS): accelerated grid pathfinding.

JPS is a speed optimization for uniform-cost grid planners like A*.
It "jumps" over symmetric paths, exploring fewer nodes.
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
class JPSConfig(PlannerConfig):
    """JPS specific config."""
    max_expansions: int = 200_000


class JPSPlanner(BasePlanner):
    """Jump Point Search planner: fast grid pathfinding with forced neighbor pruning.
    
    JPS speeds up A* by jumping over symmetric nodes in uniform-cost grids.
    On many grids, JPS is 10-40x faster than A*.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[JPSConfig] = None,
    ):
        super().__init__(heightmap, no_fly, config or JPSConfig())
        self.cfg: JPSConfig = self.cfg

    def plan(
        self,
        start: GridPos,
        goal: GridPos,
        cost_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """Plan a path using JPS."""
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

        # JPS search (A* with forced neighbor pruning)
        open_heap: List[Tuple[float, int, GridPos]] = []
        tie = 0

        g_score: Dict[GridPos, float] = {start: 0.0}
        came_from: Dict[GridPos, Tuple[GridPos, int, int]] = {}  # pos -> (parent, dx, dy)

        f0 = self._heuristic(start, goal)
        heappush(open_heap, (f0, tie, start))

        closed: set[GridPos] = set()
        expansions = 0
        jump_count = 0

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
                    reason=f"goal found (jumps: {jump_count})"
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

            # Identify forced neighbors and natural successors
            if current in came_from:
                _, parent_dx, parent_dy = came_from[current]
            else:
                parent_dx, parent_dy = 0, 0

            # JPS pruning: only explore neighbors that are not dominated by symmetry
            successors = self._get_successors(current, parent_dx, parent_dy, goal, closed)

            for nbr, nbr_cost in successors:
                if nbr in closed:
                    continue

                tentative_g = g_score[current] + nbr_cost

                if tentative_g < g_score.get(nbr, float("inf")):
                    dx = 1 if nbr[0] > current[0] else (-1 if nbr[0] < current[0] else 0)
                    dy = 1 if nbr[1] > current[1] else (-1 if nbr[1] < current[1] else 0)
                    came_from[nbr] = (current, dx, dy)
                    g_score[nbr] = tentative_g
                    f = tentative_g + self._heuristic(nbr, goal)
                    tie += 1
                    heappush(open_heap, (f, tie, nbr))
                    jump_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000
        return PlanResult(
            path=[],
            success=False,
            compute_time_ms=elapsed_ms,
            expansions=expansions,
            reason="no path found"
        )

    # ========== JPS helpers ==========

    def _get_successors(
        self,
        pos: GridPos,
        parent_dx: int,
        parent_dy: int,
        goal: GridPos,
        closed: set[GridPos],
    ) -> List[Tuple[GridPos, float]]:
        """Get valid successors using JPS pruning.
        
        Returns list of (neighbor_pos, cost_to_neighbor) tuples.
        """
        x, y = pos
        successors = []

        # If parent_dx/dy are 0, this is the start node: explore all neighbors
        if parent_dx == 0 and parent_dy == 0:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                jmp = self._jump(pos, dx, dy, goal, closed)
                if jmp is not None:
                    cost = self._euclidean_dist(pos, jmp)
                    successors.append((jmp, cost))
        else:
            # Horizontal/vertical movement
            if parent_dx != 0:
                jmp = self._jump(pos, parent_dx, 0, goal, closed)
                if jmp is not None:
                    cost = self._euclidean_dist(pos, jmp)
                    successors.append((jmp, cost))
            
            if parent_dy != 0:
                jmp = self._jump(pos, 0, parent_dy, goal, closed)
                if jmp is not None:
                    cost = self._euclidean_dist(pos, jmp)
                    successors.append((jmp, cost))
            
            # Diagonal: add forced neighbors
            if parent_dx != 0 and parent_dy != 0:
                # Try horizontal and vertical steps
                h_step = (x + parent_dx, y)
                if not self._is_blocked(h_step):
                    jmp_h = self._jump(pos, parent_dx, 0, goal, closed)
                    if jmp_h is not None:
                        cost = self._euclidean_dist(pos, jmp_h)
                        successors.append((jmp_h, cost))
                
                v_step = (x, y + parent_dy)
                if not self._is_blocked(v_step):
                    jmp_v = self._jump(pos, 0, parent_dy, goal, closed)
                    if jmp_v is not None:
                        cost = self._euclidean_dist(pos, jmp_v)
                        successors.append((jmp_v, cost))
                
                # Diagonal step
                jmp_d = self._jump(pos, parent_dx, parent_dy, goal, closed)
                if jmp_d is not None:
                    cost = self._euclidean_dist(pos, jmp_d)
                    successors.append((jmp_d, cost))

        return successors

    def _jump(
        self,
        pos: GridPos,
        dx: int,
        dy: int,
        goal: GridPos,
        closed: set[GridPos],
    ) -> Optional[GridPos]:
        """Jump in direction (dx, dy) until hitting obstacle or forced neighbor.
        
        Returns the furthest reachable position or None if no jump.
        """
        x, y = pos
        steps = 0

        while True:
            x += dx
            y += dy
            steps += 1
            new_pos = (x, y)

            # Check bounds and obstacles
            if self._is_blocked(new_pos):
                return None

            # Reached goal
            if new_pos == goal:
                return new_pos

            # Check for forced neighbors (indicates a jump point)
            if dx != 0 and dy != 0:
                # Diagonal: check for forced neighbors in cardinal directions
                if (self._is_blocked((x - dx, y)) and not self._is_blocked((x - dx, y + dy))) or \
                   (self._is_blocked((x, y - dy)) and not self._is_blocked((x + dx, y - dy))):
                    return new_pos
            elif dx != 0:
                # Horizontal: check vertical neighbors
                if (self._is_blocked((x, y - 1)) and not self._is_blocked((x + dx, y - 1))) or \
                   (self._is_blocked((x, y + 1)) and not self._is_blocked((x + dx, y + 1))):
                    return new_pos
            else:
                # Vertical: check horizontal neighbors
                if (self._is_blocked((x - 1, y)) and not self._is_blocked((x - 1, y + dy))) or \
                   (self._is_blocked((x + 1, y)) and not self._is_blocked((x + 1, y + dy))):
                    return new_pos

            # Limit jump distance to prevent runaway
            if steps > 200:  # Heuristic limit
                return new_pos

    def _euclidean_dist(self, a: GridPos, b: GridPos) -> float:
        """Euclidean distance."""
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        return float((dx**2 + dy**2) ** 0.5)

    def _reconstruct_path(
        self,
        came_from: Dict[GridPos, Tuple[GridPos, int, int]],
        current: GridPos,
    ) -> List[GridPos]:
        """Reconstruct path from start to current."""
        path = [current]
        while current in came_from:
            parent, dx, dy = came_from[current]
            # Interpolate jumps back to grid cells for path
            current = parent
            path.append(current)
        path.reverse()
        return path
