from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

import numpy as np

GridPos = Tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class AStarConfig:
    allow_diagonal: bool = False
    # If True, treat heightmap>0 as obstacle (ground planner).
    block_buildings: bool = True
    # Optional cap to prevent pathological cases
    max_expansions: int = 200_000


class AStarPlanner:
    """A* grid planner on (x,y).

    Uses:
      - no_fly mask (True = blocked)
      - optional building blocking (heightmap>0 = blocked)
    """

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[AStarConfig] = None):
        if heightmap.ndim != 2:
            raise ValueError("heightmap must be 2D array [H,W]")
        if no_fly.ndim != 2:
            raise ValueError("no_fly must be 2D array [H,W]")
        if heightmap.shape != no_fly.shape:
            raise ValueError("heightmap and no_fly must have the same shape")

        self.heightmap = heightmap
        self.no_fly = no_fly.astype(bool, copy=False)
        self.cfg = config or AStarConfig()

        self.H, self.W = self.heightmap.shape

    # ---------- Public API ----------

    def plan(self, start: GridPos, goal: GridPos) -> List[GridPos]:
        """Return path as list of (x,y) from start to goal (inclusive). Empty list if no path."""
        self._validate_pos(start, "start")
        self._validate_pos(goal, "goal")

        if start == goal:
            return [start]

        if self._is_blocked(start):
            return []
        if self._is_blocked(goal):
            return []

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
            _, _, current = heappop(open_heap)

            if current in closed:
                continue

            if current == goal:
                return self._reconstruct_path(came_from, current)

            closed.add(current)
            expansions += 1
            if expansions > self.cfg.max_expansions:
                # V&V: fail safe by returning "no solution" rather than hanging
                return []

            for nbr in self._neighbors(current):
                if nbr in closed:
                    continue
                if self._is_blocked(nbr):
                    continue

                tentative_g = g_score[current] + self._move_cost(current, nbr)

                # If new path to nbr is better, record it
                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f = tentative_g + self._heuristic(nbr, goal)
                    tie += 1
                    heappush(open_heap, (f, tie, nbr))

        return []  # no path

    # ---------- Core helpers ----------

    def _validate_pos(self, pos: GridPos, name: str) -> None:
        x, y = pos
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise ValueError(f"{name} out of bounds: {pos} for grid W={self.W}, H={self.H}")

    def _is_blocked(self, pos: GridPos) -> bool:
        x, y = pos
        if self.no_fly[y, x]:
            return True
        if self.cfg.block_buildings and float(self.heightmap[y, x]) > 0.0:
            return True
        return False

    def _neighbors(self, pos: GridPos) -> List[GridPos]:
        x, y = pos
        if self.cfg.allow_diagonal:
            deltas = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        else:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        out: List[GridPos] = []
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                out.append((nx, ny))
        return out

    def _move_cost(self, a: GridPos, b: GridPos) -> float:
        ax, ay = a
        bx, by = b
        # 4-neighborhood -> cost 1
        # diagonal -> sqrt(2) (if enabled)
        if ax != bx and ay != by:
            return 2 ** 0.5
        return 1.0

    def _heuristic(self, a: GridPos, b: GridPos) -> float:
        ax, ay = a
        bx, by = b
        dx = abs(ax - bx)
        dy = abs(ay - by)
        if self.cfg.allow_diagonal:
            # Octile distance (admissible for 8-neighborhood with diag cost sqrt(2))
            return (dx + dy) + (2 ** 0.5 - 2) * min(dx, dy)
        # Manhattan distance (admissible for 4-neighborhood)
        return float(dx + dy)

    def _reconstruct_path(self, came_from: Dict[GridPos, GridPos], current: GridPos) -> List[GridPos]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
