"""Moving target model for SAR scenarios — forces replanning.

A target (e.g. ambulance) moves along roads toward a destination,
creating a moving buffer zone that blocks the UAV's initial path.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np

GridPos = Tuple[int, int]


class MovingTargetModel:
    """Road-following moving target that creates dynamic obstacles.

    Args:
        roads_mask: bool [H, W] — road pixel mask
        start_pos: (x, y) target spawn position
        goal_pos: (x, y) target destination
        speed: cells per step
        buffer_radius: Manhattan radius of blocking zone around target
        rng: deterministic RNG
    """

    def __init__(
        self,
        roads_mask: np.ndarray,
        start_pos: GridPos,
        goal_pos: GridPos,
        speed: float = 1.0,
        buffer_radius: int = 8,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._roads = roads_mask.astype(bool)
        self.height, self.width = self._roads.shape
        self.speed = speed
        self.buffer_radius = buffer_radius
        self._rng = rng if rng is not None else np.random.default_rng()

        self.position = np.array([float(start_pos[0]), float(start_pos[1])],
                                 dtype=np.float32)
        self.reached_goal = False
        self.step_count = 0

        # BFS path on roads from start to goal
        self._road_path = self._bfs_road_path(start_pos, goal_pos)
        self._path_idx = 0

    # ------------------------------------------------------------------

    def _bfs_road_path(self, start: GridPos, goal: GridPos) -> list[GridPos]:
        """BFS shortest path on road cells."""
        sx, sy = start
        gx, gy = goal

        # If start or goal not on road, find nearest road cell
        if not self._roads[sy, sx]:
            sx, sy = self._nearest_road(sx, sy)
        if not self._roads[gy, gx]:
            gx, gy = self._nearest_road(gx, gy)

        queue: deque[tuple[GridPos, list[GridPos]]] = deque()
        queue.append(((sx, sy), [(sx, sy)]))
        visited = {(sx, sy)}

        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == (gx, gy):
                return path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and self._roads[ny, nx] and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        # Fallback: straight line (no road path exists)
        return [start, goal]

    def _nearest_road(self, x: int, y: int) -> tuple[int, int]:
        """Find nearest road pixel to (x, y)."""
        road_yx = np.argwhere(self._roads)
        if len(road_yx) == 0:
            return x, y
        dists = np.abs(road_yx[:, 1] - x) + np.abs(road_yx[:, 0] - y)
        idx = int(np.argmin(dists))
        return int(road_yx[idx, 1]), int(road_yx[idx, 0])

    # ------------------------------------------------------------------

    def step(self, dt: float = 1.0) -> None:
        """Advance target along road path."""
        if self.reached_goal:
            return
        self.step_count += 1

        steps_to_take = max(1, int(self.speed * dt))
        for _ in range(steps_to_take):
            if self._path_idx >= len(self._road_path) - 1:
                self.reached_goal = True
                return
            self._path_idx += 1
            nx, ny = self._road_path[self._path_idx]
            self.position[0] = float(nx)
            self.position[1] = float(ny)

    @property
    def current_position(self) -> np.ndarray:
        """Current [x, y] float position."""
        return self.position.copy()

    def get_buffer_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """[H, W] bool — Manhattan buffer around target."""
        H, W = shape
        mask = np.zeros((H, W), dtype=bool)
        tx, ty = int(self.position[0]), int(self.position[1])
        r = self.buffer_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dy) + abs(dx) <= r:
                    ny, nx = ty + dy, tx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        mask[ny, nx] = True
        return mask
