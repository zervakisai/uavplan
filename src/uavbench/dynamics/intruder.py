"""Intruder model for border patrol scenarios — forces replanning.

Multiple intruders spawn along a map edge and move toward a sensitive
area, crossing the UAV's patrol path.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

GridPos = Tuple[int, int]


class IntruderModel:
    """Border intruders moving toward a target area.

    Args:
        map_shape: (H, W)
        num_intruders: count
        spawn_zone: "north" | "south" | "east" | "west"
        target_area: (x, y) centre of sensitive zone
        speed: cells per step
        buffer_radius: Manhattan blocking radius
        rng: deterministic RNG
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        num_intruders: int,
        spawn_zone: str,
        target_area: GridPos,
        speed: float = 0.5,
        buffer_radius: int = 6,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.height, self.width = map_shape
        self.num_intruders = num_intruders
        self.target = target_area
        self.speed = speed
        self.buffer_radius = buffer_radius
        self._rng = rng if rng is not None else np.random.default_rng()

        self.positions = self._spawn(spawn_zone).astype(np.float32)
        self.reached_target = np.zeros(num_intruders, dtype=bool)
        self.step_count = 0

    # ------------------------------------------------------------------

    def _spawn(self, zone: str) -> np.ndarray:
        """Generate spawn positions along the given map edge.

        Returns [N, 2] as (x, y).
        """
        margin = 15
        n = self.num_intruders
        positions = np.empty((n, 2), dtype=np.float32)

        for i in range(n):
            if zone == "north":
                positions[i] = [self._rng.integers(margin, self.width - margin), margin]
            elif zone == "south":
                positions[i] = [self._rng.integers(margin, self.width - margin),
                                self.height - margin]
            elif zone == "east":
                positions[i] = [self.width - margin,
                                self._rng.integers(margin, self.height - margin)]
            elif zone == "west":
                positions[i] = [margin,
                                self._rng.integers(margin, self.height - margin)]
            else:
                positions[i] = [self._rng.integers(margin, self.width - margin), margin]

        return positions

    # ------------------------------------------------------------------

    def step(self, dt: float = 1.0) -> None:
        """Move each intruder toward the target with noise."""
        self.step_count += 1

        for i in range(self.num_intruders):
            if self.reached_target[i]:
                continue

            dx = self.target[0] - self.positions[i, 0]
            dy = self.target[1] - self.positions[i, 1]
            dist = max(float(np.sqrt(dx * dx + dy * dy)), 1e-6)

            if dist < 2.0:
                self.reached_target[i] = True
                continue

            noise_x = float(self._rng.normal(0, 0.15))
            noise_y = float(self._rng.normal(0, 0.15))

            self.positions[i, 0] += (dx / dist) * self.speed * dt + noise_x
            self.positions[i, 1] += (dy / dist) * self.speed * dt + noise_y

            self.positions[i, 0] = float(np.clip(self.positions[i, 0], 0, self.width - 1))
            self.positions[i, 1] = float(np.clip(self.positions[i, 1], 0, self.height - 1))

    @property
    def active_positions(self) -> np.ndarray:
        """[M, 2] (x, y) of intruders that have not reached target."""
        active = ~self.reached_target
        if active.any():
            return self.positions[active].copy()
        return np.empty((0, 2), dtype=np.float32)

    def get_buffer_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """[H, W] bool — Manhattan buffer around all active intruders."""
        H, W = shape
        mask = np.zeros((H, W), dtype=bool)
        r = self.buffer_radius

        for pos in self.active_positions:
            ix, iy = int(pos[0]), int(pos[1])
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dy) + abs(dx) <= r:
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            mask[ny, nx] = True
        return mask
