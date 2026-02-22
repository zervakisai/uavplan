"""Emergency traffic model using road mask for pixel-space vehicle movement.

Vehicles move along road pixels toward random destinations, creating
dynamic obstacles for UAV path planning.
"""

from __future__ import annotations

import numpy as np


class TrafficModel:
    """Pixel-space vehicle movement model on road networks.

    Args:
        roads_mask: bool [H, W] — road pixel mask
        num_vehicles: int — number of emergency vehicles
        rng: np.random.Generator — for deterministic behavior
    """

    def __init__(
        self,
        roads_mask: np.ndarray,
        num_vehicles: int,
        rng: np.random.Generator,
    ) -> None:
        self._roads = roads_mask.astype(bool)
        self._rng = rng

        # Cache all road pixel coordinates [M, 2] as (y, x)
        self._road_pixels = np.argwhere(self._roads)

        if len(self._road_pixels) == 0:
            num_vehicles = 0

        n = min(num_vehicles, len(self._road_pixels))

        # Initialize vehicle positions on random road pixels
        if n > 0:
            indices = rng.choice(len(self._road_pixels), size=n, replace=False)
            self._positions = self._road_pixels[indices].copy()  # [N, 2] (y, x)
            self._targets = self._pick_targets(n)
        else:
            self._positions = np.empty((0, 2), dtype=np.int32)
            self._targets = np.empty((0, 2), dtype=np.int32)

    def _pick_targets(self, n: int) -> np.ndarray:
        """Pick n random road pixels as movement targets."""
        indices = self._rng.choice(len(self._road_pixels), size=n)
        return self._road_pixels[indices].copy()

    def step(self, dt: float = 1.0, fire_mask: np.ndarray | None = None) -> None:
        """Move each vehicle one pixel toward its target."""
        self._fire_avoidance_events = 0

        if len(self._positions) == 0:
            return

        H, W = self._roads.shape

        for i in range(len(self._positions)):
            py, px = self._positions[i]
            ty, tx = self._targets[i]

            # Check if at target (within 3 pixels)
            if abs(py - ty) + abs(px - tx) <= 3:
                self._targets[i] = self._road_pixels[
                    self._rng.integers(len(self._road_pixels))
                ]
                ty, tx = self._targets[i]

            # Compute direction toward target
            dy = np.sign(ty - py)
            dx = np.sign(tx - px)

            # Try primary direction (larger delta first)
            moved = False
            if abs(ty - py) >= abs(tx - px):
                attempts = [(dy, 0), (0, dx), (dy, dx)]
            else:
                attempts = [(0, dx), (dy, 0), (dy, dx)]

            for ay, ax in attempts:
                if ay == 0 and ax == 0:
                    continue
                ny = int(py + ay)
                nx = int(px + ax)
                if 0 <= ny < H and 0 <= nx < W and self._roads[ny, nx]:
                    if fire_mask is not None and fire_mask[ny, nx]:
                        self._fire_avoidance_events += 1
                        continue  # skip this direction
                    self._positions[i] = [ny, nx]
                    moved = True
                    break

            # If stuck, try any adjacent road pixel
            if not moved:
                for ay, ax in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny = int(py + ay)
                    nx = int(px + ax)
                    if 0 <= ny < H and 0 <= nx < W and self._roads[ny, nx]:
                        if fire_mask is not None and fire_mask[ny, nx]:
                            self._fire_avoidance_events += 1
                            continue  # skip this direction
                        self._positions[i] = [ny, nx]
                        break

    @property
    def vehicle_positions(self) -> np.ndarray:
        """[N, 2] int — (y, x) pixel positions of all vehicles."""
        return self._positions.copy()

    @property
    def fire_avoidance_events(self) -> int:
        return getattr(self, "_fire_avoidance_events", 0)

    def get_occupancy_mask(
        self, shape: tuple[int, int], buffer_radius: int = 5
    ) -> np.ndarray:
        """[H, W] bool — cells within buffer_radius (Manhattan) of any vehicle."""
        mask = np.zeros(shape, dtype=bool)
        H, W = shape

        for i in range(len(self._positions)):
            py, px = self._positions[i]
            iy, ix = int(py), int(px)
            y_lo = max(0, iy - buffer_radius)
            y_hi = min(H, iy + buffer_radius + 1)
            x_lo = max(0, ix - buffer_radius)
            x_hi = min(W, ix + buffer_radius + 1)
            for ny in range(y_lo, y_hi):
                for nx in range(x_lo, x_hi):
                    if abs(ny - iy) + abs(nx - ix) <= buffer_radius:
                        mask[ny, nx] = True

        return mask
