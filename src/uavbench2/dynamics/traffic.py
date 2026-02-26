"""Traffic model (DC-1 compliant).

Vehicles move along road network, avoid fire, generate occupancy masks.
All randomness flows through caller-supplied rng.
"""

from __future__ import annotations

import numpy as np

# 4-connected movement: (dy, dx)
_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class TrafficModel:
    """Emergency vehicle traffic on road network.

    Vehicles move toward random targets on roads, avoiding fire.
    Generates occupancy mask with Manhattan buffer for blocking.
    """

    def __init__(
        self,
        roads_mask: np.ndarray,
        num_vehicles: int,
        rng: np.random.Generator,
        buffer_radius: int = 5,
    ) -> None:
        self._rng = rng
        self._roads = roads_mask.astype(bool)
        self._H, self._W = roads_mask.shape
        self._buffer_radius = buffer_radius

        # Find road pixels
        road_ys, road_xs = np.where(self._roads)
        self._road_pixels = np.column_stack((road_ys, road_xs))

        # Clamp vehicles to available road pixels
        n = min(num_vehicles, len(self._road_pixels))
        self._num_vehicles = n

        if n == 0:
            self._positions = np.zeros((0, 2), dtype=np.int32)
            self._targets = np.zeros((0, 2), dtype=np.int32)
        else:
            # Initial positions (y, x) on roads
            indices = rng.choice(len(self._road_pixels), size=n, replace=False)
            self._positions = self._road_pixels[indices].copy()
            self._targets = self._pick_targets()

        self._fire_avoidance_events = 0

    # -- Public properties --

    @property
    def vehicle_positions(self) -> np.ndarray:
        """int[N, 2] as (y, x): copy of current positions."""
        return self._positions.copy()

    @property
    def fire_avoidance_events(self) -> int:
        return self._fire_avoidance_events

    # -- Step --

    def step(self, dt: float = 1.0, fire_mask: np.ndarray | None = None) -> None:
        """Move each vehicle one step toward its target.

        Avoids fire cells. Picks new target when close to current one.
        """
        self._fire_avoidance_events = 0

        for i in range(self._num_vehicles):
            y, x = self._positions[i]
            ty, tx = self._targets[i]

            # Check if near target — pick new one
            dist = abs(y - ty) + abs(x - tx)
            if dist <= 3:
                self._targets[i] = self._pick_one_target()
                ty, tx = self._targets[i]

            # Greedy move toward target
            best_move = None
            best_dist = dist

            for dy, dx in _MOVES:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self._H and 0 <= nx < self._W):
                    continue
                if not self._roads[ny, nx]:
                    continue
                if fire_mask is not None and fire_mask[ny, nx]:
                    self._fire_avoidance_events += 1
                    continue

                d = abs(ny - ty) + abs(nx - tx)
                if d < best_dist:
                    best_dist = d
                    best_move = (ny, nx)

            if best_move is not None:
                self._positions[i] = best_move

    def get_occupancy_mask(
        self,
        shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """bool[H, W]: Manhattan buffer around each vehicle."""
        H, W = shape if shape is not None else (self._H, self._W)
        mask = np.zeros((H, W), dtype=bool)
        r = self._buffer_radius

        for i in range(self._num_vehicles):
            vy, vx = self._positions[i]
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dy) + abs(dx) <= r:
                        ny, nx = vy + dy, vx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            mask[ny, nx] = True

        return mask

    # -- Internal --

    def _pick_targets(self) -> np.ndarray:
        """Pick random road targets for all vehicles."""
        if len(self._road_pixels) == 0:
            return np.zeros((self._num_vehicles, 2), dtype=np.int32)
        indices = self._rng.choice(
            len(self._road_pixels), size=self._num_vehicles
        )
        return self._road_pixels[indices].copy()

    def _pick_one_target(self) -> np.ndarray:
        """Pick a single random road target."""
        if len(self._road_pixels) == 0:
            return np.array([0, 0], dtype=np.int32)
        idx = self._rng.integers(len(self._road_pixels))
        return self._road_pixels[idx].copy()
