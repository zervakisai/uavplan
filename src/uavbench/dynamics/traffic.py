"""Traffic model (DC-1 compliant).

Vehicles move along road network, avoid fire, generate occupancy masks.
Corridor vehicles patrol directly along the drone's reference corridor path.
All randomness flows through caller-supplied rng.
"""

from __future__ import annotations

import numpy as np

# 4-connected movement: (dy, dx)
_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class TrafficModel:
    """Emergency vehicle traffic on road network + corridor patrol.

    Road vehicles move toward random targets on roads, avoiding fire.
    Corridor vehicles patrol back-and-forth along the reference corridor,
    directly crossing the drone's planned flight path.
    Generates occupancy mask with Manhattan buffer for blocking.
    """

    def __init__(
        self,
        roads_mask: np.ndarray,
        num_vehicles: int,
        rng: np.random.Generator,
        buffer_radius: int = 5,
        corridor_cells: list[tuple[int, int]] | None = None,
        num_corridor_vehicles: int = 0,
        roadblock_cells: list[tuple[int, int]] | None = None,
        roadblock_step: int | None = None,
    ) -> None:
        self._rng = rng
        self._roads = roads_mask.astype(bool)
        self._H, self._W = roads_mask.shape
        self._buffer_radius = buffer_radius

        # Roadblock state: vehicles that freeze at corridor chokepoints
        self._roadblock_targets = roadblock_cells or []  # (y, x) road cells
        self._roadblock_step = roadblock_step
        self._roadblock_frozen: list[bool] = []  # per-vehicle freeze flag

        # Find road pixels
        road_ys, road_xs = np.where(self._roads)
        self._road_pixels = np.column_stack((road_ys, road_xs))

        # Corridor patrol state
        self._n_corridor = 0
        self._corridor_path_yx: np.ndarray | None = None  # [L, 2] as (y, x)
        self._corridor_indices: list[int] = []  # per-vehicle path index
        self._corridor_directions: list[int] = []  # +1 or -1
        self._corridor_seg_lo: list[int] = []  # patrol segment lower bound
        self._corridor_seg_hi: list[int] = []  # patrol segment upper bound

        # Total vehicles = corridor + road vehicles
        # Road vehicles are clamped to available road pixels
        n_corr = 0
        if corridor_cells and len(corridor_cells) >= 3:
            n_corr = min(num_corridor_vehicles, num_vehicles)

        n_road = min(num_vehicles - n_corr, len(self._road_pixels))
        self._num_vehicles = n_corr + n_road

        if self._num_vehicles == 0:
            self._positions = np.zeros((0, 2), dtype=np.int32)
            self._targets = np.zeros((0, 2), dtype=np.int32)
        else:
            # Place corridor vehicles on the actual corridor path
            corr_positions, corr_targets = self._place_corridor_vehicles(
                n_corr, corridor_cells
            )

            # Place road vehicles randomly on roads
            if n_road > 0:
                indices = rng.choice(
                    len(self._road_pixels), size=n_road, replace=False
                )
                rand_positions = self._road_pixels[indices].copy()
                rand_targets = self._pick_n_targets(n_road)
            else:
                rand_positions = np.zeros((0, 2), dtype=np.int32)
                rand_targets = np.zeros((0, 2), dtype=np.int32)

            if len(corr_positions) > 0:
                self._positions = np.vstack([corr_positions, rand_positions])
                self._targets = np.vstack([corr_targets, rand_targets])
            else:
                self._positions = rand_positions
                self._targets = rand_targets

        # Initialize per-vehicle freeze flags (all unfrozen initially)
        self._roadblock_frozen = [False] * self._num_vehicles

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

    def step(
        self,
        dt: float = 1.0,
        fire_mask: np.ndarray | None = None,
        step_idx: int = 0,
    ) -> None:
        """Move each vehicle one step.

        Corridor vehicles patrol along the reference corridor path.
        Road vehicles move greedy toward random targets on roads.
        All avoid fire. Roadblock vehicles freeze when activated.
        """
        self._fire_avoidance_events = 0

        # Activate roadblocks at the designated step
        if (self._roadblock_targets
                and self._roadblock_step is not None
                and step_idx >= self._roadblock_step):
            self._activate_roadblocks()

        for i in range(self._num_vehicles):
            # Skip frozen roadblock vehicles
            if self._roadblock_frozen[i]:
                continue

            if i < self._n_corridor:
                self._step_corridor_vehicle(i, fire_mask)
            else:
                self._step_road_vehicle(i, fire_mask)

    def _step_corridor_vehicle(self, i: int, fire_mask: np.ndarray | None) -> None:
        """Advance corridor vehicle within its bounded patrol segment.

        Each vehicle patrols back-and-forth within a fixed segment of the
        corridor. When reaching segment edges, it reverses direction.
        This ensures it stays in one area and the drone must pass through.
        """
        path = self._corridor_path_yx
        if path is None:
            return

        idx = self._corridor_indices[i]
        direction = self._corridor_directions[i]
        lo = self._corridor_seg_lo[i]
        hi = self._corridor_seg_hi[i]

        next_idx = idx + direction

        # Reverse at segment boundaries
        if next_idx < lo or next_idx > hi:
            direction = -direction
            self._corridor_directions[i] = direction
            next_idx = idx + direction

        if not (lo <= next_idx <= hi and 0 <= next_idx < len(path)):
            return  # safety

        ny, nx = int(path[next_idx, 0]), int(path[next_idx, 1])

        # Avoid fire — try reversing direction
        if fire_mask is not None and fire_mask[ny, nx]:
            self._fire_avoidance_events += 1
            direction = -direction
            self._corridor_directions[i] = direction
            alt_idx = idx + direction
            if lo <= alt_idx <= hi and 0 <= alt_idx < len(path):
                ay, ax = int(path[alt_idx, 0]), int(path[alt_idx, 1])
                if fire_mask is not None and fire_mask[ay, ax]:
                    return  # stuck in fire on both sides
                ny, nx = ay, ax
                next_idx = alt_idx
            else:
                return  # stuck at segment edge with fire

        self._positions[i] = [ny, nx]
        self._corridor_indices[i] = next_idx

    def _step_road_vehicle(self, i: int, fire_mask: np.ndarray | None) -> None:
        """Move road vehicle greedy toward target, with random walk fallback."""
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
        valid_moves = []  # fallback: any valid road neighbor

        for dy, dx in _MOVES:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < self._H and 0 <= nx < self._W):
                continue
            if not self._roads[ny, nx]:
                continue
            if fire_mask is not None and fire_mask[ny, nx]:
                self._fire_avoidance_events += 1
                continue

            valid_moves.append((ny, nx))
            d = abs(ny - ty) + abs(nx - tx)
            if d < best_dist:
                best_dist = d
                best_move = (ny, nx)

        if best_move is not None:
            self._positions[i] = best_move
        elif valid_moves:
            # Stuck (dead-end or local minimum) — random walk to keep moving
            idx = self._rng.integers(len(valid_moves))
            self._positions[i] = valid_moves[idx]
            # Pick a new target so we don't keep getting stuck
            self._targets[i] = self._pick_one_target()

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

    # -- Roadblock API --

    def _activate_roadblocks(self) -> None:
        """Freeze the closest vehicle to each roadblock target cell.

        Called once when step_idx >= roadblock_step. Each roadblock target
        (y, x) snaps the nearest unfrozen vehicle to that position and
        freezes it permanently (until clear_roadblocks).
        """
        for ry, rx in self._roadblock_targets:
            # Find closest unfrozen vehicle
            best_i = -1
            best_dist = float("inf")
            for i in range(self._num_vehicles):
                if self._roadblock_frozen[i]:
                    continue
                vy, vx = self._positions[i]
                d = abs(vy - ry) + abs(vx - rx)
                if d < best_dist:
                    best_dist = d
                    best_i = i
            if best_i >= 0:
                self._positions[best_i] = [ry, rx]
                self._roadblock_frozen[best_i] = True

        # Clear targets so we don't re-activate next step
        self._roadblock_targets = []

    def clear_roadblocks(self) -> None:
        """Unfreeze all roadblock vehicles (guardrail D1 relaxation)."""
        for i in range(self._num_vehicles):
            self._roadblock_frozen[i] = False

    @property
    def has_active_roadblocks(self) -> bool:
        """True if any vehicle is currently frozen as a roadblock."""
        return any(self._roadblock_frozen)

    # -- Internal --

    def _place_corridor_vehicles(
        self,
        n_corr: int,
        corridor_cells: list[tuple[int, int]] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place n_corr vehicles directly on the corridor path.

        Corridor vehicles patrol back-and-forth along the reference corridor,
        ensuring they cross the drone's planned flight path.
        Returns (positions[n, 2], targets[n, 2]) as (y, x) arrays.
        """
        if n_corr <= 0 or not corridor_cells or len(corridor_cells) < 3:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

        # Convert corridor from (x, y) to (y, x) for consistency with positions
        path_yx = np.array(
            [(y, x) for x, y in corridor_cells], dtype=np.int32
        )
        self._corridor_path_yx = path_yx
        self._n_corridor = n_corr

        # Place vehicles at evenly-spaced points along corridor interior
        # Each vehicle patrols a bounded segment (~patrol_half cells each way)
        interior_len = len(path_yx) - 2  # skip start (0) and goal (-1)
        step = max(1, interior_len // (n_corr + 1))
        patrol_half = max(15, step // 2)  # half-width of patrol segment

        positions = []
        self._corridor_indices = []
        self._corridor_directions = []
        self._corridor_seg_lo = []
        self._corridor_seg_hi = []

        for i in range(n_corr):
            # +1 offset to skip start cell
            path_idx = min((i + 1) * step, len(path_yx) - 2)
            positions.append(path_yx[path_idx].copy())
            self._corridor_indices.append(path_idx)
            # Alternate patrol direction for variety
            self._corridor_directions.append(1 if i % 2 == 0 else -1)
            # Bounded patrol segment: [center - patrol_half, center + patrol_half]
            seg_lo = max(1, path_idx - patrol_half)
            seg_hi = min(len(path_yx) - 2, path_idx + patrol_half)
            self._corridor_seg_lo.append(seg_lo)
            self._corridor_seg_hi.append(seg_hi)

        # Consume RNG to maintain stream compatibility with old code
        for _ in range(n_corr):
            self._rng.integers(max(1, interior_len))

        targets = [pos.copy() for pos in positions]  # unused for corridor vehicles
        return (
            np.array(positions, dtype=np.int32),
            np.array(targets, dtype=np.int32),
        )

    def _pick_n_targets(self, n: int) -> np.ndarray:
        """Pick n random road targets."""
        if len(self._road_pixels) == 0:
            return np.zeros((n, 2), dtype=np.int32)
        indices = self._rng.choice(len(self._road_pixels), size=n)
        return self._road_pixels[indices].copy()

    def _pick_targets(self) -> np.ndarray:
        """Pick random road targets for all vehicles."""
        return self._pick_n_targets(self._num_vehicles)

    def _pick_one_target(self) -> np.ndarray:
        """Pick a single random road target."""
        if len(self._road_pixels) == 0:
            return np.array([0, 0], dtype=np.int32)
        idx = self._rng.integers(len(self._road_pixels))
        return self._road_pixels[idx].copy()
