"""Fire Cellular Automaton model (DC-1, FD-1 through FD-5).

Isotropic fire spread on a 2D grid. NO WIND. 8-neighbor Moore neighborhood.
State per cell: UNBURNED(0) → BURNING(1) → BURNED_OUT(2).
"""

from __future__ import annotations

import numpy as np

# Cell states (FD-1)
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2

# Landuse spread probabilities
_LANDUSE_PROB = {
    0: 0.02,  # empty
    1: 0.15,  # forest (tuned for partial blockages with navigable detours)
    2: 0.06,  # urban
    3: 0.03,  # industrial
    4: 0.00,  # water — never burns
}

# 8-connected Moore neighborhood (FD-2: isotropic, equal probability all 8)
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class FireSpreadModel:
    """Cellular automaton fire spread model.

    Isotropic spread (FD-2): equal p_spread for all 8 neighbors.
    NO wind parameters (FD-5). All randomness via caller-supplied rng (DC-1).
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
        n_ignition: int = 3,
        landuse_map: np.ndarray | None = None,
        roads_mask: np.ndarray | None = None,
        corridor_cells: list[tuple[int, int]] | None = None,
    ) -> None:
        self._rng = rng
        self._H, self._W = map_shape

        # State arrays
        self._state = np.full((self._H, self._W), UNBURNED, dtype=np.int8)
        self._burn_timer = np.zeros((self._H, self._W), dtype=np.float32)
        self._burnout_time = rng.uniform(
            100.0, 200.0, size=(self._H, self._W)
        ).astype(np.float32)
        self._smoke = np.zeros((self._H, self._W), dtype=np.float32)

        # Landuse (default: all forest=1 if not provided)
        if landuse_map is not None:
            self._landuse = landuse_map.astype(np.int8)
        else:
            self._landuse = np.ones((self._H, self._W), dtype=np.int8)

        # Roads act as 50% firebreak
        self._roads = (
            roads_mask.astype(bool) if roads_mask is not None
            else np.zeros((self._H, self._W), dtype=bool)
        )

        # Initial ignition — corridor-aware placement
        if n_ignition > 0:
            self._ignite(n_ignition, corridor_cells)

    # -- Public properties --

    @property
    def fire_mask(self) -> np.ndarray:
        """bool[H, W]: True where cell is currently burning."""
        return (self._state == BURNING).copy()

    @property
    def burned_mask(self) -> np.ndarray:
        """bool[H, W]: True where cell has burned out."""
        return (self._state == BURNED_OUT).copy()

    @property
    def smoke_mask(self) -> np.ndarray:
        """float32[H, W] in [0, 1]: smoke concentration."""
        return self._smoke.copy()

    @property
    def total_affected(self) -> int:
        """Count of cells that are burning or burned out."""
        return int((self._state > UNBURNED).sum())

    # -- Step --

    def step(self, dt: float = 1.0) -> None:
        """Advance fire by one timestep (FD-1, FD-4).

        1. Spread: burning cells attempt to ignite neighbors (isotropic).
        2. Burnout: cells that have burned long enough become BURNED_OUT.
        3. Smoke: update smoke field (no wind advection).
        """
        burning = self._state == BURNING

        # Increment burn timers for burning cells
        self._burn_timer[burning] += dt

        # --- Spread (FD-2: isotropic, equal probability all 8 neighbors) ---
        # Deduplicate: when multiple burning neighbors nominate the same
        # unburned cell, keep only the max probability. This avoids wasting
        # RNG draws on duplicates (FD-4 determinism stability).
        candidate_map: dict[tuple[int, int], float] = {}

        burning_ys, burning_xs = np.where(burning)
        for by, bx in zip(burning_ys, burning_xs):
            for dy, dx in _NEIGHBORS:
                ny, nx = by + dy, bx + dx
                if not (0 <= ny < self._H and 0 <= nx < self._W):
                    continue
                if self._state[ny, nx] != UNBURNED:
                    continue

                # Base probability from landuse (same for all 8 directions)
                lu = int(self._landuse[ny, nx])
                prob = _LANDUSE_PROB.get(lu, 0.02)

                # Road firebreak: halve probability
                if self._roads[ny, nx]:
                    prob *= 0.5

                key = (int(ny), int(nx))
                if key not in candidate_map or prob > candidate_map[key]:
                    candidate_map[key] = prob

        # Stochastic ignition of unique candidates
        if candidate_map:
            cells = list(candidate_map.keys())
            probs = np.array([candidate_map[c] for c in cells])
            rolls = self._rng.random(len(cells))
            for i, (cy, cx) in enumerate(cells):
                if rolls[i] < probs[i]:
                    self._state[cy, cx] = BURNING

        # --- Burnout ---
        burnout_mask = burning & (self._burn_timer >= self._burnout_time)
        self._state[burnout_mask] = BURNED_OUT

        # --- Smoke ---
        self._update_smoke()

    # -- Test hook --

    def force_cell_state(self, x: int, y: int, state: int) -> None:
        """Set cell state directly. FOR TESTS ONLY.

        Coordinates: x=col, y=row. Access: _state[y, x].
        """
        self._state[y, x] = state

    # -- Internal --

    def _ignite(
        self,
        n: int,
        corridor_cells: list[tuple[int, int]] | None = None,
    ) -> None:
        """Ignite n cells. When corridor_cells is provided, place at least
        half of ignitions near the corridor (within 20-40 cells offset)
        so fire meaningfully interacts with planned paths."""
        n_near = 0
        near_buffer = 8  # cells offset from corridor (close for path interaction)

        if corridor_cells and len(corridor_cells) > 2:
            n_near = max(1, n // 2)  # at least 1 near corridor
            n_random = n - n_near

            # Build a mask of cells near the corridor
            near_mask = np.zeros((self._H, self._W), dtype=bool)
            for cx, cy in corridor_cells:
                y0 = max(0, cy - near_buffer)
                y1 = min(self._H, cy + near_buffer + 1)
                x0 = max(0, cx - near_buffer)
                x1 = min(self._W, cx + near_buffer + 1)
                near_mask[y0:y1, x0:x1] = True

            # Exclude the corridor itself (don't ignite ON the path)
            for cx, cy in corridor_cells:
                if 0 <= cy < self._H and 0 <= cx < self._W:
                    near_mask[cy, cx] = False

            # Near-corridor ignitions: forest cells near corridor
            near_forest = near_mask & (self._landuse == 1)
            near_ys, near_xs = np.where(near_forest)

            placed_near = 0
            if len(near_ys) >= n_near:
                indices = self._rng.choice(
                    len(near_ys), size=n_near, replace=False,
                )
                for idx in indices:
                    self._state[near_ys[idx], near_xs[idx]] = BURNING
                    placed_near += 1

            # Remaining as random
            n_random += (n_near - placed_near)
            if n_random > 0:
                self._ignite_random(n_random)
        else:
            self._ignite_random(n)

    def _ignite_random(self, n: int) -> None:
        """Ignite n cells at random, preferring forest (landuse=1)."""
        forest_ys, forest_xs = np.where(
            (self._landuse == 1) & (self._state == UNBURNED),
        )
        if len(forest_ys) >= n:
            indices = self._rng.choice(len(forest_ys), size=n, replace=False)
            for idx in indices:
                self._state[forest_ys[idx], forest_xs[idx]] = BURNING
        else:
            valid_ys, valid_xs = np.where(
                (self._landuse != 4) & (self._state == UNBURNED),
            )
            if len(valid_ys) > 0:
                k = min(n, len(valid_ys))
                indices = self._rng.choice(len(valid_ys), size=k, replace=False)
                for idx in indices:
                    self._state[valid_ys[idx], valid_xs[idx]] = BURNING

    def _update_smoke(self) -> None:
        """Update smoke field: source from fire, diffuse (no wind advection)."""
        # Source: burning=1.0, burned_out=0.3
        source = np.zeros_like(self._smoke)
        source[self._state == BURNING] = 1.0
        source[self._state == BURNED_OUT] = 0.3

        # Diffuse with 3x3 box blur (2 passes)
        blurred = source.copy()
        for _ in range(2):
            padded = np.pad(blurred, 1, mode="constant", constant_values=0)
            blurred = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
                + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
                + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0

        # Persistence (tuned for thinner smoke halos with navigable gaps)
        self._smoke = np.clip(
            0.7 * self._smoke + 0.45 * blurred, 0.0, 1.0
        ).astype(np.float32)
