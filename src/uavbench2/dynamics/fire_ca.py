"""Fire Cellular Automaton model (DC-1 compliant).

Stochastic fire spread on a 2D grid with wind-driven advection,
burnout, and smoke generation.

State per cell: UNBURNED(0) → BURNING(1) → BURNED_OUT(2).
"""

from __future__ import annotations

import numpy as np

# Cell states
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2

# Landuse spread probabilities
_LANDUSE_PROB = {
    0: 0.02,  # empty
    1: 0.30,  # forest
    2: 0.10,  # urban
    3: 0.05,  # industrial
    4: 0.00,  # water — never burns
}

# 4-connected neighbor offsets (dy, dx)
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class FireSpreadModel:
    """Cellular automaton fire spread model.

    All randomness flows through caller-supplied rng (DC-1).
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
        n_ignition: int = 3,
        wind_speed: float = 0.2,
        wind_dir: float = 0.0,
        landuse_map: np.ndarray | None = None,
        roads_mask: np.ndarray | None = None,
    ) -> None:
        self._rng = rng
        self._H, self._W = map_shape
        self._wind_speed = np.clip(wind_speed, 0.0, 1.0)
        self._wind_dir = wind_dir

        # State arrays
        self._state = np.full((self._H, self._W), UNBURNED, dtype=np.int8)
        self._burn_timer = np.zeros((self._H, self._W), dtype=np.float32)
        self._burnout_time = rng.uniform(
            30.0, 60.0, size=(self._H, self._W)
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

        # Precompute wind modifiers for 4 directions
        self._wind_mod = self._compute_wind_modifiers()

        # Initial ignition
        if n_ignition > 0:
            self._ignite(n_ignition)

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
        """Advance fire by one timestep.

        1. Spread: burning cells attempt to ignite neighbors.
        2. Burnout: cells that have burned long enough become BURNED_OUT.
        3. Smoke: update smoke field.
        """
        burning = self._state == BURNING

        # Increment burn timers for burning cells
        self._burn_timer[burning] += dt

        # --- Spread ---
        candidates_y, candidates_x = [], []
        candidate_probs = []

        burning_ys, burning_xs = np.where(burning)
        for by, bx in zip(burning_ys, burning_xs):
            for di, (dy, dx) in enumerate(_NEIGHBORS):
                ny, nx = by + dy, bx + dx
                if not (0 <= ny < self._H and 0 <= nx < self._W):
                    continue
                if self._state[ny, nx] != UNBURNED:
                    continue

                # Base probability from landuse
                lu = int(self._landuse[ny, nx])
                prob = _LANDUSE_PROB.get(lu, 0.02)

                # Wind modifier
                prob *= self._wind_mod[di]

                # Road firebreak: halve probability
                if self._roads[ny, nx]:
                    prob *= 0.5

                candidates_y.append(ny)
                candidates_x.append(nx)
                candidate_probs.append(prob)

        # Stochastic ignition of candidates
        if candidates_y:
            rolls = self._rng.random(len(candidates_y))
            probs = np.array(candidate_probs)
            ignite_mask = rolls < probs
            for i in range(len(candidates_y)):
                if ignite_mask[i]:
                    self._state[candidates_y[i], candidates_x[i]] = BURNING

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

    def _ignite(self, n: int) -> None:
        """Ignite n cells, preferring forest (landuse=1)."""
        forest_ys, forest_xs = np.where(self._landuse == 1)
        if len(forest_ys) >= n:
            indices = self._rng.choice(len(forest_ys), size=n, replace=False)
            for idx in indices:
                self._state[forest_ys[idx], forest_xs[idx]] = BURNING
        else:
            # Fall back to any non-water cell
            valid_ys, valid_xs = np.where(self._landuse != 4)
            if len(valid_ys) > 0:
                k = min(n, len(valid_ys))
                indices = self._rng.choice(len(valid_ys), size=k, replace=False)
                for idx in indices:
                    self._state[valid_ys[idx], valid_xs[idx]] = BURNING

    def _compute_wind_modifiers(self) -> list[float]:
        """Compute spread probability multipliers for each direction."""
        # Wind blows FROM wind_dir; fire spreads in the downwind direction
        # Directions: UP(-1,0), DOWN(1,0), LEFT(0,-1), RIGHT(0,1)
        dir_angles = [np.pi, 0.0, np.pi / 2, 3 * np.pi / 2]
        mods = []
        for angle in dir_angles:
            alignment = np.cos(self._wind_dir - angle)
            mod = 1.0 + self._wind_speed * alignment
            mods.append(max(0.1, mod))
        return mods

    def _update_smoke(self) -> None:
        """Update smoke field: source from fire, diffuse, advect with wind."""
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

        # Wind advection via shift (NOT np.roll — no wrapping)
        shift_y = int(round(np.cos(self._wind_dir) * self._wind_speed * 3))
        shift_x = int(round(np.sin(self._wind_dir) * self._wind_speed * 3))
        if shift_y != 0 or shift_x != 0:
            advected = np.zeros_like(blurred)
            src_y0 = max(0, -shift_y)
            src_y1 = min(self._H, self._H - shift_y)
            src_x0 = max(0, -shift_x)
            src_x1 = min(self._W, self._W - shift_x)
            dst_y0 = max(0, shift_y)
            dst_y1 = min(self._H, self._H + shift_y)
            dst_x0 = max(0, shift_x)
            dst_x1 = min(self._W, self._W + shift_x)
            advected[dst_y0:dst_y1, dst_x0:dst_x1] = blurred[src_y0:src_y1, src_x0:src_x1]
            blurred = advected

        # Persistence
        self._smoke = np.clip(
            0.85 * self._smoke + 0.6 * blurred, 0.0, 1.0
        ).astype(np.float32)
