"""Fire Cellular Automaton model (DC-1, FD-1 through FD-5).

Isotropic fire spread on a 2D grid. NO WIND. 8-neighbor Moore neighborhood.
State per cell: UNBURNED(0) → BURNING(1) → BURNED_OUT(2).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

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

# Pre-computed Moore dilation structure
_MOORE_STRUCT = np.ones((3, 3), dtype=bool)


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

        self._step_count = 0

        # Landuse (default: urban=2 for realistic spread in urban grids)
        if landuse_map is not None:
            self._landuse = landuse_map.astype(np.int8)
        else:
            self._landuse = np.full((self._H, self._W), 2, dtype=np.int8)

        # Pre-compute probability map (avoid rebuilding each step)
        self._prob_map = np.zeros((self._H, self._W), dtype=np.float32)
        for lu_val, prob in _LANDUSE_PROB.items():
            self._prob_map[self._landuse == lu_val] = prob

        # Roads act as 50% firebreak
        self._roads = (
            roads_mask.astype(bool) if roads_mask is not None
            else np.zeros((self._H, self._W), dtype=bool)
        )

        # Apply road firebreak to pre-computed prob map
        self._prob_map[self._roads] *= 0.5

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

        # --- Spread (FD-2: vectorized isotropic 8-neighbor Moore) ---
        if burning.any():
            # Moore neighborhood structure (8-connected, FD-2)
            spread_candidates = (
                binary_dilation(burning, structure=_MOORE_STRUCT)
                & (self._state == UNBURNED)
            )

            if spread_candidates.any():
                # Stochastic ignition — vectorized random rolls
                candidate_ys, candidate_xs = np.where(spread_candidates)
                rolls = self._rng.random(len(candidate_ys))
                probs = self._prob_map[candidate_ys, candidate_xs]
                ignite = rolls < probs

                ignite_ys = candidate_ys[ignite]
                ignite_xs = candidate_xs[ignite]
                if len(ignite_ys) > 0:
                    self._state[ignite_ys, ignite_xs] = BURNING

        # --- Burnout ---
        burnout_mask = burning & (self._burn_timer >= self._burnout_time)
        self._state[burnout_mask] = BURNED_OUT

        # --- Smoke (update every 2 steps — smoke changes slowly) ---
        if self._step_count % 2 == 0:
            self._update_smoke()
        self._step_count += 1

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
        """Ignite n cells. Random placement on burnable cells.

        Fire acts as a background environmental hazard that adaptive
        planners must detect and route around. Random placement ensures
        fire creates distributed obstacles rather than directly blocking
        the shortest path.
        """
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

        # Diffuse with 3x3 box blur (2 passes) — uniform_filter is faster
        blurred = uniform_filter(source, size=3, mode="constant", cval=0.0)
        blurred = uniform_filter(blurred, size=3, mode="constant", cval=0.0)

        # Persistence (tuned for thinner smoke halos with navigable gaps)
        self._smoke = np.clip(
            0.7 * self._smoke + 0.45 * blurred, 0.0, 1.0
        ).astype(np.float32)
