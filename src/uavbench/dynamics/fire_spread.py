"""Cellular automaton fire spread model.

Fire ignites at configurable points and spreads through the grid based on
landuse type, wind direction, and road firebreaks. All operations are
numpy-only for runtime performance.
"""

from __future__ import annotations

import numpy as np

# Landuse-dependent spread probabilities (per neighbor, per second)
SPREAD_PROB: dict[int, float] = {
    0: 0.02,   # empty / unknown
    1: 0.30,   # forest (most flammable)
    2: 0.10,   # urban / residential
    3: 0.05,   # industrial
    4: 0.00,   # water (immune)
}

# Cell states
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2

# 4-connected neighbor offsets: (dy, dx)
_NEIGHBORS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

# Wind direction vectors for each neighbor (N, S, W, E as angles in radians)
# 0=North means wind blows from north to south, so fire spreads southward
_NEIGHBOR_ANGLES = np.array([
    np.pi,     # neighbor to the north — fire spreads north (against south wind)
    0.0,       # neighbor to the south — fire spreads south (with north wind)
    np.pi / 2, # neighbor to the west
    3 * np.pi / 2,  # neighbor to the east
])


class FireSpreadModel:
    """Cellular automaton wildfire model.

    Args:
        landuse_map: int8 [H, W] — landuse categories (0–4)
        roads_mask: bool [H, W] — road cells (act as firebreaks)
        wind_dir: float — wind direction in radians (0 = North, π/2 = East)
        wind_speed: float — wind intensity in [0, 1]
        rng: np.random.Generator — for deterministic ignition + spread
        n_ignition: int — number of initial fire ignition points
    """

    def __init__(
        self,
        landuse_map: np.ndarray,
        roads_mask: np.ndarray,
        wind_dir: float,
        wind_speed: float,
        rng: np.random.Generator,
        n_ignition: int = 3,
    ) -> None:
        H, W = landuse_map.shape
        self._shape = (H, W)
        self._landuse = landuse_map.astype(np.int8)
        self._roads = roads_mask.astype(bool)
        self._wind_dir = float(wind_dir)
        self._wind_speed = float(np.clip(wind_speed, 0.0, 1.0))
        self._rng = rng

        # State arrays
        self._state = np.zeros((H, W), dtype=np.int8)  # UNBURNED
        self._burn_timer = np.zeros((H, W), dtype=np.float32)
        # Random burnout duration per cell (30–60 seconds)
        self._burnout_time = rng.uniform(30.0, 60.0, size=(H, W)).astype(np.float32)

        # Precompute base spread probability per cell
        self._base_prob = np.zeros((H, W), dtype=np.float32)
        for lu_val, prob in SPREAD_PROB.items():
            self._base_prob[self._landuse == lu_val] = prob
        # Roads reduce probability by 50%
        self._base_prob[self._roads] *= 0.5

        # Precompute wind modifier per neighbor direction
        # Wind boosts spread in the downwind direction by up to 1.5×
        angle_diff = np.abs(_NEIGHBOR_ANGLES - self._wind_dir)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        # cos(0)=1 when aligned, cos(π)=-1 when opposed
        self._wind_modifier = 1.0 + 0.5 * self._wind_speed * np.cos(angle_diff)

        # Ignite
        self._ignite(n_ignition)

    def _ignite(self, n: int) -> None:
        """Place initial fire at n random forest cells (fallback: any burnable)."""
        # Prefer forest cells
        forest_mask = (self._landuse == 1) & (self._base_prob > 0)
        candidates = np.argwhere(forest_mask)

        if len(candidates) < n:
            # Fallback: any burnable cell
            burnable = self._base_prob > 0
            candidates = np.argwhere(burnable)

        if len(candidates) == 0:
            return  # nothing to burn

        n = min(n, len(candidates))
        chosen = self._rng.choice(len(candidates), size=n, replace=False)
        for idx in chosen:
            y, x = candidates[idx]
            self._state[y, x] = BURNING

    def step(self, dt: float = 1.0) -> None:
        """Advance fire by dt seconds."""
        H, W = self._shape
        state = self._state

        # Find currently burning cells
        burning = np.argwhere(state == BURNING)
        if len(burning) == 0:
            return

        # Update burn timers
        self._burn_timer[state == BURNING] += dt

        # Check burnout
        burnout_mask = (state == BURNING) & (self._burn_timer >= self._burnout_time)
        state[burnout_mask] = BURNED_OUT

        # Spread fire to neighbors
        new_fires_y = []
        new_fires_x = []

        for dir_idx, (dy, dx) in enumerate(_NEIGHBORS):
            # Shift burning cell coordinates by neighbor offset
            ny = burning[:, 0] + dy
            nx = burning[:, 1] + dx

            # Bounds check
            valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            ny = ny[valid]
            nx = nx[valid]

            if len(ny) == 0:
                continue

            # Only spread to unburned cells
            unburned = state[ny, nx] == UNBURNED
            ny = ny[unburned]
            nx = nx[unburned]

            if len(ny) == 0:
                continue

            # Compute spread probability with wind modifier
            prob = self._base_prob[ny, nx] * self._wind_modifier[dir_idx] * dt
            prob = np.clip(prob, 0.0, 1.0)

            # Roll dice
            rolls = self._rng.random(len(ny))
            ignited = rolls < prob
            new_fires_y.append(ny[ignited])
            new_fires_x.append(nx[ignited])

        # Apply new fires
        if new_fires_y:
            all_y = np.concatenate(new_fires_y)
            all_x = np.concatenate(new_fires_x)
            state[all_y, all_x] = BURNING

    @property
    def fire_mask(self) -> np.ndarray:
        """[H, W] bool — cells currently on fire."""
        return self._state == BURNING

    @property
    def burned_mask(self) -> np.ndarray:
        """[H, W] bool — cells that have burned out."""
        return self._state == BURNED_OUT

    @property
    def total_affected(self) -> int:
        """Number of cells burning or burned out."""
        return int((self._state > UNBURNED).sum())
