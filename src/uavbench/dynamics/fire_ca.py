"""Fire Cellular Automaton model (DC-1, FD-1 through FD-5b).

Fire spread on a 2D grid. 8-neighbor Moore neighborhood.
State per cell: UNBURNED(0) → BURNING(1) → BURNED_OUT(2).

FD-5b: Wind is OPTIONAL. wind_speed=0 → isotropic (backward compat).
Wind modulation follows Alexandridis et al. 2008, Eq. 4.
"""

from __future__ import annotations

import math

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

# 8-connected Moore neighborhood (FD-2)
# (dy, dx) offsets in array coords; world direction computed via atan2(-dy, dx)
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Alexandridis et al. 2008 wind constants (Eq. 4)
_WIND_C1 = 0.045
_WIND_C2 = 0.131

# Pre-computed Moore dilation structure
_MOORE_STRUCT = np.ones((3, 3), dtype=bool)


class FireSpreadModel:
    """Cellular automaton fire spread model.

    FD-5b: Wind is OPTIONAL. wind_speed=0 → isotropic (backward compat).
    When wind_speed>0, spread probability is directionally modulated per
    Alexandridis et al. 2008. All randomness via caller-supplied rng (DC-1).
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
        n_ignition: int = 3,
        landuse_map: np.ndarray | None = None,
        roads_mask: np.ndarray | None = None,
        corridor_cells: list[tuple[int, int]] | None = None,
        guarantee_targets: list[tuple[int, int]] | None = None,
        guarantee_step: int | None = None,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0,
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
        self._fire_events: list[dict] = []

        # Fire corridor guarantee: corridor cells that must be burning
        # by guarantee_step. Approach ignitions try natural spread;
        # safety net force-ignites any missed targets.
        self._guarantee_targets = guarantee_targets or []
        self._guarantee_step = guarantee_step

        # Guarantee targets use extended burnout — effectively infinite so
        # they never transition to BURNED_OUT within episode duration.
        # This keeps the CA state machine consistent (no re-ignition hack).
        for gx, gy in self._guarantee_targets:
            if 0 <= gy < self._H and 0 <= gx < self._W:
                self._burnout_time[gy, gx] = 999999.0

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

        # Wind modulation (FD-5b, WD-1: deterministic given same params)
        self._wind_speed = wind_speed
        self._wind_factors = np.ones(8, dtype=np.float32)
        if wind_speed > 0.0:
            for i, (dy, dx) in enumerate(_NEIGHBORS):
                # _NEIGHBORS[i] = (dy, dx) is the offset to the BURNING neighbor.
                # Spread direction is OPPOSITE: from burning TO candidate = (-dy, -dx).
                # World coords: x=-dx (East), y=dy (North, array row inverted).
                spread_angle = math.atan2(dy, -dx)
                theta = spread_angle - wind_direction
                self._wind_factors[i] = math.exp(
                    wind_speed * (_WIND_C1 + _WIND_C2 * (math.cos(theta) - 1.0))
                )

        # Initial ignition — random placement for environmental hazards
        if n_ignition > 0:
            self._ignite(n_ignition, corridor_cells)

        # Approach ignitions: place fire seeds near each guarantee target
        # so natural spread reaches the corridor by guarantee_step.
        if self._guarantee_targets and corridor_cells:
            self._ignite_approach_fires(corridor_cells)

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

    def pop_events(self) -> list[dict]:
        """Return and clear pending fire events.

        Events emitted when fire reaches building cells (landuse 2/3).
        Used by TriageMission for dynamic casualty injection.
        """
        events = self._fire_events
        self._fire_events = []
        return events

    # -- Step --

    def step(self, dt: float = 1.0) -> None:
        """Advance fire by one timestep (FD-1, FD-4).

        1. Spread: burning cells attempt to ignite neighbors (isotropic).
        2. Guarantee: force-ignite corridor targets at guarantee_step.
        3. Burnout: cells that have burned long enough become BURNED_OUT.
           Guarantee targets use extended burnout (set in __init__) so they
           never burn out within episode duration — no re-ignition needed.
        4. Smoke: update smoke field (no wind advection).
        """
        burning = self._state == BURNING

        # Increment burn timers for burning cells
        self._burn_timer[burning] += dt

        # --- Spread (FD-2 / FD-5b) ---
        if burning.any():
            if self._wind_speed == 0.0:
                # Isotropic: original binary_dilation (backward compat, bit-identical)
                spread_candidates = (
                    binary_dilation(burning, structure=_MOORE_STRUCT)
                    & (self._state == UNBURNED)
                )
                if spread_candidates.any():
                    candidate_ys, candidate_xs = np.where(spread_candidates)
                    rolls = self._rng.random(len(candidate_ys))
                    probs = self._prob_map[candidate_ys, candidate_xs]
                    ignite = rolls < probs
                    ignite_ys = candidate_ys[ignite]
                    ignite_xs = candidate_xs[ignite]
                    if len(ignite_ys) > 0:
                        self._state[ignite_ys, ignite_xs] = BURNING
            else:
                # Wind-modulated: per-direction spread (Alexandridis 2008)
                unburned = self._state == UNBURNED
                for i, (dy, dx) in enumerate(_NEIGHBORS):
                    # Shift burning mask: find cells with a burning neighbor
                    # in direction (dy, dx). np.roll wraps; fix boundaries.
                    shifted = np.roll(np.roll(burning, -dy, axis=0), -dx, axis=1)
                    if dy < 0:
                        shifted[dy:, :] = False
                    elif dy > 0:
                        shifted[:dy, :] = False
                    if dx < 0:
                        shifted[:, dx:] = False
                    elif dx > 0:
                        shifted[:, :dx] = False
                    candidates = shifted & unburned
                    if not candidates.any():
                        continue
                    ys, xs = np.where(candidates)
                    rolls = self._rng.random(len(ys))
                    probs = self._prob_map[ys, xs] * self._wind_factors[i]
                    ignite_mask = rolls < probs
                    if ignite_mask.any():
                        self._state[ys[ignite_mask], xs[ignite_mask]] = BURNING
                        # Update unburned to avoid double-counting
                        unburned = self._state == UNBURNED

        # --- Detect building fires (Upgrade 11: fire-spawned tasks) ---
        newly_burning = (self._state == BURNING) & ~burning  # cells that just became BURNING
        if newly_burning.any():
            building_fire = newly_burning & np.isin(self._landuse, [2, 3])
            if building_fire.any():
                bfy, bfx = np.where(building_fire)
                for y, x in zip(bfy, bfx):
                    self._fire_events.append({
                        "type": "building_fire",
                        "x": int(x),
                        "y": int(y),
                        "step": self._step_count,
                    })

        # --- Guarantee safety net: force-ignite corridor targets ---
        # Targets that haven't ignited naturally by guarantee_step are
        # force-ignited here. Extended burnout (999999) set in __init__
        # ensures they stay BURNING for the entire episode.
        if (self._guarantee_targets
                and self._guarantee_step is not None
                and self._step_count >= self._guarantee_step):
            for gx, gy in self._guarantee_targets:
                if (0 <= gy < self._H and 0 <= gx < self._W
                        and self._state[gy, gx] == UNBURNED):
                    self._state[gy, gx] = BURNING

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
        """Ignite n cells at random locations.

        Random placement creates distributed environmental hazards without
        overwhelming chokepoints on dense OSM maps. Corridor interdiction
        is handled by fire corridor guarantee (approach ignitions +
        force-ignite safety net).
        """
        self._ignite_random(n)

    def _ignite_approach_fires(
        self,
        corridor_cells: list[tuple[int, int]],
    ) -> None:
        """Place approach ignitions near each guarantee target.

        For each guarantee target on the corridor, ignite one cell at
        offset [8, 15] Manhattan distance. Natural spread from these
        seeds should reach the corridor by guarantee_step; the safety
        net in step() handles failures.
        """
        for gx, gy in self._guarantee_targets:
            candidates_y = []
            candidates_x = []
            r_min, r_max = 8, 15
            for dy in range(-r_max, r_max + 1):
                for dx in range(-r_max, r_max + 1):
                    dist = abs(dy) + abs(dx)
                    if dist < r_min or dist > r_max:
                        continue
                    cy, cx = gy + dy, gx + dx
                    if not (0 <= cy < self._H and 0 <= cx < self._W):
                        continue
                    if self._state[cy, cx] != UNBURNED:
                        continue
                    if self._landuse[cy, cx] == 4:  # water — never burns
                        continue
                    candidates_y.append(cy)
                    candidates_x.append(cx)

            if candidates_y:
                idx = self._rng.integers(len(candidates_y))
                self._state[candidates_y[idx], candidates_x[idx]] = BURNING

    def _ignite_near_corridor(
        self,
        n: int,
        corridor_cells: list[tuple[int, int]],
    ) -> None:
        """Ignite some cells near corridor, rest randomly (CC-2 calibrated).

        Only half the ignitions (at least 1) are placed near the corridor at
        offset [15, 25] Manhattan cells. The rest are placed randomly to avoid
        overwhelming chokepoints. The larger offset (vs original [8,15]) gives
        fire time to expand visibly while preserving alternative paths.
        """
        interior = corridor_cells[1:-1]  # skip start/goal
        if len(interior) == 0:
            self._ignite_random(n)
            return

        # CC-2 calibration: only half near corridor, rest random
        n_corridor = max(1, n // 2)
        n_random = n - n_corridor

        step = max(1, len(interior) // n_corridor)
        placed = 0

        for i in range(n_corridor):
            anchor_idx = min(i * step, len(interior) - 1)
            ax, ay = interior[anchor_idx]

            # Find burnable cells in ring [15, 25] from anchor
            candidates_y = []
            candidates_x = []
            r_min, r_max = 15, 25
            for dy in range(-r_max, r_max + 1):
                for dx in range(-r_max, r_max + 1):
                    dist = abs(dy) + abs(dx)
                    if dist < r_min or dist > r_max:
                        continue
                    cy, cx = ay + dy, ax + dx
                    if not (0 <= cy < self._H and 0 <= cx < self._W):
                        continue
                    if self._state[cy, cx] != UNBURNED:
                        continue
                    if self._landuse[cy, cx] == 4:  # water — never burns
                        continue
                    candidates_y.append(cy)
                    candidates_x.append(cx)

            if candidates_y:
                idx = self._rng.integers(len(candidates_y))
                self._state[candidates_y[idx], candidates_x[idx]] = BURNING
                placed += 1

        # Fill remaining corridor slots + random slots
        remaining = n - placed
        if remaining > 0:
            self._ignite_random(remaining)

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
