"""Artificial Potential Field (APF) planner — reactive gradient-following.

Fundamentally different paradigm from graph search: no explicit search,
no tree, no heuristic expansion. Computes an attractive potential toward
the goal and repulsive potentials away from obstacles, then greedily
follows the negative gradient on a 4-connected grid.

Known weakness: local minima when fire/obstacles create U-shaped traps.
This is expected behavior that validates benchmark difficulty.

Risk-modulated repulsion: k_rep *= (1 + δ * risk) where δ=3.0.

Uses scipy.ndimage.distance_transform_cdt for O(H*W) precomputation of
nearest-obstacle distances, making per-cell potential evaluation O(1).

References:
    Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile
    Robots", International Journal of Robotics Research, 1986.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_cdt

from uavbench.blocking import compute_blocking_mask
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import PlannerBase, PlanResult

# 4-connected grid actions: (dx, dy)
_ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
_N_ACTIONS = len(_ACTIONS)

# Risk coefficient: risk modulates repulsive potential
_RISK_DELTA = 3.0


class APFPlanner(PlannerBase):
    """Artificial Potential Field planner on 4-connected grid.

    plan() computes a full path via gradient descent in one shot.
    should_replan() triggers only when the blocking mask changes along
    the current path (same pattern as AggressiveReplan, for fairness).
    Falls back to A* when stuck in a local minimum.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)
        self._k_att = 1.0       # Attractive gain
        self._k_rep = 50.0      # Repulsive gain
        self._d0 = 8            # Repulsive influence radius (cells)
        self._dyn_state: dict[str, Any] | None = None
        self._cached_mask: np.ndarray | None = None
        # Replan tracking (same pattern as aggressive_replan)
        self._last_mask_hash: str = ""
        self._last_replan_step = -3
        self._last_replan_pos: tuple[int, int] | None = None
        self._cooldown = 3
        # Wind-aware momentum (Upgrade 12)
        self._wind_speed = getattr(config, "wind_speed", 0.0) if config else 0.0
        self._wind_dir = math.radians(getattr(config, "wind_direction_deg", 0.0)) if config else 0.0
        self._momentum_alpha = 0.7  # blending: 0.7*prev + 0.3*gradient

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Generate a path by greedy gradient descent on the potential field.

        One distance_transform precomputation, then O(path_len) rollout.
        Risk-modulated: k_rep *= (1 + δ * risk) where δ=3.0.
        Falls back to A* if stuck in a local minimum.
        """
        t0 = time.perf_counter()

        blocked = self._get_blocked()
        sx, sy = start
        gx, gy = goal

        # Precompute distance-to-nearest-obstacle (O(H*W), single pass)
        free = ~blocked
        dist_to_obs = distance_transform_cdt(free, metric="taxicab").astype(np.float32)

        max_steps = 2 * (self._H + self._W)
        path = [(sx, sy)]
        cx, cy = sx, sy
        visited_count: dict[tuple[int, int], int] = {(sx, sy): 1}

        prev_dx, prev_dy = 0, 0  # momentum direction

        for _ in range(max_steps):
            if (cx, cy) == (gx, gy):
                break

            best_action = -1
            best_potential = float("inf")

            for a in range(_N_ACTIONS):
                adx, ady = _ACTIONS[a]
                nx, ny = cx + adx, cy + ady

                if not (0 <= nx < self._W and 0 <= ny < self._H):
                    continue
                if blocked[ny, nx]:
                    continue

                # Attractive: quadratic Euclidean distance (Khatib 1986)
                u_att = 0.5 * self._k_att * ((nx - gx) ** 2 + (ny - gy) ** 2)

                # Repulsive: from precomputed distance transform
                # Risk-modulated: k_rep scaled by local risk (δ=3.0)
                d = dist_to_obs[ny, nx]
                u_rep = 0.0
                if 0 < d < self._d0:
                    local_risk = float(cost_map[ny, nx]) if cost_map is not None else 0.0
                    k_rep_eff = self._k_rep * (1.0 + _RISK_DELTA * local_risk)
                    u_rep = k_rep_eff * ((1.0 / d) - (1.0 / self._d0)) ** 2
                    # Wind bias: favor moving upwind (safer from fire spread)
                    if self._wind_speed > 0:
                        cell_angle = math.atan2(ny - cy, nx - cx)
                        # Upwind = opposite to wind direction
                        upwind_align = math.cos(cell_angle - self._wind_dir + math.pi)
                        u_rep += self._wind_speed * 2.0 * max(0.0, -upwind_align)

                pot = u_att + u_rep
                # Penalize revisits to escape local minima
                pot += visited_count.get((nx, ny), 0) * 20.0
                # Momentum: penalize direction changes (smooth trajectory)
                if self._wind_speed > 0 and (prev_dx != 0 or prev_dy != 0):
                    # Dot product with previous direction (1=same, -1=reverse)
                    dot = adx * prev_dx + ady * prev_dy
                    pot += (1.0 - dot) * 5.0  # penalize direction changes

                if pot < best_potential:
                    best_potential = pot
                    best_action = a

            if best_action < 0:
                break  # All neighbors blocked

            adx, ady = _ACTIONS[best_action]
            cx, cy = cx + adx, cy + ady
            path.append((cx, cy))
            visited_count[(cx, cy)] = visited_count.get((cx, cy), 0) + 1
            prev_dx, prev_dy = _ACTIONS[best_action]

        elapsed = (time.perf_counter() - t0) * 1000.0

        if (cx, cy) == (gx, gy):
            return PlanResult(
                path=path,
                success=True,
                compute_time_ms=elapsed,
                expansions=len(path),
            )

        # APF stuck — fall back to A*
        fb = self._fallback_plan(start, goal, blocked)
        if fb is not None:
            return PlanResult(
                path=fb,
                success=True,
                compute_time_ms=elapsed,
                expansions=len(path),
                reason="apf_fallback_astar",
            )

        return PlanResult(
            path=path,
            success=False,
            compute_time_ms=elapsed,
            expansions=len(path),
            reason="apf_local_minimum",
        )

    def update(self, dyn_state: dict[str, Any]) -> None:
        """Store dynamic state; mask cached lazily for plan()."""
        self._dyn_state = dyn_state
        self._cached_mask = None  # invalidate; recompute in should_replan()

    def should_replan(
        self,
        current_pos: tuple[int, int],
        current_path: list[tuple[int, int]],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        """Replan when blocking mask changes along path (like aggressive)."""
        if step - self._last_replan_step < self._cooldown:
            return (False, "cooldown")

        # Compute and cache mask (reused by plan() if replan triggers)
        self._cached_mask = self._get_blocked_from(dyn_state)
        mask_hash = hashlib.sha256(self._cached_mask.tobytes()).hexdigest()

        if self._last_mask_hash == "":
            self._last_mask_hash = mask_hash
            self._last_replan_pos = current_pos
            return (False, "calibration")

        if mask_hash == self._last_mask_hash:
            return (False, "no_change")

        # Check if path is actually blocked
        path_blocked = False
        for px, py in current_path:
            if 0 <= py < self._cached_mask.shape[0] and 0 <= px < self._cached_mask.shape[1]:
                if self._cached_mask[py, px]:
                    path_blocked = True
                    break

        if not path_blocked:
            self._last_mask_hash = mask_hash
            return (False, "path_clear")

        self._last_replan_step = step
        self._last_replan_pos = current_pos
        self._last_mask_hash = mask_hash
        return (True, "obstacle_changed")

    # -- Internal --

    def _get_blocked(self) -> np.ndarray:
        """Build current blocking mask (uses cache from update())."""
        if self._cached_mask is not None:
            return self._cached_mask
        if self._dyn_state is not None and self._config is not None:
            return compute_blocking_mask(
                self._heightmap, self._no_fly,
                self._config, self._dyn_state,
            )
        return (self._heightmap > 0) | self._no_fly

    def _get_blocked_from(self, dyn_state: dict[str, Any]) -> np.ndarray:
        """Build blocking mask from explicit dyn_state."""
        if self._config is not None:
            return compute_blocking_mask(
                self._heightmap, self._no_fly,
                self._config, dyn_state,
            )
        return (self._heightmap > 0) | self._no_fly

    def _fallback_plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: np.ndarray,
    ) -> list[tuple[int, int]] | None:
        """A* fallback when APF is stuck in local minimum.

        Passes full blocked mask as no_fly to A*, so it sees all
        dynamic obstacles. Uses compute_blocking_mask() output directly
        instead of modifying heightmap (MP-1 compliance).
        """
        fb = AStarPlanner(self._heightmap, blocked, self._config)
        result = fb.plan(start, goal)
        return result.path if result.success else None
