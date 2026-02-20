"""Grid-discretized Model Predictive Path Integral (MPPI) planner.

Sampling-based MPC that rolls out N stochastic trajectories,
weights them by exponentiated negative cost, and executes
the best first action.  Falls back to A* on local minima.

IMPORTANT: The weighted-average first displacement is discretized to 4
cardinal directions (right/up/left/down) for compatibility with the
4-connected grid environment.  This collapses the continuous control
output of canonical MPPI into discrete grid moves.

Canonical reference (sampling/weighting IS implemented; continuous control
output is NOT — discretized to 4 cardinal moves):
    Williams, G., Aldrich, A., & Theodorou, E. A. (2017).
    Model Predictive Path Integral Control. ICRA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import BasePlanner, PlannerConfig, PlanResult

GridPos = tuple[int, int]

# Cardinal direction unit vectors: right, up, left, down
_DIRECTIONS = np.array([
    [1, 0],   # right  (+x)
    [0, -1],  # up     (-y)
    [-1, 0],  # left   (-x)
    [0, 1],   # down   (+y)
], dtype=float)


@dataclass
class MPPIConfig(PlannerConfig):
    num_samples: int = 256
    horizon: int = 12
    lambda_temp: float = 1.0
    noise_sigma: float = 0.5
    goal_weight: float = 2.0
    obstacle_penalty: float = 50.0
    risk_weight: float = 1.0
    max_steps: int = 2000


class GridMPPIPlanner(BasePlanner):
    """Grid-discretized MPPI planner (4-cardinal output).

    Implements the MPPI sampling/weighting algorithm (Williams et al. 2017)
    but discretizes the weighted-average displacement to 4 cardinal directions.
    Falls back to A* after repeated stuck steps.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[MPPIConfig] = None,
    ):
        super().__init__(heightmap, no_fly, config or MPPIConfig())
        self.cfg: MPPIConfig = self.cfg  # type: ignore[assignment]

    def plan(self, start: GridPos, goal: GridPos, cost_map=None) -> PlanResult:
        t0 = time.monotonic()
        self._validate_pos(start, "start")
        self._validate_pos(goal, "goal")

        cur = start
        path = [cur]
        visited = {cur}
        expansions = 0
        stuck_count = 0

        rng = np.random.default_rng(42)

        while cur != goal and expansions < self.cfg.max_steps:
            if (time.monotonic() - t0) * 1000 > self.cfg.max_planning_time_ms:
                return PlanResult(
                    path=path, success=False,
                    compute_time_ms=(time.monotonic() - t0) * 1000,
                    expansions=expansions, reason="timeout",
                )

            action = self._mppi_step(cur, goal, cost_map, rng)
            expansions += self.cfg.num_samples

            if action is None or self._is_blocked(action):
                # Try greedy fallback among neighbors
                nbrs = self._neighbors(cur)
                if nbrs:
                    action = min(nbrs, key=lambda p: self._heuristic(p, goal))
                else:
                    break

            if action == cur or action in visited:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count > 5:
                # Fall back to A*
                fallback = AStarPlanner(self.heightmap, self.no_fly).plan(cur, goal, cost_map)
                if fallback.success:
                    path.extend(fallback.path[1:])
                    return PlanResult(
                        path=path, success=True,
                        compute_time_ms=(time.monotonic() - t0) * 1000,
                        expansions=expansions + fallback.expansions,
                        reason="mppi_fallback_astar",
                    )
                break

            cur = action
            path.append(cur)
            visited.add(cur)

        ok = cur == goal
        if not ok:
            fallback = AStarPlanner(self.heightmap, self.no_fly).plan(start, goal, cost_map)
            if fallback.success:
                return PlanResult(
                    path=fallback.path, success=True,
                    compute_time_ms=(time.monotonic() - t0) * 1000,
                    expansions=expansions + fallback.expansions,
                    reason="mppi_fallback_astar",
                )

        return PlanResult(
            path=path if ok else [],
            success=ok,
            compute_time_ms=(time.monotonic() - t0) * 1000,
            expansions=expansions,
            reason="goal found" if ok else "local_minima",
        )

    def _mppi_step(
        self,
        cur: GridPos,
        goal: GridPos,
        cost_map: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> Optional[GridPos]:
        """Sample N trajectories, compute weighted-average first action."""
        N = self.cfg.num_samples
        K = self.cfg.horizon
        sigma = self.cfg.noise_sigma
        lam = self.cfg.lambda_temp

        cx, cy = float(cur[0]), float(cur[1])
        gx, gy = float(goal[0]), float(goal[1])

        # Sample noise for all trajectories: (N, K, 2)
        noise = rng.normal(0.0, sigma, size=(N, K, 2))

        # Mean control: unit vector toward goal
        dx = gx - cx
        dy = gy - cy
        dist = max(abs(dx) + abs(dy), 1e-6)
        mean_u = np.array([dx / dist, dy / dist])

        costs = np.zeros(N)
        first_displacements = np.zeros((N, 2))

        for i in range(N):
            px, py = cx, cy
            traj_cost = 0.0
            for k in range(K):
                ux = mean_u[0] + noise[i, k, 0]
                uy = mean_u[1] + noise[i, k, 1]
                nx_ = px + ux
                ny_ = py + uy
                # Clamp to grid
                nx_ = max(0.0, min(float(self.W - 1), nx_))
                ny_ = max(0.0, min(float(self.H - 1), ny_))

                ix, iy = int(round(nx_)), int(round(ny_))

                # Step cost
                traj_cost += 1.0

                # Obstacle penalty
                if 0 <= ix < self.W and 0 <= iy < self.H:
                    if self.no_fly[iy, ix] or (self.cfg.block_buildings and self.heightmap[iy, ix] > 0):
                        traj_cost += self.cfg.obstacle_penalty
                    # Risk cost
                    if cost_map is not None:
                        traj_cost += self.cfg.risk_weight * float(cost_map[iy, ix])

                px, py = nx_, ny_

                if k == 0:
                    first_displacements[i] = [ux, uy]

            # Goal distance penalty at end of rollout
            goal_dist = abs(px - gx) + abs(py - gy)
            traj_cost += self.cfg.goal_weight * goal_dist
            costs[i] = traj_cost

        # Softmax weights
        costs_shifted = costs - costs.min()
        weights = np.exp(-costs_shifted / max(lam, 1e-6))
        weights /= weights.sum() + 1e-12

        # Weighted first displacement
        weighted_disp = weights @ first_displacements  # shape (2,)

        # Discretize to cardinal direction
        dots = _DIRECTIONS @ weighted_disp
        best_dir = int(np.argmax(dots))
        delta = _DIRECTIONS[best_dir].astype(int)

        new_x = cur[0] + delta[0]
        new_y = cur[1] + delta[1]
        return (int(new_x), int(new_y))


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
MPPIPlanner = GridMPPIPlanner
