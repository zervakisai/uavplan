"""MPPI (Model Predictive Path Integral) grid planner.

Sampling-based planner that evaluates random action sequences
and selects the best trajectory via cost-weighted averaging.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from uavbench2.planners.base import PlannerBase, PlanResult


# Actions: 0=up(y-1), 1=down(y+1), 2=left(x-1), 3=right(x+1), 4=stay
_DX = [0, 0, -1, 1, 0]
_DY = [-1, 1, 0, 0, 0]


class MPPIGridPlanner(PlannerBase):
    """MPPI sampling-based planner on 4-connected grid.

    Generates K random action sequences of horizon H, simulates
    them on the grid, and returns the cost-weighted best trajectory.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)
        self._K = 128         # number of samples
        self._horizon = 40    # rollout horizon
        self._temperature = 1.0
        # Deterministic RNG without calling default_rng (DC-1 compliance)
        self._rng = np.random.Generator(np.random.PCG64(0))
        self._goal: tuple[int, int] | None = None

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Plan via MPPI sampling."""
        t0 = time.perf_counter()
        self._goal = goal

        blocked = (self._heightmap > 0) | self._no_fly
        H, W = blocked.shape
        sx, sy = start
        gx, gy = goal

        # Generate K random action sequences of length horizon
        action_seqs = self._rng.integers(0, 4, size=(self._K, self._horizon))

        # Simulate all trajectories
        costs = np.zeros(self._K)
        trajectories: list[list[tuple[int, int]]] = []

        for k in range(self._K):
            traj = [(sx, sy)]
            cx, cy = sx, sy
            collision_cost = 0.0

            for t in range(self._horizon):
                a = action_seqs[k, t]
                nx = cx + _DX[a]
                ny = cy + _DY[a]

                # Bounds check
                if not (0 <= nx < W and 0 <= ny < H):
                    collision_cost += 10.0
                    # Stay in place
                elif blocked[ny, nx]:
                    collision_cost += 10.0
                    # Stay in place
                else:
                    cx, cy = nx, ny

                traj.append((cx, cy))

                # Early termination on goal
                if (cx, cy) == (gx, gy):
                    break

            # Cost = distance to goal + collision penalty
            dist_to_goal = abs(cx - gx) + abs(cy - gy)
            costs[k] = float(dist_to_goal) + collision_cost
            trajectories.append(traj)

        # Cost-weighted selection (MPPI: softmin)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / max(self._temperature, 1e-6))
        weights /= weights.sum() + 1e-12

        # Pick best trajectory
        best_k = int(np.argmin(costs))
        best_traj = trajectories[best_k]

        # If best trajectory doesn't reach goal, extend with greedy walk
        if best_traj[-1] != (gx, gy):
            cx, cy = best_traj[-1]
            for _ in range(H * W):
                if (cx, cy) == (gx, gy):
                    break
                dx = gx - cx
                dy = gy - cy
                # Try primary direction first
                if abs(dx) >= abs(dy):
                    nx = cx + (1 if dx > 0 else -1)
                    ny = cy
                else:
                    nx = cx
                    ny = cy + (1 if dy > 0 else -1)
                if 0 <= nx < W and 0 <= ny < H and not blocked[ny, nx]:
                    cx, cy = nx, ny
                    best_traj.append((cx, cy))
                else:
                    # Try other direction
                    if abs(dx) >= abs(dy):
                        nx = cx
                        ny = cy + (1 if dy > 0 else -1) if dy != 0 else cy
                    else:
                        nx = cx + (1 if dx > 0 else -1) if dx != 0 else cx
                        ny = cy
                    if 0 <= nx < W and 0 <= ny < H and not blocked[ny, nx]:
                        cx, cy = nx, ny
                        best_traj.append((cx, cy))
                    else:
                        break

        success = best_traj[-1] == (gx, gy)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return PlanResult(
            path=best_traj,
            success=success,
            compute_time_ms=elapsed,
            expansions=self._K * self._horizon,
            reason="" if success else "mppi_no_convergence",
        )

    def update(self, dyn_state: dict[str, Any]) -> None:
        """Accept dynamic state (no-op for MPPI)."""
        pass

    def should_replan(
        self,
        current_pos: tuple[int, int],
        current_path: list[tuple[int, int]],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        """MPPI replans every step by design (sampling-based)."""
        return (False, "mppi_no_replan")
