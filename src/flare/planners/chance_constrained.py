"""Chance-constrained planner — risk as a hard tolerance threshold.

Distinct paradigm from the soft cost-inflation planners: ρ sets a risk *tolerance*
τ = 1/(1+ρ); cells whose continuous risk R(x) exceeds τ are forbidden outright
(chance-constrained / safe-planning). As ρ grows the tolerance tightens and the feasible
corridor narrows, forcing longer detours — or, if over-constrained, a graceful fallback
to the hazard-only grid. Replans on blocking-mask change (reuses AggressiveReplanPlanner).
"""

from __future__ import annotations

import numpy as np

from flare.planners.aggressive_replan import AggressiveReplanPlanner
from flare.planners.astar import AStarPlanner
from flare.planners.base import PlanResult


class ChanceConstrainedPlanner(AggressiveReplanPlanner):
    """Hard risk-threshold planner. ρ = risk tolerance (τ = 1/(1+ρ))."""

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        rho = self._rho if self._rho is not None else 1.0

        effective_height = self._heightmap.copy()
        if self._cached_mask is not None:
            effective_height[self._cached_mask & ~self._static_mask] = 999.0

        if cost_map is not None and rho > 0:
            tau = 1.0 / (1.0 + rho)
            risky = cost_map > tau
            sx, sy = start
            gx, gy = goal
            # Never forbid the current start or the goal cell.
            risky[sy, sx] = False
            risky[gy, gx] = False
            constrained = effective_height.copy()
            constrained[risky] = 999.0
            res = AStarPlanner(constrained, self._no_fly, self._config).search(
                start, goal, cost_map=None
            )
            if res.success:
                return res
            # Over-constrained → relax the risk threshold, keep hard obstacles.

        return AStarPlanner(effective_height, self._no_fly, self._config).search(
            start, goal, cost_map=None
        )
