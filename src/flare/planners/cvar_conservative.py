"""CVaR / conservative planner — worst-case risk over a ρ-scaled neighborhood.

Robust-planning paradigm: rather than the pointwise risk R(x), the planner routes against
the worst risk within a Chebyshev neighborhood whose radius grows with ρ (a grid proxy for
CVaR / robustness to risk-field spread and localisation error), then applies the soft cost
1 + ρ·R_cvar. Larger ρ ⇒ larger conservatism radius *and* stronger penalty. Replans on
blocking-mask change (reuses AggressiveReplanPlanner).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from flare.planners.aggressive_replan import AggressiveReplanPlanner
from flare.planners.astar import AStarPlanner
from flare.planners.base import PlanResult


class CVaRConservativePlanner(AggressiveReplanPlanner):
    """Worst-case-neighborhood planner. ρ scales both the radius and the penalty."""

    _MAX_RADIUS = 12

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        rho = self._rho if self._rho is not None else 1.0

        weighted = None
        if cost_map is not None:
            r = min(int(round(rho)), self._MAX_RADIUS)
            if r > 0:
                r_cvar = maximum_filter(cost_map, size=2 * r + 1, mode="nearest")
            else:
                r_cvar = cost_map
            weighted = (1.0 + rho * r_cvar).astype(np.float32)

        if self._cached_mask is not None:
            eff = self._heightmap.copy()
            eff[self._cached_mask & ~self._static_mask] = 999.0
            return AStarPlanner(eff, self._no_fly, self._config).search(
                start, goal, cost_map=weighted
            )
        return self._inner.search(start, goal, cost_map=weighted)
