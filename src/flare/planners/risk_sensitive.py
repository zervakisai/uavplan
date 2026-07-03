"""Risk-sensitive (entropic / exponential-utility) planner.

Risk-sensitive optimal-control paradigm (exponential utility; cf. Whittle's LEQG):
the per-cell edge cost is exp(ρ·R(x)) rather than the linear 1 + ρ·R(x). Risky cells are
penalised multiplicatively, and the model is the exponential generalisation of the linear
cost (1 + ρ·R is exactly its first-order Taylor term). ρ=0 ⇒ exp(0)=1 (risk-neutral, uniform
cost); larger ρ ⇒ sharply increasing aversion. The minimum edge cost stays 1 (for R=0), so
the Manhattan heuristic remains admissible. Replans on blocking-mask change.
"""

from __future__ import annotations

import numpy as np

from flare.planners.aggressive_replan import AggressiveReplanPlanner
from flare.planners.astar import AStarPlanner
from flare.planners.base import PlanResult


class RiskSensitivePlanner(AggressiveReplanPlanner):
    """Exponential-utility planner: edge cost exp(ρ·R)."""

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        rho = self._rho if self._rho is not None else 1.0

        weighted = None
        if cost_map is not None:
            weighted = np.exp(rho * cost_map).astype(np.float32)

        if self._cached_mask is not None:
            eff = self._heightmap.copy()
            eff[self._cached_mask & ~self._static_mask] = 999.0
            return AStarPlanner(eff, self._no_fly, self._config).search(
                start, goal, cost_map=weighted
            )
        return self._inner.search(start, goal, cost_map=weighted)
