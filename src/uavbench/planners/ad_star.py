"""Periodic-replan A* planner (aggressive cadence variant).

Simplified replanning planner that re-runs A* from scratch at aggressive
intervals (base_interval=4, lookahead=10).  Named after the concept of
anytime dynamic replanning (cf. AD*, Likhachev et al. 2005) but does NOT
implement canonical AD* data structures (epsilon-suboptimality bounds,
anytime refinement, backward search, rhs values).  This is a periodic
full-replan strategy with tighter cadence than PeriodicReplanPlanner.

Canonical reference (NOT implemented here):
    Likhachev, M., Ferguson, D., Gordon, G., Stentz, A., & Thrun, S. (2005).
    Anytime Dynamic A*. NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class AggressiveReplanConfig(AdaptiveAStarConfig):
    """Aggressive replan cadence with extended lookahead (dynamic-first)."""
    base_interval: int = 4
    lookahead_steps: int = 10
    fire_close_threshold: int = 10


# Keep old name as alias for backward compatibility
ADStarConfig = AggressiveReplanConfig


class AggressiveReplanPlanner(AdaptiveAStarPlanner):
    """Periodic full-replan A* with aggressive cadence (base_interval=4).

    This planner re-runs standard A* from the current position at tight
    intervals and with extended lookahead.  It does NOT implement canonical
    AD* (no epsilon-suboptimality, no anytime refinement, no backward search).
    Canonical ref: Likhachev et al. (2005), Anytime Dynamic A*, NIPS.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[AggressiveReplanConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or AggressiveReplanConfig())


# Backward-compatible alias
ADStarPlanner = AggressiveReplanPlanner
