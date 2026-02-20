"""Periodic-replan A* planner (relaxed cadence variant).

Simplified replanning planner that re-runs A* from scratch at adaptive
intervals.  Named after the concept of incremental replanning (cf. D* Lite,
Koenig & Likhachev 2002) but does NOT implement canonical D* Lite data
structures (rhs values, backward search, key-ordered priority queue,
UpdateVertex).  This is a periodic full-replan strategy.

Canonical reference (NOT implemented here):
    Koenig, S. & Likhachev, M. (2002). D* Lite. AAAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class PeriodicReplanConfig(AdaptiveAStarConfig):
    """Config tuned for relaxed-cadence periodic replanning."""
    base_interval: int = 6
    lookahead_steps: int = 8


# Keep old name as alias for backward compatibility
DStarLiteConfig = PeriodicReplanConfig


class PeriodicReplanPlanner(AdaptiveAStarPlanner):
    """Periodic full-replan A* with relaxed cadence (base_interval=6).

    This planner re-runs standard A* from the current position at adaptive
    intervals and when path blockages are detected.  It does NOT implement
    canonical D* Lite (no rhs values, no backward search, no incremental
    updates).  Canonical ref: Koenig & Likhachev (2002), D* Lite, AAAI.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[PeriodicReplanConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or PeriodicReplanConfig())


# Backward-compatible alias
DStarLitePlanner = PeriodicReplanPlanner
