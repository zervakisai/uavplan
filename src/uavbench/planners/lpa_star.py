"""LPA* inspired incremental replanner (lightweight Python implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class LPAStarConfig(AdaptiveAStarConfig):
    """Config tuned for moderate replan cadence."""
    base_interval: int = 8
    lookahead_steps: int = 6


class LPAStarPlanner(AdaptiveAStarPlanner):
    """LPA*-style planner with persistent incremental replan behavior."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[LPAStarConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or LPAStarConfig())
