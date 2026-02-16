"""AD* inspired anytime-dynamic replanner (lightweight Python implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class ADStarConfig(AdaptiveAStarConfig):
    """Aggressive replan cadence with short lookahead (dynamic-first)."""
    base_interval: int = 4
    lookahead_steps: int = 10
    fire_close_threshold: int = 10


class ADStarPlanner(AdaptiveAStarPlanner):
    """AD*-style anytime dynamic planner backed by incremental A* updates."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[ADStarConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or ADStarConfig())
