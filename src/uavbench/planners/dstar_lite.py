"""D* Lite inspired incremental replanner (lightweight Python implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class DStarLiteConfig(AdaptiveAStarConfig):
    """Config tuned for frequent incremental replans."""
    base_interval: int = 6
    lookahead_steps: int = 8


class DStarLitePlanner(AdaptiveAStarPlanner):
    """Pragmatic D* Lite-style planner built on incremental A* replanning."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[DStarLiteConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or DStarLiteConfig())
