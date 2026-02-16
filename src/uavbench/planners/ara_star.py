"""ARA* inspired anytime-repairing replanner (lightweight Python implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class ARAStarConfig(AdaptiveAStarConfig):
    """Anytime-biased settings with slightly longer baseline intervals."""
    base_interval: int = 7
    lookahead_steps: int = 7


class ARAStarPlanner(AdaptiveAStarPlanner):
    """ARA*-style planner using repairable path updates over time."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[ARAStarConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or ARAStarConfig())
