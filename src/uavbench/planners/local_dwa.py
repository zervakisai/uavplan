"""DWA-lite local reactive planner utilities (grid adaptation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class LocalDWAConfig(AdaptiveAStarConfig):
    """Very reactive behavior suited for moving-obstacle avoidance."""
    base_interval: int = 3
    lookahead_steps: int = 4
    traffic_buffer: int = 6


class LocalDWAPlanner(AdaptiveAStarPlanner):
    """Grid-safe approximation of a DWA-like local reactive module."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[LocalDWAConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or LocalDWAConfig())
