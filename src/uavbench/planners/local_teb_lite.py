"""TEB-lite local trajectory optimizer (grid approximation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class LocalTEBLiteConfig(AdaptiveAStarConfig):
    """Smoother but still reactive local replanning behavior."""
    base_interval: int = 4
    lookahead_steps: int = 6
    smoke_threshold: float = 0.2


class LocalTEBLitePlanner(AdaptiveAStarPlanner):
    """Python-only elastic-band inspired local replanner."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[LocalTEBLiteConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or LocalTEBLiteConfig())
