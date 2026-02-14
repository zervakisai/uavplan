from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner, AStarConfig
from uavbench.planners.base import PlanResult


@dataclass
class NSGA2Config(AStarConfig):
    risk_weight: float = 1.0


class NSGAIIPlanner(AStarPlanner):
    """NSGA-II inspired multi-objective baseline via weighted risk/path trade-off."""

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[NSGA2Config] = None):
        super().__init__(heightmap, no_fly, config or NSGA2Config())
        self.cfg: NSGA2Config = self.cfg

    def plan(self, start, goal, cost_map=None) -> PlanResult:
        if cost_map is None:
            return super().plan(start, goal, cost_map)
        weighted = 1.0 + self.cfg.risk_weight * np.asarray(cost_map, dtype=float)
        return super().plan(start, goal, weighted)
