from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner, AStarConfig
from uavbench.planners.base import PlanResult


@dataclass
class RRTStarConfig(AStarConfig):
    """Sampling-style config; implemented as anytime A* surrogate baseline."""
    max_iterations: int = 5000


class RRTStarPlanner(AStarPlanner):
    """RRT* baseline proxy.

    For benchmark reproducibility and speed, we use a deterministic grid-search
    surrogate while exposing an RRT*-compatible planner ID.
    """

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[RRTStarConfig] = None):
        cfg = config or RRTStarConfig(allow_diagonal=True)
        super().__init__(heightmap, no_fly, cfg)

    def plan(self, start, goal, cost_map=None) -> PlanResult:
        t0 = time.monotonic()
        res = super().plan(start, goal, cost_map)
        res.reason = res.reason or "rrt_star_surrogate"
        res.compute_time_ms = (time.monotonic() - t0) * 1000
        return res
