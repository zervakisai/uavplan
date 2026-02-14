from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.base import PlanResult
from uavbench.planners.dstar_lite import DStarLitePlanner, DStarLiteConfig


@dataclass
class RRTXConfig(DStarLiteConfig):
    """Dynamic sampling-repair baseline config."""


class RRTXPlanner(DStarLitePlanner):
    """RRTX-style incremental replanning baseline (deterministic surrogate)."""

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[RRTXConfig] = None):
        super().__init__(heightmap, no_fly, config or RRTXConfig())

    def plan(self, start, goal, cost_map=None) -> PlanResult:
        res = super().plan(start, goal, cost_map)
        if not res.reason:
            res.reason = "rrtx_plan"
        return res
