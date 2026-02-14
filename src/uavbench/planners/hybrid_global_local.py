from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner, AStarConfig
from uavbench.planners.base import PlanResult
from uavbench.planners.dwa import DWAPlanner


@dataclass
class HybridGlobalLocalConfig(AStarConfig):
    local_window: int = 12


class HybridGlobalLocalPlanner(AStarPlanner):
    """Hybrid global-local planner (A* global + DWA-style local refinement)."""

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[HybridGlobalLocalConfig] = None):
        super().__init__(heightmap, no_fly, config or HybridGlobalLocalConfig())
        self.cfg: HybridGlobalLocalConfig = self.cfg

    def plan(self, start, goal, cost_map=None) -> PlanResult:
        t0 = time.monotonic()
        global_res = super().plan(start, goal, cost_map)
        if not global_res.success:
            return global_res

        local = DWAPlanner(self.heightmap, self.no_fly)
        local_goal = global_res.path[min(len(global_res.path)-1, self.cfg.local_window)]
        local_res = local.plan(start, local_goal, cost_map)

        if local_res.success and local_res.path:
            merged = local_res.path[:-1] + global_res.path[min(len(global_res.path)-1, self.cfg.local_window):]
        else:
            merged = global_res.path

        return PlanResult(path=merged, success=True,
                          compute_time_ms=(time.monotonic()-t0)*1000,
                          expansions=global_res.expansions + local_res.expansions,
                          reason="hybrid_global_local")
