from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner, AStarConfig
from uavbench.planners.base import PlanResult


@dataclass
class MPCConfig(AStarConfig):
    horizon: int = 8
    max_steps: int = 512


class MPCPlanner(AStarPlanner):
    """Receding-horizon MPC-style planner using repeated short-horizon rollout."""

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[MPCConfig] = None):
        super().__init__(heightmap, no_fly, config or MPCConfig())
        self.cfg: MPCConfig = self.cfg

    def plan(self, start, goal, cost_map=None) -> PlanResult:
        t0 = time.monotonic()
        cur = start
        path = [cur]
        expansions = 0

        for _ in range(self.cfg.max_steps):
            if cur == goal:
                break
            if (time.monotonic() - t0) * 1000 > self.cfg.max_planning_time_ms:
                return PlanResult([], False, (time.monotonic()-t0)*1000, expansions, reason="timeout")

            sub = super().plan(cur, goal, cost_map)
            expansions += sub.expansions
            if not sub.success or len(sub.path) < 2:
                return PlanResult([], False, (time.monotonic()-t0)*1000, expansions, reason="no feasible control")

            step_count = min(self.cfg.horizon, len(sub.path) - 1)
            for k in range(1, step_count + 1):
                path.append(sub.path[k])
                cur = sub.path[k]
                if cur == goal:
                    break

        ok = cur == goal
        return PlanResult(path if ok else [], ok, (time.monotonic()-t0)*1000, expansions,
                          reason="goal found" if ok else "max_steps")
