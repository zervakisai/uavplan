from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import BasePlanner, PlannerConfig, PlanResult


@dataclass
class DWAConfig(PlannerConfig):
    max_steps: int = 2000


class DWAPlanner(BasePlanner):
    """Reactive local planner baseline (DWA-style greedy rollout)."""

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[DWAConfig] = None):
        super().__init__(heightmap, no_fly, config or DWAConfig())
        self.cfg: DWAConfig = self.cfg

    def plan(self, start, goal, cost_map=None) -> PlanResult:
        t0 = time.monotonic()
        self._validate_pos(start, "start")
        self._validate_pos(goal, "goal")
        cur = start
        path = [cur]
        expansions = 0
        visited = {cur}

        while cur != goal and expansions < self.cfg.max_steps:
            if (time.monotonic() - t0) * 1000 > self.cfg.max_planning_time_ms:
                return PlanResult(path=path, success=False, compute_time_ms=(time.monotonic()-t0)*1000,
                                  expansions=expansions, reason="timeout")

            nbrs = self._neighbors(cur)
            if not nbrs:
                break

            def score(p):
                risk = float(cost_map[p[1], p[0]]) if cost_map is not None else 0.0
                revisit = 0.5 if p in visited else 0.0
                return self._heuristic(p, goal) + risk + revisit

            cur = min(nbrs, key=score)
            path.append(cur)
            visited.add(cur)
            expansions += 1

        ok = cur == goal
        if not ok:
            fallback = AStarPlanner(self.heightmap, self.no_fly).plan(start, goal, cost_map)
            if fallback.success:
                return PlanResult(path=fallback.path, success=True,
                                  compute_time_ms=(time.monotonic()-t0)*1000,
                                  expansions=expansions + fallback.expansions,
                                  reason="dwa_fallback_astar")

        return PlanResult(path=path if ok else [], success=ok,
                          compute_time_ms=(time.monotonic()-t0)*1000,
                          expansions=expansions,
                          reason="goal found" if ok else "local_minima")
