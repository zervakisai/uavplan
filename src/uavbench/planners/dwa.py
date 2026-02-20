"""Greedy local search planner with A* fallback.

.. deprecated:: 1.0
    This planner is not part of the paper benchmark suite and will be
    removed in v2.0.  It is kept for one release cycle for backward
    compatibility.  Use one of the 6 paper-suite planners instead
    (see ``PAPER_PLANNERS``).

A reactive planner that greedily picks the best immediate neighbor at each
step, scored by heuristic distance to goal + risk + revisit penalty.  Falls
back to A* when stuck in local minima.

This is NOT a canonical Dynamic Window Approach (DWA).  Canonical DWA
requires velocity-space sampling, acceleration-limited dynamic windows,
and trajectory simulation over sampled (v, ω) pairs.  This planner
operates on a discrete grid with 1-step greedy moves.

Canonical reference (NOT implemented here):
    Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach
    to collision avoidance. IEEE Robotics & Automation Magazine, 4(1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import BasePlanner, PlannerConfig, PlanResult


@dataclass
class GreedyLocalConfig(PlannerConfig):
    max_steps: int = 2000


# Backward-compatible alias
DWAConfig = GreedyLocalConfig


class GreedyLocalPlanner(BasePlanner):
    """Greedy 1-step local search with A* fallback.

    At each step, picks the neighbor minimizing heuristic(pos, goal) + risk
    + revisit_penalty.  Falls back to full A* if no goal-reaching greedy
    path is found.  This is a simplified reactive baseline, not canonical DWA
    (Fox, Burgard & Thrun 1997, IO pedestrian velocity space sampling).
    """

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


# Backward-compatible alias
DWAPlanner = GreedyLocalPlanner
