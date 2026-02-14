from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import BasePlanner, PlannerConfig, PlanResult

GridPos = tuple[int, int]


@dataclass
class DStarLiteConfig(PlannerConfig):
    replan_interval: int = 8


class DStarLitePlanner(BasePlanner):
    """Incremental-search baseline with D* Lite-style replanning hooks."""

    def __init__(self, heightmap: np.ndarray, no_fly: np.ndarray, config: Optional[DStarLiteConfig] = None):
        super().__init__(heightmap, no_fly, config or DStarLiteConfig())
        self.cfg: DStarLiteConfig = self.cfg
        self._current_path: list[GridPos] = []
        self._steps_since_replan = 0
        self._total_replans = 0

    def plan(self, start: GridPos, goal: GridPos, cost_map: Optional[np.ndarray] = None) -> PlanResult:
        t0 = time.monotonic()
        planner = AStarPlanner(self.heightmap, self.no_fly)
        res = planner.plan(start, goal, cost_map)
        self._current_path = list(res.path)
        self._steps_since_replan = 0
        res.compute_time_ms = (time.monotonic() - t0) * 1000
        return res

    def should_replan(self, current_pos: GridPos, current_path: list[GridPos], dyn_state: dict[str, Any], step: int) -> tuple[bool, str]:
        self._steps_since_replan += 1
        if len(current_path) > 1 and self._is_blocked(current_path[1]):
            return True, "path_blocked"
        if self._steps_since_replan >= self.cfg.replan_interval:
            return True, "scheduled"
        return False, ""

    def replan(self, current_pos: GridPos, goal: GridPos, cost_map: Optional[np.ndarray] = None) -> PlanResult:
        self._total_replans += 1
        self._steps_since_replan = 0
        return self.plan(current_pos, goal, cost_map)
