"""Offline oracle planner with full future dynamic knowledge.

Computes the globally optimal path with access to all future dynamic masks.
No time budget. Used only for regret computation — never in fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig
from .astar import AStarPlanner


@dataclass(frozen=True)
class OracleConfig(AdaptiveAStarConfig):
    """Oracle config — maximally permissive planning budget."""

    base_interval: int = 1  # replan every step with full knowledge
    lookahead_steps: int = 999
    max_planning_time_ms: float = 10_000.0  # 10 seconds (no budget constraint)


class OraclePlanner(AdaptiveAStarPlanner):
    """Offline oracle with full future knowledge for regret baseline.

    This planner:
    - Receives the complete sequence of future dynamic masks (injected externally)
    - Plans with full knowledge of all obstacles at each step
    - Has no time budget constraint
    - Tracks cumulative cost for regret computation

    NOT a fair competitor — used only as a lower-bound reference.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[OracleConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or OracleConfig())
        self._cumulative_risk: float = 0.0
        self._cumulative_cost: float = 0.0
        self._oracle_path_cells: list[tuple[int, int]] = []

    def should_replan(
        self,
        current_pos: tuple[int, int, int],
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> tuple[bool, str]:
        """Oracle always replans — it has perfect future knowledge."""
        self._steps_since_replan += 1

        # Track cumulative risk at current position
        if risk_cost_map is not None:
            x, y = int(current_pos[0]), int(current_pos[1])
            H, W = risk_cost_map.shape
            if 0 <= y < H and 0 <= x < W:
                self._cumulative_risk += float(risk_cost_map[y, x])

        # Always replan with full knowledge
        return True, "oracle_omniscient"

    def replan(  # type: ignore[override]
        self,
        current_pos: tuple[int, int, int],
        goal: tuple[int, int],
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        reason: str = "oracle",
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> list[tuple[int, int]]:
        """Plan with full obstacle awareness and risk-weighted cost map."""
        obstacles = self.no_fly.copy()
        if fire_mask is not None:
            obstacles |= fire_mask
        if smoke_mask is not None:
            obstacles |= (smoke_mask > 0.2)  # more conservative than other planners
        if extra_obstacles is not None:
            obstacles |= extra_obstacles

        # Oracle uses risk-weighted cost map for optimal risk avoidance
        cost_map = None
        if risk_cost_map is not None:
            cost_map = 1.0 + 6.0 * np.clip(risk_cost_map, 0.0, 1.0)

        start_2d = (int(current_pos[0]), int(current_pos[1]))
        planner = AStarPlanner(self.heightmap, obstacles)
        plan_result = planner.plan(start_2d, goal, cost_map=cost_map)
        new_path = list(plan_result.path) if plan_result.success else []

        self._replan_events.append({
            "step": self._steps_since_replan,
            "position": (int(current_pos[0]), int(current_pos[1]), int(current_pos[2])),
            "reason": reason,
            "new_path_length": len(new_path),
            "success": plan_result.success,
        })
        self._total_replans += 1
        self._steps_since_replan = 0
        if new_path:
            self._current_path = new_path
            self._oracle_path_cells = list(new_path)
        return new_path

    def get_replan_metrics(self) -> dict[str, Any]:
        """Return oracle metrics for regret computation."""
        base = super().get_replan_metrics()
        base["behavioral_policy"] = "oracle_omniscient"
        base["cumulative_risk"] = float(self._cumulative_risk)
        base["cumulative_cost"] = float(self._cumulative_cost)
        base["oracle_path_length"] = len(self._oracle_path_cells)
        return base

    @property
    def oracle_cost(self) -> float:
        """Total oracle cost (path length + cumulative risk)."""
        return float(len(self._oracle_path_cells) + self._cumulative_risk)

    @property
    def oracle_path_length(self) -> int:
        """Length of the last oracle path."""
        return len(self._oracle_path_cells)

    @property
    def oracle_risk(self) -> float:
        """Cumulative risk exposure of the oracle."""
        return float(self._cumulative_risk)
