"""Risk-aware MPC-like replanner (grid receding-horizon approximation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig
from .astar import AStarPlanner


@dataclass(frozen=True)
class RiskMPCConfig(AdaptiveAStarConfig):
    """Risk-focused settings for high-hazard environments."""
    base_interval: int = 3
    lookahead_steps: int = 8
    risk_weight: float = 4.0


class RiskMPCPlanner(AdaptiveAStarPlanner):
    """Receding-horizon risk-aware path controller on grid costmaps."""

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[RiskMPCConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or RiskMPCConfig())
        self.cfg = cast(RiskMPCConfig, self.cfg)
        self._latest_risk: np.ndarray | None = None

    def should_replan(  # type: ignore[override]
        self,
        current_pos: tuple[int, int, int],
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> tuple[bool, str]:
        self._latest_risk = risk_cost_map
        should, reason = super().should_replan(
            current_pos,
            fire_mask=fire_mask,
            traffic_positions=traffic_positions,
            smoke_mask=smoke_mask,
            extra_obstacles=extra_obstacles,
        )
        if should:
            return should, reason

        if risk_cost_map is not None and self._current_path:
            end_idx = min(self.cfg.lookahead_steps, len(self._current_path))
            for x, y in self._current_path[:end_idx]:
                if float(risk_cost_map[y, x]) > 0.65:
                    return True, "risk_spike"
        return False, ""

    def replan(  # type: ignore[override]
        self,
        current_pos: tuple[int, int, int],
        goal: tuple[int, int],
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        reason: str = "unknown",
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> list[tuple[int, int]]:
        obstacles = self.no_fly.copy()
        if fire_mask is not None:
            obstacles |= fire_mask
        if smoke_mask is not None:
            obstacles |= (smoke_mask > self.cfg.smoke_threshold)
        if extra_obstacles is not None:
            obstacles |= extra_obstacles

        cost_map = None
        risk = risk_cost_map if risk_cost_map is not None else self._latest_risk
        if risk is not None:
            cfg = cast(RiskMPCConfig, self.cfg)
            cost_map = 1.0 + float(cfg.risk_weight) * np.clip(risk, 0.0, 1.0)

        start_2d = (int(current_pos[0]), int(current_pos[1]))
        planner = AStarPlanner(self.heightmap, obstacles)
        plan_result = planner.plan(start_2d, goal, cost_map=cost_map)
        new_path = list(plan_result.path) if plan_result.success else []

        self._replan_events.append(
            {
                "step": self._steps_since_replan,
                "position": (int(current_pos[0]), int(current_pos[1]), int(current_pos[2])),
                "reason": reason,
                "new_path_length": len(new_path),
                "success": plan_result.success,
            }
        )
        self._total_replans += 1
        self._steps_since_replan = 0
        if new_path:
            self._current_path = new_path
        return new_path
