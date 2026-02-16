"""Risk-gradient replanner: replans based on rolling risk derivative.

Computes a rolling derivative of risk_exposure_integral and replans when
the risk gradient exceeds a threshold OR local peak risk is too high.
Independent of path blocking — purely risk-aware responsiveness.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class RiskGradientConfig(AdaptiveAStarConfig):
    """Risk-gradient-specific thresholds."""

    base_interval: int = 999_999  # disable cadence — replans driven by risk only
    lookahead_steps: int = 8
    risk_gradient_threshold: float = 0.15  # d(risk)/dt threshold for replan
    peak_risk_threshold: float = 0.60      # local peak risk that triggers replan
    risk_window_size: int = 8              # rolling window for gradient computation
    risk_weight: float = 4.0               # cost-map amplification


class RiskGradientPlanner(AdaptiveAStarPlanner):
    """Replans when the risk gradient or local peak risk exceeds thresholds.

    Decision policy:
    - Maintain a rolling window of per-step risk values.
    - Compute d(risk)/dt as the finite difference over the window.
    - Replan if gradient exceeds ``risk_gradient_threshold``.
    - Replan if any cell in lookahead has risk > ``peak_risk_threshold``.
    - Also replans on path invalidation (standard safety).
    - DOES NOT replan on cadence or forced events alone.

    Produces a trigger distribution dominated by risk_spike / risk_gradient
    entries, distinguishing it from EventTriggeredPlanner and StabilityAwarePlanner.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[RiskGradientConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or RiskGradientConfig())
        cfg = config or RiskGradientConfig()
        self._risk_history: deque[float] = deque(maxlen=cfg.risk_window_size)
        self._latest_risk: np.ndarray | None = None
        self._trigger_histogram: dict[str, int] = {
            "risk_gradient": 0,
            "risk_spike": 0,
            "path_invalid": 0,
        }

    def should_replan(
        self,
        current_pos: tuple[int, int, int],
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> tuple[bool, str]:
        """Replan on risk gradient, peak risk, or path invalidation."""
        self._steps_since_replan += 1
        self._latest_risk = risk_cost_map

        cfg = self.cfg  # type: RiskGradientConfig  # noqa: F841

        # 1. Path invalidation (safety baseline)
        if self._is_path_blocked(fire_mask, traffic_positions, smoke_mask,
                                 extra_obstacles):
            self._trigger_histogram["path_invalid"] += 1
            return True, "path_invalid"

        # 2. Risk gradient check
        if risk_cost_map is not None:
            x, y = int(current_pos[0]), int(current_pos[1])
            H, W = risk_cost_map.shape
            if 0 <= y < H and 0 <= x < W:
                current_risk = float(risk_cost_map[y, x])
            else:
                current_risk = 0.0
            self._risk_history.append(current_risk)

            if len(self._risk_history) >= 3:
                arr = list(self._risk_history)
                gradient = (arr[-1] - arr[0]) / len(arr)
                if gradient > self.cfg.risk_gradient_threshold:
                    self._trigger_histogram["risk_gradient"] += 1
                    return True, "risk_gradient"

            # 3. Local peak risk in lookahead
            if self._current_path:
                end_idx = min(self.cfg.lookahead_steps, len(self._current_path))
                for px, py in self._current_path[:end_idx]:
                    if 0 <= py < H and 0 <= px < W:
                        if float(risk_cost_map[py, px]) > self.cfg.peak_risk_threshold:
                            self._trigger_histogram["risk_spike"] += 1
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
        """Replan with risk-amplified cost map."""
        from .astar import AStarPlanner

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
            cost_map = 1.0 + float(self.cfg.risk_weight) * np.clip(risk, 0.0, 1.0)

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
        return new_path

    def get_replan_metrics(self) -> dict[str, Any]:
        """Return replanning statistics with trigger histogram."""
        base = super().get_replan_metrics()
        base["trigger_histogram"] = dict(self._trigger_histogram)
        base["behavioral_policy"] = "risk_gradient"
        base["risk_history_final"] = list(self._risk_history)
        return base
