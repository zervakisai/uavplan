"""Event-triggered replanner: replans ONLY on forced events or path invalidation.

Ignores risk gradients, cadence triggers, and scheduled replanning.
Purpose: isolates pure event-driven adaptation as a behavioral baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class EventTriggeredConfig(AdaptiveAStarConfig):
    """Config that disables cadence-based replanning entirely."""

    base_interval: int = 999_999  # effectively infinite — no scheduled replans
    lookahead_steps: int = 6


class EventTriggeredPlanner(AdaptiveAStarPlanner):
    """Replans ONLY when a forced event occurs or the current path is invalid.

    Decision policy:
    - Replan on ``forced_event`` (external interdiction)
    - Replan on ``path_invalid`` (upcoming waypoints blocked)
    - IGNORE risk gradients, cadence, and planner-internal heuristics

    This produces a distinct replan trigger distribution compared to
    RiskGradientPlanner (risk-driven) and StabilityAwarePlanner (hysteresis).
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[EventTriggeredConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config or EventTriggeredConfig())
        self._trigger_histogram: dict[str, int] = {
            "forced_event": 0,
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
        """Only replan on path invalidation — never on cadence or risk."""
        self._steps_since_replan += 1

        # Check if upcoming path is physically blocked
        if self._is_path_blocked(fire_mask, traffic_positions, smoke_mask,
                                 extra_obstacles):
            self._trigger_histogram["path_invalid"] += 1
            return True, "path_invalid"

        # No cadence, no risk gradient, no scheduled replan
        return False, ""

    def get_replan_metrics(self) -> dict[str, Any]:
        """Return replanning statistics with trigger histogram."""
        base = super().get_replan_metrics()
        base["trigger_histogram"] = dict(self._trigger_histogram)
        base["behavioral_policy"] = "event_triggered"
        return base
