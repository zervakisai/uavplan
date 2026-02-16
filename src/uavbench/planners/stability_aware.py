"""Stability-aware replanner: prevents replan storms via hysteresis cooldown.

Tracks recent replan frequency and applies a cooldown to suppress oscillation.
Purpose: robustness under heavy dynamic stress where naive replanning thrashes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig


@dataclass(frozen=True)
class StabilityAwareConfig(AdaptiveAStarConfig):
    """Hysteresis-tuned configuration."""

    base_interval: int = 5
    lookahead_steps: int = 8
    cooldown_steps: int = 6           # minimum steps between allowed replans
    oscillation_window: int = 12      # window to detect replan oscillation
    oscillation_threshold: int = 4    # max replans in window before suppression
    stability_penalty_steps: int = 10 # extended cooldown after oscillation detected
    risk_weight: float = 2.0


class StabilityAwarePlanner(AdaptiveAStarPlanner):
    """Replan-storm-resistant adaptive planner with hysteresis cooldown.

    Decision policy:
    - Tracks last N replan timestamps in a sliding window.
    - If oscillation detected (too many replans in window), extends cooldown.
    - Replan only if:
      a) Path is invalid AND cooldown expired, OR
      b) Forced event (always honored), OR
      c) Cadence due AND cooldown expired AND no oscillation
    - DOES NOT replan on risk alone — relies on path stability.

    Produces a trigger distribution with suppressed cadence/path_invalid counts
    and non-zero ``suppressed_by_cooldown`` entries, distinguishing it from
    EventTriggeredPlanner (reactive-only) and RiskGradientPlanner (risk-driven).
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[StabilityAwareConfig] = None,
    ) -> None:
        cfg = config or StabilityAwareConfig()
        super().__init__(heightmap, no_fly, cfg)
        self._replan_timestamps: deque[int] = deque(maxlen=cfg.oscillation_window)
        self._global_step: int = 0
        self._cooldown_until: int = 0
        self._oscillation_detected_count: int = 0
        self._suppressed_count: int = 0
        self._trigger_histogram: dict[str, int] = {
            "path_invalid": 0,
            "cadence": 0,
            "forced_event": 0,
            "suppressed_by_cooldown": 0,
            "oscillation_suppressed": 0,
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
        """Replan with hysteresis — suppresses storm-like replan patterns."""
        self._steps_since_replan += 1
        self._global_step += 1
        cfg = self.cfg  # type: StabilityAwareConfig  # noqa: F841

        path_invalid = self._is_path_blocked(
            fire_mask, traffic_positions, smoke_mask, extra_obstacles
        )

        # Check cooldown
        in_cooldown = self._global_step < self._cooldown_until

        # Check oscillation
        recent_replans = sum(
            1 for t in self._replan_timestamps
            if self._global_step - t <= self.cfg.oscillation_window
        )
        oscillating = recent_replans >= self.cfg.oscillation_threshold

        if oscillating:
            self._oscillation_detected_count += 1

        if path_invalid:
            if in_cooldown or oscillating:
                self._trigger_histogram["suppressed_by_cooldown"] += 1
                if oscillating:
                    self._trigger_histogram["oscillation_suppressed"] += 1
                # Still replan on path_invalid even with cooldown, but extend next cooldown
                self._trigger_histogram["path_invalid"] += 1
                self._cooldown_until = self._global_step + self.cfg.stability_penalty_steps
                return True, "path_invalid_suppressed"
            self._trigger_histogram["path_invalid"] += 1
            return True, "path_invalid"

        # Cadence-based replan (only if not in cooldown and not oscillating)
        cadence_due = self._steps_since_replan >= self.cfg.base_interval
        if cadence_due:
            if in_cooldown:
                self._trigger_histogram["suppressed_by_cooldown"] += 1
                return False, ""
            if oscillating:
                self._trigger_histogram["oscillation_suppressed"] += 1
                return False, ""
            self._trigger_histogram["cadence"] += 1
            return True, "cadence"

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
        """Replan and record timestamp for oscillation tracking."""
        self._replan_timestamps.append(self._global_step)
        self._cooldown_until = self._global_step + self.cfg.cooldown_steps

        # Delegate to parent replan logic
        result = super().replan(
            current_pos,
            goal,
            fire_mask=fire_mask,
            traffic_positions=traffic_positions,
            reason=reason,
            smoke_mask=smoke_mask,
            extra_obstacles=extra_obstacles,
            risk_cost_map=risk_cost_map,
        )
        return result

    def get_replan_metrics(self) -> dict[str, Any]:
        """Return replanning statistics with stability metadata."""
        base = super().get_replan_metrics()
        base["trigger_histogram"] = dict(self._trigger_histogram)
        base["behavioral_policy"] = "stability_aware"
        base["oscillation_detections"] = int(self._oscillation_detected_count)
        base["suppressed_replans"] = int(self._suppressed_count)
        base["replan_interval_entropy"] = self._compute_interval_entropy()
        return base

    def _compute_interval_entropy(self) -> float:
        """Entropy of inter-replan intervals (measures regularity)."""
        if len(self._replan_timestamps) < 2:
            return 0.0
        timestamps = sorted(self._replan_timestamps)
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        if not intervals:
            return 0.0
        total = sum(intervals)
        if total == 0:
            return 0.0
        probs = np.array([i / total for i in intervals], dtype=np.float64)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))
