"""Battery/energy model (BC-1, BC-2, BC-3).

Tracks energy consumption per step. MOVE costs base_cost + wind penalty.
STAY costs hover_cost. Deterministic from env state (no internal RNG).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatteryModel:
    """UAV battery state tracker.

    Enforces BC-1 (monotonic decrease), BC-2 (depletion termination),
    BC-3 (deterministic trace — no internal RNG).
    """

    capacity_wh: float
    base_cost_per_step: float = 0.15
    hover_cost: float = 0.10
    wind_penalty_factor: float = 0.05
    low_threshold: float = 0.2
    critical_threshold: float = 0.1

    def __post_init__(self) -> None:
        self._wh: float = self.capacity_wh

    @property
    def wh(self) -> float:
        """Current battery level in Wh."""
        return self._wh

    @property
    def percent(self) -> float:
        """Current battery as percentage [0, 100]."""
        if self.capacity_wh <= 0:
            return 0.0
        return max(0.0, self._wh / self.capacity_wh * 100.0)

    @property
    def depleted(self) -> bool:
        """True if battery is at or below zero."""
        return self._wh <= 0.0

    @property
    def status(self) -> str:
        """Battery status string: OK, LOW, or CRITICAL."""
        pct = self.percent / 100.0
        if pct <= self.critical_threshold:
            return "CRITICAL"
        if pct <= self.low_threshold:
            return "LOW"
        return "OK"

    @property
    def estimated_range_steps(self) -> int:
        """Estimated steps remaining at base cost rate."""
        if self.base_cost_per_step <= 0:
            return 99999
        return max(0, int(self._wh / self.base_cost_per_step))

    def step(self, action: int, wind_speed: float = 0.0) -> float:
        """Consume energy for one step. Returns cost consumed.

        Args:
            action: 0-3=MOVE, 4=STAY
            wind_speed: current wind speed for penalty calculation

        Returns:
            Energy consumed this step (Wh).
        """
        if action == 4:  # STAY
            cost = self.hover_cost
        else:
            cost = self.base_cost_per_step + wind_speed * self.wind_penalty_factor

        self._wh = max(0.0, self._wh - cost)
        return cost

    def reset(self) -> None:
        """Reset battery to full capacity."""
        self._wh = self.capacity_wh
