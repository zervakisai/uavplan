"""TRIAGE mission — multi-casualty rescue with survival decay (TR-1).

Novel contribution: survival probability couples with fire proximity,
creating urgency-driven planning where the right planner depends on
the hazard environment.

Survival function: S(t) = exp(-λ_eff * t)
  where λ_eff = λ_base * (1 + κ / max(d_fire, 1))

Fire proximity amplifies decay rate, making nearby casualties expire
faster and rewarding planners that can navigate risky fire-adjacent areas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


class Severity(str, Enum):
    """Casualty severity levels with distinct decay rates."""
    CRITICAL = "critical"
    SERIOUS = "serious"
    MINOR = "minor"


# Severity parameters (TR-1: monotonically decreasing survival)
_SEVERITY_PARAMS: dict[Severity, dict[str, float]] = {
    Severity.CRITICAL: {"base_lambda": 0.02, "weight": 3.0},
    Severity.SERIOUS: {"base_lambda": 0.008, "weight": 2.0},
    Severity.MINOR: {"base_lambda": 0.002, "weight": 1.0},
}

# Fire-coupling constant (κ): how strongly fire proximity amplifies decay
_KAPPA = 5.0


@dataclass
class Casualty:
    """A single casualty in a TRIAGE mission.

    Survival decays exponentially with time, amplified by fire proximity.
    """
    xy: tuple[int, int]
    severity: Severity
    base_lambda: float
    injected_at: int  # step when spawned
    weight: float = 1.0
    rescued: bool = False
    expired: bool = False

    def survival_prob(self, current_step: int, d_fire: float) -> float:
        """Compute survival probability S(t) (TR-1: continuous, monotone decreasing).

        Args:
            current_step: current simulation step
            d_fire: distance (cells) from this casualty to nearest fire

        Returns:
            S(t) ∈ (0, 1]
        """
        elapsed = max(0, current_step - self.injected_at)
        lambda_eff = self.base_lambda * (1.0 + _KAPPA / max(d_fire, 1.0))
        return math.exp(-lambda_eff * elapsed)

    def value(self, current_step: int, d_fire: float) -> float:
        """Compute rescue value: weight * S(t).

        Higher for critical casualties rescued quickly before fire approaches.
        """
        if self.rescued or self.expired:
            return 0.0
        return self.weight * self.survival_prob(current_step, d_fire)


class TriageMission:
    """Multi-casualty TRIAGE mission engine.

    Spawns casualties with varying severity. Survival decays over time,
    coupled with fire proximity. Planner must prioritize by urgency.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
        n_casualties: int = 5,
        start_xy: tuple[int, int] = (0, 0),
        goal_xy: tuple[int, int] = (0, 0),
        heightmap: np.ndarray | None = None,
    ) -> None:
        self._H, self._W = map_shape
        self._rng = rng
        self._casualties: list[Casualty] = []
        self._events: list[dict] = []
        self._total_value_earned = 0.0

        # Spawn casualties at random free cells
        severities = list(Severity)
        for i in range(n_casualties):
            sev = severities[i % len(severities)]
            params = _SEVERITY_PARAMS[sev]

            # Random position (avoiding start/goal and buildings)
            xy = self._random_free_pos(heightmap, start_xy, goal_xy)

            self._casualties.append(Casualty(
                xy=xy,
                severity=sev,
                base_lambda=params["base_lambda"],
                injected_at=0,
                weight=params["weight"],
            ))

    @property
    def casualties(self) -> list[Casualty]:
        return list(self._casualties)

    @property
    def events(self) -> list[dict]:
        return self._events

    @property
    def total_value(self) -> float:
        return self._total_value_earned

    @property
    def rescued_count(self) -> int:
        return sum(1 for c in self._casualties if c.rescued)

    @property
    def expired_count(self) -> int:
        return sum(1 for c in self._casualties if c.expired)

    @property
    def active_casualties(self) -> list[Casualty]:
        return [c for c in self._casualties if not c.rescued and not c.expired]

    def step(
        self,
        agent_xy: tuple[int, int],
        fire_mask: np.ndarray | None,
        current_step: int,
    ) -> None:
        """Update triage mission state.

        - Check if agent is at a casualty → rescue
        - Check if any casualty survival has dropped below threshold → expired
        """
        # Compute fire distance map
        if fire_mask is not None and fire_mask.any():
            fire_dist = distance_transform_edt(~fire_mask).astype(np.float32)
        else:
            fire_dist = np.full((self._H, self._W), 999.0, dtype=np.float32)

        for cas in self._casualties:
            if cas.rescued or cas.expired:
                continue

            cx, cy = cas.xy
            d_fire = float(fire_dist[cy, cx])

            # Check rescue (agent at casualty position)
            if agent_xy == cas.xy:
                cas.rescued = True
                value = cas.value(current_step, d_fire)
                # Use the pre-rescue value since value() returns 0 after rescued
                value = cas.weight * cas.survival_prob(current_step, d_fire)
                self._total_value_earned += value
                self._events.append({
                    "type": "casualty_rescued",
                    "step_idx": current_step,
                    "casualty_xy": cas.xy,
                    "severity": cas.severity.value,
                    "survival_at_rescue": cas.survival_prob(current_step, d_fire),
                    "value": value,
                })
                continue

            # Check expiry (survival below 5%)
            if cas.survival_prob(current_step, d_fire) < 0.05:
                cas.expired = True
                self._events.append({
                    "type": "casualty_expired",
                    "step_idx": current_step,
                    "casualty_xy": cas.xy,
                    "severity": cas.severity.value,
                })

    def inject_casualty(
        self,
        xy: tuple[int, int],
        severity: Severity,
        step: int,
    ) -> None:
        """Inject a new casualty at runtime (fire-spawned task).

        Called when fire reaches a building, creating a new rescue target.
        """
        params = _SEVERITY_PARAMS[severity]
        self._casualties.append(Casualty(
            xy=xy,
            severity=severity,
            base_lambda=params["base_lambda"],
            injected_at=step,
            weight=params["weight"],
        ))

    def get_metrics(self) -> dict[str, Any]:
        """Compute triage-specific metrics."""
        return {
            "total_triage_value": self._total_value_earned,
            "casualties_rescued": self.rescued_count,
            "casualties_expired": self.expired_count,
            "casualties_total": len(self._casualties),
            "avg_survival_at_rescue": self._avg_survival_at_rescue(),
        }

    def _avg_survival_at_rescue(self) -> float:
        """Average survival probability at the moment of rescue."""
        rescues = [e for e in self._events if e["type"] == "casualty_rescued"]
        if not rescues:
            return 0.0
        return sum(e["survival_at_rescue"] for e in rescues) / len(rescues)

    def _random_free_pos(
        self,
        heightmap: np.ndarray | None,
        start_xy: tuple[int, int],
        goal_xy: tuple[int, int],
    ) -> tuple[int, int]:
        """Pick a random free cell, avoiding start/goal/buildings."""
        for _ in range(100):
            x = int(self._rng.integers(0, self._W))
            y = int(self._rng.integers(0, self._H))
            if (x, y) == start_xy or (x, y) == goal_xy:
                continue
            if heightmap is not None and heightmap[y, x] > 0:
                continue
            return (x, y)
        # Fallback: return midpoint
        return ((start_xy[0] + goal_xy[0]) // 2, (start_xy[1] + goal_xy[1]) // 2)
