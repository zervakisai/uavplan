"""SafetyMonitor: enforces safety contracts and no-ghosting invariant.

Responsibilities:
  - Validate every step: agent must not occupy a blocked cell
  - Increment violation counter on infraction
  - Log violation events with position, type, step
  - Trigger fail-safe hover when violation threshold exceeded
  - Publish CONSTRAINT events on the bus for visualization (red flash)

Safety contracts enforced:
  SC-1  No ghosting through buildings / static obstacles
  SC-2  No entry into active NFZ
  SC-3  No occupation of fire / hard-obstacle cells
  SC-4  Fail-safe hover after N consecutive violations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from uavbench.updates.bus import EventType, UpdateEvent, UpdateBus


GridPos = tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Violation record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Violation:
    """Record of a single safety violation."""
    step: int
    position: GridPos
    violation_type: str       # "building", "nfz", "obstacle", "out_of_bounds"
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# SafetyMonitor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SafetyConfig:
    """Configuration for safety monitor."""
    fail_safe_threshold: int = 3        # consecutive violations → hover
    max_violations: int = 100           # hard cap before abort
    check_buildings: bool = True
    check_nfz: bool = True
    check_dynamic_obstacles: bool = True


class SafetyMonitor:
    """Enforces safety contracts and logs violations.

    Integrated into the episode loop: called once per step with the
    agent's new position and current obstacle state.

    Parameters
    ----------
    heightmap : np.ndarray
        [H, W] static obstacle map (>0 = building).
    no_fly : np.ndarray
        [H, W] bool static no-fly mask.
    bus : UpdateBus, optional
        Event bus for publishing violation events.
    config : SafetyConfig, optional
        Safety monitor configuration.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        bus: UpdateBus | None = None,
        config: SafetyConfig | None = None,
    ) -> None:
        self.heightmap = heightmap
        self.static_nfz = no_fly.astype(bool, copy=True)
        self.bus = bus
        self.cfg = config or SafetyConfig()
        self.H, self.W = heightmap.shape

        # State
        self._violations: list[Violation] = []
        self._consecutive_violations: int = 0
        self._fail_safe_active: bool = False
        self._hover_position: GridPos | None = None

    # ── Core check ────────────────────────────────────────────────

    def check(
        self,
        position: GridPos,
        step: int,
        *,
        dynamic_obstacle_mask: np.ndarray | None = None,
        dynamic_nfz_mask: np.ndarray | None = None,
    ) -> list[Violation]:
        """Validate agent position against all safety contracts.

        Returns list of violations (empty = safe).
        """
        x, y = position
        violations: list[Violation] = []

        # SC-0: Bounds check
        if not (0 <= x < self.W and 0 <= y < self.H):
            violations.append(Violation(
                step=step,
                position=position,
                violation_type="out_of_bounds",
                description=f"Position ({x},{y}) out of grid [{self.W}×{self.H}]",
            ))
            self._record_violations(violations, step)
            return violations

        # SC-1: No ghosting through buildings
        if self.cfg.check_buildings and self.heightmap[y, x] > 0:
            violations.append(Violation(
                step=step,
                position=position,
                violation_type="building",
                description=f"Building collision at ({x},{y}) h={self.heightmap[y, x]}",
            ))

        # SC-2: Static NFZ
        if self.cfg.check_nfz and self.static_nfz[y, x]:
            violations.append(Violation(
                step=step,
                position=position,
                violation_type="nfz",
                description=f"Static NFZ entry at ({x},{y})",
            ))

        # SC-2b: Dynamic NFZ
        if self.cfg.check_nfz and dynamic_nfz_mask is not None and dynamic_nfz_mask[y, x]:
            violations.append(Violation(
                step=step,
                position=position,
                violation_type="nfz",
                description=f"Dynamic NFZ entry at ({x},{y})",
            ))

        # SC-3: Dynamic obstacles (fire, traffic, etc.)
        if self.cfg.check_dynamic_obstacles and dynamic_obstacle_mask is not None:
            if dynamic_obstacle_mask[y, x]:
                violations.append(Violation(
                    step=step,
                    position=position,
                    violation_type="obstacle",
                    description=f"Dynamic obstacle collision at ({x},{y})",
                ))

        self._record_violations(violations, step)
        return violations

    def _record_violations(self, violations: list[Violation], step: int) -> None:
        """Update state and publish events for detected violations."""
        if violations:
            self._violations.extend(violations)
            self._consecutive_violations += 1

            # Fail-safe hover
            if self._consecutive_violations >= self.cfg.fail_safe_threshold:
                self._fail_safe_active = True

            # Publish violation events on bus
            if self.bus is not None:
                for v in violations:
                    self.bus.publish(UpdateEvent(
                        event_type=EventType.CONSTRAINT,
                        step=step,
                        description=f"VIOLATION:{v.violation_type}:{v.description}",
                        severity=1.0,
                        position=v.position,
                        payload={"violation_type": v.violation_type},
                    ))
        else:
            self._consecutive_violations = 0
            if self._fail_safe_active:
                self._fail_safe_active = False

    # ── Fail-safe ─────────────────────────────────────────────────

    @property
    def fail_safe_active(self) -> bool:
        """True if fail-safe hover is active (too many consecutive violations)."""
        return self._fail_safe_active

    def get_safe_position(self, current_pos: GridPos) -> GridPos:
        """Return the position the agent should hold during fail-safe.

        If hover position was recorded, return that; else return current.
        """
        if self._hover_position is not None:
            return self._hover_position
        return current_pos

    def set_hover_position(self, pos: GridPos) -> None:
        """Record the last known safe position for fail-safe hover."""
        self._hover_position = pos

    # ── Query ─────────────────────────────────────────────────────

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    @property
    def violations(self) -> list[Violation]:
        return list(self._violations)

    def violations_by_type(self) -> dict[str, int]:
        """Count violations by type."""
        counts: dict[str, int] = {}
        for v in self._violations:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "total_violations": len(self._violations),
            "by_type": self.violations_by_type(),
            "fail_safe_active": self._fail_safe_active,
            "consecutive_violations": self._consecutive_violations,
        }
