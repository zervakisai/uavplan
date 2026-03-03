"""Planner base class and PlanResult (PL-1, PL-2).

All planners implement PlannerBase with plan(), update(), should_replan().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlanResult:
    """Result of a planner.plan() call (PL-2)."""

    path: list[tuple[int, int]]  # (x, y) waypoints, inclusive start+goal
    success: bool
    compute_time_ms: float
    expansions: int = 0
    replans: int = 0
    reason: str = ""


class PlannerBase(ABC):
    """Abstract planner interface (PL-1).

    Constructor receives heightmap, no_fly mask, and optional config.
    plan() returns a PlanResult with path in (x, y) order (PL-6).
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Any = None,
    ) -> None:
        self._heightmap = heightmap
        self._no_fly = no_fly
        self._H, self._W = heightmap.shape
        self._config = config

    @abstractmethod
    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Plan a path from start to goal (PL-1).

        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            cost_map: optional per-cell cost overlay

        Returns:
            PlanResult with path as list[(x, y)] inclusive start+goal (PL-6).
        """
        ...

    def update(self, dyn_state: dict[str, Any]) -> None:
        """Accept dynamic state update (PL-5). Default: no-op."""

    def set_seed(self, seed: int) -> None:
        """Set RNG seed for stochastic planners. Default: no-op."""

    def should_replan(
        self,
        current_pos: tuple[int, int],
        current_path: list[tuple[int, int]],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        """Check if replanning is needed (PL-4). Default: no."""
        return (False, "")
