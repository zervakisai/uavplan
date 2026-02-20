"""Base planner interface for UAVBench.

All planners must implement this interface to be compatible with the benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

GridPos = tuple[int, int]  # (x, y)


@dataclass
class PlannerConfig:
    """Base config for all planners (extensible)."""
    max_planning_time_ms: float = 2000.0  # Maximum allowed planning time per call (generous for 500×500 maps)
    allow_diagonal: bool = False
    block_buildings: bool = True


@dataclass
class PlanResult:
    """Result of a planning attempt."""
    path: list[GridPos]  # Sequence of (x, y) waypoints (inclusive start and goal)
    success: bool  # True if goal reached, False if no path found
    compute_time_ms: float  # Wall-clock planning time
    expansions: int = 0  # Number of nodes expanded (for performance analysis)
    replans: int = 0  # How many times replanning was triggered
    reason: str = ""  # If not success, explanation (e.g., "timeout", "no path", "blocked")


class BasePlanner(ABC):
    """Abstract base for all UAVBench planners.
    
    A planner is responsible for:
    1. Computing a path from start to goal given static obstacles
    2. Optionally updating its internal state when dynamics change
    3. Deciding when to replan (if incremental)
    4. Enforcing time budgets
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[PlannerConfig] = None,
    ):
        """Initialize planner with map and constraint layers.
        
        Args:
            heightmap: 2D array [H, W]; h > 0 = obstacle level
            no_fly: 2D bool array [H, W]; True = no-fly zone
            config: Planner-specific config
        """
        if heightmap.ndim != 2:
            raise ValueError("heightmap must be 2D array [H, W]")
        if no_fly.ndim != 2:
            raise ValueError("no_fly must be 2D array [H, W]")
        if heightmap.shape != no_fly.shape:
            raise ValueError("heightmap and no_fly must have the same shape")

        self.heightmap = heightmap
        self.no_fly = no_fly.astype(bool, copy=False)
        self.cfg = config or PlannerConfig()
        self.H, self.W = heightmap.shape

    # ========== Core API ==========

    @abstractmethod
    def plan(
        self,
        start: GridPos,
        goal: GridPos,
        cost_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """Plan a path from start to goal.
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            cost_map: Optional 2D array of per-cell costs (higher = more expensive)
                      If not provided, uniform cost (free cell = 0, blocked = inf)
        
        Returns:
            PlanResult with path, success flag, and metrics
        """
        pass

    # ========== Optional Dynamic Planner API ==========

    def update(self, dyn_state: dict[str, Any]) -> None:
        """Optional: update planner state with dynamic obstacle snapshot.
        
        Called by benchmark after each environment step, before asking for replan decision.
        Subclasses override if they track dynamic obstacles (e.g., D*, SIPP, LPA*).
        
        Args:
            dyn_state: Current dynamic state (fire, traffic, moving_target, etc.)
        """
        pass

    def should_replan(
        self,
        current_pos: GridPos,
        current_path: list[GridPos],
        dyn_state: dict[str, Any],
        step: int,
    ) -> tuple[bool, str]:
        """Optional: decide if replanning is needed.
        
        Used by adaptive planners (D*, SIPP, etc.) to trigger incremental updates.
        Default: replans every K steps or if path blocked.
        
        Args:
            current_pos: Current UAV position
            current_path: Active plan (tail from current pos to goal)
            dyn_state: Current dynamic state
            step: Current episode step number
        
        Returns:
            (should_replan: bool, reason: str)
        """
        # Simple heuristic: replan if next cell in path is now blocked
        if len(current_path) > 1:
            next_pos = current_path[1]
            if self._is_blocked(next_pos):
                return True, "path_blocked"
        return False, ""

    # ========== Utilities ==========

    def _validate_pos(self, pos: GridPos, label: str = "pos") -> None:
        """Raise ValueError if position is out of bounds."""
        x, y = pos
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise ValueError(
                f"{label}={pos} out of bounds [0,{self.W}) x [0,{self.H})"
            )

    def _is_blocked(self, pos: GridPos) -> bool:
        """Return True if position is an obstacle or no-fly zone."""
        x, y = pos
        if not (0 <= x < self.W and 0 <= y < self.H):
            return True  # Out of bounds is blocked
        if self.no_fly[y, x]:
            return True
        if self.cfg.block_buildings and self.heightmap[y, x] > 0:
            return True
        return False

    def _neighbors(self, pos: GridPos) -> list[GridPos]:
        """Return list of free neighbors (4-connected or 8-connected)."""
        x, y = pos
        candidates = [
            (x - 1, y),  # left
            (x + 1, y),  # right
            (x, y - 1),  # up
            (x, y + 1),  # down
        ]
        if self.cfg.allow_diagonal:
            candidates.extend([
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y + 1),
            ])
        return [pos for pos in candidates if not self._is_blocked(pos)]

    def _heuristic(self, pos: GridPos, goal: GridPos) -> float:
        """Manhattan or Euclidean heuristic."""
        x1, y1 = pos
        x2, y2 = goal
        if self.cfg.allow_diagonal:
            return float(max(abs(x2 - x1), abs(y2 - y1)))  # Chebyshev
        return float(abs(x2 - x1) + abs(y2 - y1))  # Manhattan
