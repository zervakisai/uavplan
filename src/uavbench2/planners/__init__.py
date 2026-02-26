"""Planner registry (PL-3).

Exactly 6 paper planners. Phase 2 implements only astar;
others are registered as stubs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uavbench2.planners.base import PlannerBase

from uavbench2.planners.astar import AStarPlanner

# Phase 2: only astar is fully implemented.
# Other planners will be added in Phase 7.
PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
}
