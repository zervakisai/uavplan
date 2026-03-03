"""Planner registry (PL-3).

4 paper planners: 1 static baseline (A*) + 3 adaptive.
"""

from __future__ import annotations

from uavbench.planners.aggressive_replan import AggressiveReplanPlanner
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.dstar_lite import DStarLitePlanner
from uavbench.planners.periodic_replan import PeriodicReplanPlanner

PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
    "periodic_replan": PeriodicReplanPlanner,
    "aggressive_replan": AggressiveReplanPlanner,
    "dstar_lite": DStarLitePlanner,
}
