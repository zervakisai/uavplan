"""Planner registry (PL-3).

5 paper planners across 3 families:
  - Search (static): A*
  - Search (adaptive): Periodic Replan, Aggressive Replan, D* Lite
  - Reactive (potential field): APF
"""

from __future__ import annotations

from uavbench.planners.aggressive_replan import AggressiveReplanPlanner
from uavbench.planners.apf import APFPlanner
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.dstar_lite import DStarLitePlanner
from uavbench.planners.periodic_replan import PeriodicReplanPlanner

PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
    "periodic_replan": PeriodicReplanPlanner,
    "aggressive_replan": AggressiveReplanPlanner,
    "dstar_lite": DStarLitePlanner,
    "apf": APFPlanner,
}
