"""Planner registry (PL-3).

5 paper planners across 3 families:
  - Search (static): A*
  - Search (adaptive): Periodic Replan, Aggressive Replan, Incremental A*
  - Reactive (potential field): APF
"""

from __future__ import annotations

from uavbench.planners.aggressive_replan import AggressiveReplanPlanner
from uavbench.planners.apf import APFPlanner
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.incremental_astar import IncrementalAStarPlanner
from uavbench.planners.periodic_replan import PeriodicReplanPlanner

PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
    "periodic_replan": PeriodicReplanPlanner,
    "aggressive_replan": AggressiveReplanPlanner,
    "incremental_astar": IncrementalAStarPlanner,
    "apf": APFPlanner,
    # Backward compatibility alias
    "dstar_lite": IncrementalAStarPlanner,
}
