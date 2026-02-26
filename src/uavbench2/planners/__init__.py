"""Planner registry (PL-3).

Exactly 6 paper planners registered.
"""

from __future__ import annotations

from uavbench2.planners.aggressive_replan import AggressiveReplanPlanner
from uavbench2.planners.astar import AStarPlanner
from uavbench2.planners.dstar_lite import DStarLitePlanner
from uavbench2.planners.mppi_grid import MPPIGridPlanner
from uavbench2.planners.periodic_replan import PeriodicReplanPlanner
from uavbench2.planners.theta_star import ThetaStarPlanner

PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
    "theta_star": ThetaStarPlanner,
    "periodic_replan": PeriodicReplanPlanner,
    "aggressive_replan": AggressiveReplanPlanner,
    "dstar_lite": DStarLitePlanner,
    "mppi_grid": MPPIGridPlanner,
}
