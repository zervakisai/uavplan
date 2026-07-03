"""Planner registry (PL-3).

5 paper planners across 3 families:
  - Search (static): A*
  - Search (adaptive): Periodic Replan, Aggressive Replan, Incremental A*
  - Reactive (potential field): APF
"""

from __future__ import annotations

from flare.planners.aggressive_replan import AggressiveReplanPlanner
from flare.planners.apf import APFPlanner
from flare.planners.astar import AStarPlanner
from flare.planners.chance_constrained import ChanceConstrainedPlanner
from flare.planners.cvar_conservative import CVaRConservativePlanner
from flare.planners.incremental_astar import IncrementalAStarPlanner
from flare.planners.periodic_replan import PeriodicReplanPlanner
from flare.planners.risk_aware_rrt import RiskAwareRRTStarPlanner
from flare.planners.risk_sensitive import RiskSensitivePlanner

PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
    "periodic_replan": PeriodicReplanPlanner,
    "aggressive_replan": AggressiveReplanPlanner,
    "incremental_astar": IncrementalAStarPlanner,
    "apf": APFPlanner,
    # Reviewer revision: additional risk-handling paradigms (unified ρ axis)
    "chance_constrained": ChanceConstrainedPlanner,
    "cvar_conservative": CVaRConservativePlanner,
    "risk_sensitive": RiskSensitivePlanner,
    "risk_aware_rrt": RiskAwareRRTStarPlanner,
    # Backward compatibility alias
    "dstar_lite": IncrementalAStarPlanner,
}
