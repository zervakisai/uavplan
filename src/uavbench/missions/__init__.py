"""Greece Government-Ready Mission Bank for UAVBench.

Provides 3 benchmark-level, dual-use, non-tactical missions:
  1. Civil Protection (wildfire crisis SA + evacuation corridor)
  2. Maritime Domain Awareness (coastal patrol + distress events)
  3. Critical Infrastructure (time-critical inspection tour)

Architecture (2-layer separation):
  Mission layer — task ordering/selection (Greedy / Lookahead OPTW)
  Route layer  — path planning between tasks (any PLANNERS planner)
"""

from uavbench.missions.spec import (
    MissionID,
    MissionSpec,
    TaskSpec,
    TaskStatus,
    DifficultyKnobs,
    ProductType,
    MissionProduct,
)
from uavbench.missions.engine import MissionEngine
from uavbench.missions.policies import GreedyPolicy, LookaheadOPTWPolicy
from uavbench.missions.runner import plan_mission, MissionResult

__all__ = [
    "MissionID",
    "MissionSpec",
    "TaskSpec",
    "TaskStatus",
    "DifficultyKnobs",
    "ProductType",
    "MissionProduct",
    "MissionEngine",
    "GreedyPolicy",
    "LookaheadOPTWPolicy",
    "plan_mission",
    "MissionResult",
]
