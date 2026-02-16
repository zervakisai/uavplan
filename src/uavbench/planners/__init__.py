from .base import BasePlanner, PlannerConfig, PlanResult
from .astar import AStarPlanner, AStarConfig
from .theta_star import ThetaStarPlanner, ThetaStarConfig
from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig
from .dstar_lite import DStarLitePlanner, DStarLiteConfig
from .lpa_star import LPAStarPlanner, LPAStarConfig
from .ad_star import ADStarPlanner, ADStarConfig
from .ara_star import ARAStarPlanner, ARAStarConfig
from .rrtx import RRTXPlanner, RRTXConfig
from .rrt_star import RRTStarPlanner, RRTStarConfig
from .dwa import DWAPlanner, DWAConfig
from .mpc import MPCPlanner, MPCConfig
from .nsga2 import NSGAIIPlanner, NSGA2Config
from .local_dwa import LocalDWAPlanner, LocalDWAConfig
from .local_teb_lite import LocalTEBLitePlanner, LocalTEBLiteConfig
from .hybrid_global_local import (
    HybridGlobalLocalPlanner,
    HybridGlobalLocalConfig,
    HybridDStarDWAPlanner,
    HybridDStarDWAConfig,
    HybridDStarTEBLitePlanner,
    HybridDStarTEBLiteConfig,
)
from .risk_mpc import RiskMPCPlanner, RiskMPCConfig
from .event_triggered import EventTriggeredPlanner, EventTriggeredConfig
from .risk_gradient import RiskGradientPlanner, RiskGradientConfig
from .stability_aware import StabilityAwarePlanner, StabilityAwareConfig
from .oracle import OraclePlanner, OracleConfig

PLANNERS = {
    "astar": AStarPlanner,
    "theta_star": ThetaStarPlanner,
    "adaptive_astar": AdaptiveAStarPlanner,
    "dstar_lite": DStarLitePlanner,
    "lpa_star": LPAStarPlanner,
    "ad_star": ADStarPlanner,
    "ara_star": ARAStarPlanner,
    "rrtx": RRTXPlanner,
    "rrt_star": RRTStarPlanner,
    "dwa": DWAPlanner,
    "mpc": MPCPlanner,
    "nsga2": NSGAIIPlanner,
    "local_dwa": LocalDWAPlanner,
    "local_teb_lite": LocalTEBLitePlanner,
    "hybrid_global_local": HybridGlobalLocalPlanner,
    "hybrid_dstar_dwa": HybridDStarDWAPlanner,
    "hybrid_dstar_teb_lite": HybridDStarTEBLitePlanner,
    "risk_mpc": RiskMPCPlanner,
    "event_triggered": EventTriggeredPlanner,
    "risk_gradient": RiskGradientPlanner,
    "stability_aware": StabilityAwarePlanner,
    "oracle": OraclePlanner,
}

__all__ = [
    "BasePlanner",
    "PlannerConfig",
    "PlanResult",
    "AStarPlanner",
    "AStarConfig",
    "ThetaStarPlanner",
    "ThetaStarConfig",
    "AdaptiveAStarPlanner",
    "AdaptiveAStarConfig",
    "DStarLitePlanner",
    "DStarLiteConfig",
    "LPAStarPlanner",
    "LPAStarConfig",
    "ADStarPlanner",
    "ADStarConfig",
    "ARAStarPlanner",
    "ARAStarConfig",
    "RRTXPlanner",
    "RRTXConfig",
    "RRTStarPlanner",
    "RRTStarConfig",
    "DWAPlanner",
    "DWAConfig",
    "MPCPlanner",
    "MPCConfig",
    "NSGAIIPlanner",
    "NSGA2Config",
    "LocalDWAPlanner",
    "LocalDWAConfig",
    "LocalTEBLitePlanner",
    "LocalTEBLiteConfig",
    "HybridGlobalLocalPlanner",
    "HybridGlobalLocalConfig",
    "HybridDStarDWAPlanner",
    "HybridDStarDWAConfig",
    "HybridDStarTEBLitePlanner",
    "HybridDStarTEBLiteConfig",
    "RiskMPCPlanner",
    "RiskMPCConfig",
    "EventTriggeredPlanner",
    "EventTriggeredConfig",
    "RiskGradientPlanner",
    "RiskGradientConfig",
    "StabilityAwarePlanner",
    "StabilityAwareConfig",
    "OraclePlanner",
    "OracleConfig",
    "PLANNERS",
]
