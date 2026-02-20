import warnings

from .base import BasePlanner, PlannerConfig, PlanResult
from .astar import AStarPlanner, AStarConfig
from .theta_star import ThetaStarPlanner, ThetaStarConfig
from .adaptive_astar import AdaptiveAStarPlanner, AdaptiveAStarConfig
from .dstar_lite import PeriodicReplanPlanner, PeriodicReplanConfig, DStarLitePlanner, DStarLiteConfig
from .ad_star import AggressiveReplanPlanner, AggressiveReplanConfig, ADStarPlanner, ADStarConfig
from .dwa import GreedyLocalPlanner, GreedyLocalConfig, DWAPlanner, DWAConfig
from .mppi import GridMPPIPlanner, MPPIConfig, MPPIPlanner
from .dstar_lite_real import DStarLiteRealPlanner


# ── Deprecation metadata ────────────────────────────────────────

# Legacy alias → preferred honest key.
# These aliases will be removed in v2.0.
DEPRECATED_ALIASES: dict[str, str] = {
    "dstar_lite": "periodic_replan",
    "ad_star": "aggressive_replan",
    "dwa": "greedy_local",
    "mppi": "grid_mppi",
}

# Planners excluded from the paper benchmark suite.
# Kept for one release cycle; will be removed in v2.0.
DEPRECATED_PLANNERS: frozenset[str] = frozenset({"greedy_local"})


# ── Paper-suite planner list (the 6 planners for the paper) ─────

PAPER_PLANNERS: tuple[str, ...] = (
    "astar",
    "theta_star",
    "periodic_replan",
    "aggressive_replan",
    "incremental_dstar_lite",
    "grid_mppi",
)


# ── Registry with deprecation warnings ─────────────────────────

class _PlannerRegistry(dict):
    """Planner registry that emits DeprecationWarning for legacy keys.

    Deprecated alias keys (``dstar_lite``, ``ad_star``, ``dwa``, ``mppi``)
    and deprecated planners (``greedy_local``) remain accessible for one
    release cycle but emit warnings on access.  Planned removal: v2.0.
    """

    def __getitem__(self, key):
        if key in DEPRECATED_ALIASES:
            new_key = DEPRECATED_ALIASES[key]
            extra = ""
            if new_key in DEPRECATED_PLANNERS:
                extra = f" Note: '{new_key}' is itself deprecated (not in paper suite)."
            warnings.warn(
                f"Planner key '{key}' is deprecated and will be removed in v2.0. "
                f"Use '{new_key}' instead.{extra}",
                DeprecationWarning,
                stacklevel=2,
            )
        elif key in DEPRECATED_PLANNERS:
            warnings.warn(
                f"Planner '{key}' is deprecated (not in the paper benchmark suite) "
                f"and will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)


PLANNERS = _PlannerRegistry({
    # ── Paper suite (6 planners) ──
    "astar": AStarPlanner,
    "theta_star": ThetaStarPlanner,
    "periodic_replan": PeriodicReplanPlanner,
    "aggressive_replan": AggressiveReplanPlanner,
    "incremental_dstar_lite": DStarLiteRealPlanner,
    "grid_mppi": GridMPPIPlanner,
    # ── Deprecated: not in paper suite (removal planned v2.0) ──
    "greedy_local": GreedyLocalPlanner,
    # ── Deprecated aliases (removal planned v2.0) ──
    "dstar_lite": DStarLitePlanner,
    "ad_star": ADStarPlanner,
    "dwa": DWAPlanner,
    "mppi": MPPIPlanner,
})

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
    # Paper-suite planners
    "PeriodicReplanPlanner",
    "PeriodicReplanConfig",
    "AggressiveReplanPlanner",
    "AggressiveReplanConfig",
    "GridMPPIPlanner",
    "DStarLiteRealPlanner",
    # Deprecated (removal planned v2.0)
    "GreedyLocalPlanner",
    "GreedyLocalConfig",
    "DStarLitePlanner",
    "DStarLiteConfig",
    "ADStarPlanner",
    "ADStarConfig",
    "DWAPlanner",
    "DWAConfig",
    "MPPIPlanner",
    "MPPIConfig",
    # Registry
    "PLANNERS",
    "PAPER_PLANNERS",
    "DEPRECATED_ALIASES",
    "DEPRECATED_PLANNERS",
]
