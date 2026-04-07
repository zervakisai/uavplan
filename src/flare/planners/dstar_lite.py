"""Backward compatibility shim — D*Lite renamed to IncrementalAStar.

Import from flare.planners.incremental_astar instead.
"""

from flare.planners.incremental_astar import (  # noqa: F401
    DStarLitePlanner,
    IncrementalAStarPlanner,
)

__all__ = ["DStarLitePlanner", "IncrementalAStarPlanner"]
