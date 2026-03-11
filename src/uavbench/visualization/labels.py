"""Unified planner display names and colors for all figures.

Single source of truth — imported by all figure-generating scripts.
"""

from __future__ import annotations

# Registry order (uses actual planner registry keys)
PLANNER_ORDER = [
    "astar", "periodic_replan", "aggressive_replan",
    "incremental_astar", "apf",
]

# Full names (for figure titles, HUD)
PLANNER_LABELS = {
    "astar": "A*",
    "periodic_replan": "Periodic Replan",
    "aggressive_replan": "Aggressive Replan",
    "dstar_lite": "Incremental A*",
    "incremental_astar": "Incremental A*",  # alias
    "apf": "APF",
}

# Short names (for table columns, axis labels)
PLANNER_SHORT = {
    "astar": "A*",
    "periodic_replan": "Periodic",
    "aggressive_replan": "Aggressive",
    "dstar_lite": "Incr. A*",
    "incremental_astar": "Incr. A*",
    "apf": "APF",
}

# Okabe-Ito colorblind-safe palette (consistent across all figures)
PLANNER_COLORS = {
    "astar": "#E69F00",             # orange
    "periodic_replan": "#009E73",   # bluish green
    "aggressive_replan": "#D55E00", # vermillion
    "dstar_lite": "#CC79A7",        # reddish purple
    "incremental_astar": "#CC79A7", # alias
    "apf": "#0072B2",              # blue
}
