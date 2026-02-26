"""Metrics schemas (ME-1, ME-4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    """Per-episode metrics (ME-1)."""

    scenario_id: str
    planner_id: str
    seed: int
    success: bool
    termination_reason: str
    objective_completed: bool
    path_length: int
    executed_steps_len: int
    planned_waypoints_len: int
    planning_time_ms: float
    replans: int
    collision_count: int
    nfz_violations: int
