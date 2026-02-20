"""Operational metrics: safety, efficiency, feasibility.

All functions are pure numpy — no heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# --------------- Safety ---------------

def compute_safety_metrics(
    path: list[tuple[int, int]],
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    *,
    risk_map: np.ndarray | None = None,
) -> dict[str, float]:
    """Safety metrics along a planned path.

    Returns:
        nfz_violation_count: cells on no-fly mask
        building_violation_count: cells where heightmap > 0
        risk_exposure_sum: cumulative risk along path (0.0 if no risk_map)
    """
    if not path:
        return {
            "nfz_violation_count": 0.0,
            "building_violation_count": 0.0,
            "risk_exposure_sum": 0.0,
        }

    nfz = 0
    building = 0
    risk_sum = 0.0

    H, W = heightmap.shape
    for x, y in path:
        if 0 <= x < W and 0 <= y < H:
            if no_fly[y, x]:
                nfz += 1
            if heightmap[y, x] > 0:
                building += 1
            if risk_map is not None:
                risk_sum += float(risk_map[y, x])

    return {
        "nfz_violation_count": float(nfz),
        "building_violation_count": float(building),
        "risk_exposure_sum": round(risk_sum, 4),
    }


# --------------- Efficiency ---------------

def compute_efficiency_metrics(
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    planning_time: float,
) -> dict[str, float]:
    """Efficiency metrics for a planned path.

    Returns:
        path_optimality: manhattan_dist / path_length (1.0 = optimal, <1.0 = sub-optimal)
        planning_time_ms: planning wall-clock time in milliseconds
        steps_per_meter: path_length / manhattan_dist (lower = more direct)
    """
    manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    n = len(path)

    if n == 0 or manhattan == 0:
        return {
            "path_optimality": 0.0,
            "planning_time_ms": round(planning_time * 1000, 2),
            "steps_per_meter": 0.0,
        }

    return {
        "path_optimality": round(manhattan / n, 4),
        "planning_time_ms": round(planning_time * 1000, 2),
        "steps_per_meter": round(n / manhattan, 4),
    }


# --------------- Feasibility ---------------

def compute_feasibility_metrics(
    path: list[tuple[int, int]],
    success: bool,
    constraint_violations: int,
    *,
    termination_reason: str = "unknown",
) -> dict[str, float]:
    """Feasibility and path quality metrics.

    Returns:
        success: 1.0 if path found with no violations, else 0.0
        constraint_violation_rate: violations / path_length (per step)
        path_smoothness: 1 - direction_changes / (path_length - 2), range [0, 1]
    """
    n = len(path)

    if n < 2:
        return {
            "success": 1.0 if success else 0.0,
            "constraint_violation_rate": 0.0,
            "path_smoothness": 0.0,
        }

    # Direction changes
    direction_changes = 0
    if n >= 3:
        for i in range(1, n - 1):
            dx_prev = path[i][0] - path[i - 1][0]
            dy_prev = path[i][1] - path[i - 1][1]
            dx_next = path[i + 1][0] - path[i][0]
            dy_next = path[i + 1][1] - path[i][1]
            if (dx_prev, dy_prev) != (dx_next, dy_next):
                direction_changes += 1

        smoothness = round(1.0 - direction_changes / (n - 2), 4)
    else:
        smoothness = 1.0

    viol_rate = round(constraint_violations / n, 4) if n > 0 else 0.0

    return {
        "success": 1.0 if success else 0.0,
        "constraint_violation_rate": viol_rate,
        "path_smoothness": smoothness,
        "collision_terminated": 1.0 if termination_reason.startswith("collision") else 0.0,
    }


# --------------- Aggregate ---------------

def compute_all_metrics(trial_result: dict[str, Any]) -> dict[str, float]:
    """Compute all operational metrics from a single run_planner_once result.

    Loads risk_map from .npz tile if available (OSM scenarios).
    """
    path = trial_result.get("path") or []
    heightmap = trial_result.get("heightmap")
    no_fly = trial_result.get("no_fly")
    start = trial_result.get("start") or (0, 0)
    goal = trial_result.get("goal") or (0, 0)
    success = trial_result.get("success", False)
    violations = trial_result.get("constraint_violations", 0)
    planning_time = trial_result.get("planning_time", 0.0)

    # Load risk_map from tile if OSM scenario
    risk_map = None
    if trial_result.get("map_source") == "osm" and trial_result.get("osm_tile_id"):
        tile_path = Path(__file__).resolve().parents[3] / "data" / "maps" / f"{trial_result['osm_tile_id']}.npz"
        if tile_path.exists():
            risk_map = np.load(str(tile_path))["risk_map"]

    out: dict[str, float] = {}

    if heightmap is not None and no_fly is not None:
        out.update(compute_safety_metrics(path, heightmap, no_fly, risk_map=risk_map))

    out.update(compute_efficiency_metrics(path, start, goal, planning_time))
    termination_reason = trial_result.get("termination_reason", "unknown")
    out.update(compute_feasibility_metrics(path, success, violations,
                                           termination_reason=termination_reason))

    return out
