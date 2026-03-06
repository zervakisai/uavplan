"""Metrics computation (ME-1..ME-4)."""

from __future__ import annotations

from uavbench.metrics.schema import EpisodeMetrics


def compute_episode_metrics(
    scenario_id: str,
    planner_id: str,
    seed: int,
    trajectory: list[tuple[int, int]],
    events: list[dict],
    final_info: dict,
    plan_result: "PlanResult | None" = None,
    replan_count: int = 0,
    goal_xy: tuple[int, int] | None = None,
) -> dict:
    """Compute per-episode metrics dict (ME-1)."""
    success = final_info.get("termination_reason", "").value == "success" if hasattr(
        final_info.get("termination_reason", ""), "value"
    ) else final_info.get("termination_reason", "") == "success"

    tr = final_info.get("termination_reason", "in_progress")
    tr_str = tr.value if hasattr(tr, "value") else str(tr)

    planned_len = len(plan_result.path) if plan_result and plan_result.success else 0

    # Distance to goal at end of episode
    distance_to_goal_final = -1.0
    if goal_xy is not None and trajectory:
        last = trajectory[-1]
        distance_to_goal_final = abs(last[0] - goal_xy[0]) + abs(last[1] - goal_xy[1])

    # Count collisions and NFZ violations from events (EC-1)
    collision_count = 0
    nfz_violations = 0
    fire_exposure_steps = 0
    for ev in events:
        reason = ev.get("reject_reason", "")
        reason_str = reason.value if hasattr(reason, "value") else str(reason)
        if reason_str in ("building", "fire", "fire_buffer", "smoke",
                          "traffic_closure", "traffic_buffer"):
            collision_count += 1
        elif reason_str in ("no_fly", "dynamic_nfz"):
            nfz_violations += 1
        if reason_str in ("fire", "fire_buffer"):
            fire_exposure_steps += 1

    return {
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "seed": seed,
        "success": success,
        "termination_reason": tr_str,
        "objective_completed": final_info.get("objective_completed", False),
        "path_length": planned_len,
        "executed_steps_len": len(trajectory),
        "planned_waypoints_len": planned_len,
        "replans": replan_count,
        "collision_count": collision_count,
        "nfz_violations": nfz_violations,
        "distance_to_goal_final": distance_to_goal_final,
        "fire_exposure_steps": fire_exposure_steps,
    }
