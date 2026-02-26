"""Metrics computation (ME-1..ME-4)."""

from __future__ import annotations

from uavbench2.metrics.schema import EpisodeMetrics


def compute_episode_metrics(
    scenario_id: str,
    planner_id: str,
    seed: int,
    trajectory: list[tuple[int, int]],
    events: list[dict],
    final_info: dict,
    plan_result: "PlanResult | None" = None,
) -> dict:
    """Compute per-episode metrics dict (ME-1)."""
    success = final_info.get("termination_reason", "").value == "success" if hasattr(
        final_info.get("termination_reason", ""), "value"
    ) else final_info.get("termination_reason", "") == "success"

    tr = final_info.get("termination_reason", "in_progress")
    tr_str = tr.value if hasattr(tr, "value") else str(tr)

    planned_len = len(plan_result.path) if plan_result and plan_result.success else 0
    replans = plan_result.replans if plan_result else 0

    # Energy metrics (BC-1, BC-3)
    battery_capacity = final_info.get("battery_wh", None)
    battery_remaining = final_info.get("battery_wh", 0.0)
    battery_percent = final_info.get("battery_percent", 100.0)
    energy_consumed = 0.0
    if "battery_wh" in final_info:
        # Compute from capacity (available from config via briefing)
        briefing_events = [e for e in events if e.get("type") == "mission_briefing"]
        if briefing_events:
            capacity = briefing_events[0].get("battery_capacity_wh", 150.0)
            energy_consumed = capacity - battery_remaining
        else:
            energy_consumed = 150.0 - battery_remaining  # fallback

    return {
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "seed": seed,
        "success": success,
        "termination_reason": tr_str,
        "objective_completed": final_info.get("objective_completed", False),
        "path_length": len(trajectory),
        "executed_steps_len": len(trajectory),
        "planned_waypoints_len": planned_len,
        "replans": replans,
        "collision_count": 0,
        "nfz_violations": 0,
        "energy_consumed_wh": round(energy_consumed, 4),
        "battery_remaining_percent": round(battery_percent, 2),
    }
