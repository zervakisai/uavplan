"""Metrics computation (ME-1..ME-4)."""

from __future__ import annotations

import math

from flare.metrics.schema import EpisodeMetrics


# ---------------------------------------------------------------------------
# Mission-impact scoring (Change B)
# ---------------------------------------------------------------------------

# Fire-coupled survival parameters (paper Eq. 3).
_KAPPA = 5.0  # fire-coupling constant κ
# Per-severity base decay λ₀ keyed by casualty weight
# (3.0 → CRITICAL, 2.0 → SERIOUS, 1.0 → MINOR), matching missions/triage.py.
_WEIGHT_TO_LAMBDA0 = {3.0: 0.02, 2.0: 0.008, 1.0: 0.002}

# Canonical mission-score decay horizon T (paper §3.5: T = 800 steps).
_DECAY_HORIZON = 800


def medication_efficacy(
    delivery_step: int,
    decay_horizon: int = _DECAY_HORIZON,
    max_steps: int = 2000,
) -> float:
    """Insulin / pharma efficacy: degrades quadratically with delay, E(t).

    E(t) = max(0, 1 - (t/T)^2) with decay horizon T = decay_horizon.
    Returns 1.0 if delivered instantly, 0.0 if never delivered (delivery_step
    at/after max_steps = episode length).
    """
    if delivery_step <= 0:
        return 1.0
    if delivery_step >= max_steps:
        return 0.0
    return max(0.0, 1.0 - (delivery_step / decay_horizon) ** 2)


def triage_value(
    task_events: list[dict],
    max_steps: int = 2000,
    kappa: float = _KAPPA,
    lambda_scale: float = 1.0,
) -> float:
    """SAR rescue value: Σ(weight × fire-coupled survival) — paper Eq. (3).

    S_i(t) = exp(-λ_eff·t), λ_eff = λ₀(1 + κ / max(d_fire, 1)), where λ₀ is the
    per-severity base decay rate (mapped from the casualty weight and scaled by
    lambda_scale) and κ (kappa) is the fire-coupling constant. Each
    task_completed event carries 'weight', 'step_idx' (t) and 'd_fire' (L2
    distance to the nearest active fire at completion; large → no coupling).
    """
    total = 0.0
    for ev in task_events:
        if ev.get("type") != "task_completed":
            continue
        w = ev.get("weight", 1.0)
        t = ev.get("step_idx", max_steps)
        d_fire = ev.get("d_fire", 999.0)
        lambda0 = _WEIGHT_TO_LAMBDA0.get(w, 0.008) * lambda_scale
        lambda_eff = lambda0 * (1.0 + kappa / max(d_fire, 1.0))
        total += w * math.exp(-lambda_eff * t)
    return total


def surveillance_value(
    task_events: list[dict],
    fire_areas: list[int] | None = None,
    decay_horizon: int = _DECAY_HORIZON,
    max_steps: int = 2000,
) -> float:
    """Fire-perimeter surveillance value: Σ freshness — paper F(t) = 1 − t/T.

    freshness = max(0, 1 − t/T) with T = decay_horizon (earlier surveys are
    more valuable). fire_proximity weighting is not applied in the reported
    model (fixed at 1.0); documented as a modeling simplification.
    """
    total = 0.0
    for ev in task_events:
        if ev.get("type") != "task_completed":
            continue
        t = ev.get("step_idx", max_steps)
        freshness = max(0.0, 1.0 - t / decay_horizon)
        total += freshness
    return total


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
    mission_type: str = "",
    max_steps: int = 2000,
    decay_horizon: int = _DECAY_HORIZON,
    kappa: float = _KAPPA,
    lambda_scale: float = 1.0,
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

    # Mission-impact scoring
    task_events = [e for e in events if e.get("type") == "task_completed"]
    mission_score = 0.0
    if mission_type == "pharma_delivery":
        delivery_step = task_events[0]["step_idx"] if task_events else max_steps
        mission_score = medication_efficacy(
            delivery_step, decay_horizon=decay_horizon, max_steps=max_steps
        )
    elif mission_type == "urban_rescue":
        mission_score = triage_value(
            task_events, max_steps=max_steps, kappa=kappa, lambda_scale=lambda_scale
        )
    elif mission_type == "fire_surveillance":
        mission_score = surveillance_value(
            task_events, decay_horizon=decay_horizon, max_steps=max_steps
        )

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
        "mission_score": mission_score,
        "tasks_completed": len(task_events),
        "tasks_total": final_info.get("task_progress", "0/0").split("/")[-1],
    }
