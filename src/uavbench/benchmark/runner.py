"""Benchmark runner (RU-1, RU-3, RU-4).

Exactly ONE runner. Orchestrates: scenario load -> env reset -> plan ->
step loop -> metrics. Owns the authoritative step_idx (EV-1).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any

from uavbench.envs.base import TerminationReason
from uavbench.envs.urban import UrbanEnvV2
from uavbench.metrics.compute import compute_episode_metrics
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario


@dataclass
class EpisodeResult:
    """Result of a single episode run.

    Attributes used by determinism tests (DC-2):
        events: list of event dicts
        trajectory: list of (x, y) agent positions
        metrics: episode metrics dict
        frame_hashes: per-frame hash strings (empty if not rendering)
    """

    events: list[dict] = field(default_factory=list)
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    frame_hashes: list[str] = field(default_factory=list)


def run_episode(
    scenario_id: str,
    planner_id: str,
    seed: int,
    render: bool = False,
    frame_callback: "Callable | None" = None,
) -> EpisodeResult:
    """Run a single deterministic episode (RU-3).

    Args:
        scenario_id: registered scenario name
        planner_id: registered planner name
        seed: RNG seed (DC-1)
        render: whether to capture frame hashes (Phase 8)
        frame_callback: optional read-only callback called each step with
            (heightmap, state_dict, dynamic_state_dict, config) for rendering.
            Does NOT affect determinism — callback must be side-effect-free
            on the simulation state.

    Returns:
        EpisodeResult with events, trajectory, metrics, frame_hashes.
    """
    # Load scenario
    config = load_scenario(scenario_id)

    # Create environment
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)

    # Export planner inputs
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    # Capture static map layers for visualization
    _landuse_map = getattr(env, '_landuse_map', None)
    _roads_mask = getattr(env, '_roads', None)

    # Mission metadata for rendering
    _mission_meta = _extract_mission_meta(config, info)

    # Mission-aware routing (MC-2): POI snapped to corridor midpoint
    mission_poi = info.get("objective_poi", goal_xy)
    mission_task_done = (mission_poi == goal_xy)  # fly-through: no POI phase

    # Create planner
    planner_cls = PLANNERS[planner_id]
    planner = planner_cls(heightmap, no_fly, config)
    planner.set_seed(seed)

    # Initial plan: ALWAYS start→goal (FC-1 corridor alignment).
    # POI is snapped to the corridor midpoint by the env, so the agent
    # naturally visits it while following the start→goal route.
    # Static planners get exactly ONE plan() call — fire/roadblock
    # interdictions on the corridor force them to fail. Adaptive
    # planners replan via should_replan().
    plan_result = planner.plan(start_xy, goal_xy)

    # For adaptive planner replans: target POI first, then goal
    current_target = goal_xy if mission_task_done else mission_poi

    # Track path execution
    path = plan_result.path if plan_result.success else []
    path_idx = 0
    trajectory: list[tuple[int, int]] = [start_xy]

    # Accumulated plan length across all plan() calls
    planned_waypoints_total = len(path)

    # Step loop (RU-4: runner owns step_idx)
    step_idx = 0
    replan_count = 0
    replan_attempts = 0  # includes suppressed replans
    naive_replan_count = 0  # replans suppressed by RS-1
    _failed_plan_cooldown = 0  # steps to skip replanning after a failed plan
    _last_plan_step = 0  # step_idx of last successful plan (for VC-2 STALE badge)
    should = False  # last replan decision (for frame_callback)
    reason = ""  # last replan reason (for frame_callback)
    terminated = False
    truncated = False
    final_info = info

    # POI unreachability detection: if the agent makes no progress toward
    # the POI for _POI_STUCK_LIMIT steps, abandon POI and target goal directly.
    _poi_stuck_counter = 0
    _poi_best_dist = float("inf")
    _POI_STUCK_LIMIT = 30

    # Wall-clock timeout safety net
    _wall_start = _time.perf_counter()
    _WALL_TIMEOUT_S = 300.0

    while not terminated and not truncated:
        step_idx += 1

        # Mission POI: STAY at POI until task completed, then switch to goal
        if not mission_task_done and env.agent_xy == current_target:
            action = 4  # STAY at POI for service_time
        else:
            # Determine action from path
            action = _path_to_action(env.agent_xy, path, path_idx)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info

        # Wall-clock timeout check
        if _time.perf_counter() - _wall_start > _WALL_TIMEOUT_S:
            final_info = dict(final_info)
            final_info["termination_reason"] = TerminationReason.WALL_TIMEOUT
            final_info["objective_completed"] = False
            truncated = True
        trajectory.append(env.agent_xy)

        # Check if mission task completed → switch replan target to goal
        if not mission_task_done:
            task_progress = info.get("task_progress", "0/0")
            if task_progress.startswith("1/"):
                mission_task_done = True
                current_target = goal_xy
                _poi_stuck_counter = 0
                # Only replan if current path doesn't lead to goal.
                # Static planners on start→goal path: path[-1]==goal → no
                # replan (they continue on original path through fire/
                # roadblock interdictions → stuck → timeout).
                # Adaptive planners that replanned to POI during leg 1:
                # path[-1]==POI → need fresh plan to goal.
                if not path or path[-1] != goal_xy:
                    plan_result_new = planner.plan(env.agent_xy, goal_xy)
                    if plan_result_new.success:
                        path = plan_result_new.path
                        path_idx = 0
                        replan_count += 1
                        planned_waypoints_total += len(plan_result_new.path)
                        _last_plan_step = step_idx
            else:
                # POI unreachability detection: if stuck for _POI_STUCK_LIMIT
                # steps with no progress toward POI, abandon POI and go to goal.
                # Does NOT replan — adaptive planners replan via should_replan(),
                # static planners (A*) continue on original start→goal path.
                poi_dist = (abs(env.agent_xy[0] - mission_poi[0])
                            + abs(env.agent_xy[1] - mission_poi[1]))
                if poi_dist < _poi_best_dist:
                    _poi_best_dist = poi_dist
                    _poi_stuck_counter = 0
                else:
                    _poi_stuck_counter += 1
                if _poi_stuck_counter >= _POI_STUCK_LIMIT:
                    mission_task_done = True  # abandon POI
                    current_target = goal_xy
                    _poi_stuck_counter = 0

        # Advance path index if we moved to expected next waypoint
        if path_idx < len(path) - 1:
            next_wp = path[path_idx + 1]
            if env.agent_xy == next_wp:
                path_idx += 1

        # Update planner with dynamic state
        dyn_state = env.get_dynamic_state()
        planner.update(dyn_state)

        # Check if replan needed
        if _failed_plan_cooldown > 0:
            _failed_plan_cooldown -= 1
            should, reason = False, "cooldown"
        else:
            should, reason = planner.should_replan(
                env.agent_xy, path, dyn_state, step_idx
            )
            if reason == "naive_skip":
                naive_replan_count += 1
            if should:
                replan_attempts += 1
                plan_result_new = planner.plan(env.agent_xy, current_target)
                if plan_result_new.success:
                    path = plan_result_new.path
                    path_idx = 0
                    replan_count += 1
                    planned_waypoints_total += len(plan_result_new.path)
                    _failed_plan_cooldown = 0
                    _last_plan_step = step_idx
                else:
                    # Back off after failed plan to avoid futile retries
                    _failed_plan_cooldown = 15

        # Frame callback for visualization (read-only, no sim effect)
        if frame_callback is not None:
            remaining_path = path[path_idx:] if path else []
            frame_state = {
                "step_idx": step_idx,
                "agent_xy": env.agent_xy,
                "start_xy": start_xy,
                "goal_xy": goal_xy,
                "plan_path": remaining_path,
                "plan_len": len(remaining_path),
                "plan_age_steps": step_idx - _last_plan_step,
                "plan_reason": reason if should else "",
                "trajectory": list(trajectory),
                "replans": replan_count,
                "planner_name": planner_id,
                "scenario_id": scenario_id,
                "replan_every_steps": getattr(config, "replan_every_steps", 6),
                "dynamic_block_hits": 0,
                "landuse_map": _landuse_map,
                "roads_mask": _roads_mask,
                **_mission_meta,
            }
            # Add mission fields from info
            for k in ("objective_label", "distance_to_task", "task_progress",
                       "deliverable_name", "mission_domain"):
                if k in info:
                    frame_state[k] = info[k]
            frame_callback(heightmap, frame_state, dyn_state, config)

    # Compute metrics
    metrics = compute_episode_metrics(
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
        trajectory=trajectory,
        events=env.events,
        final_info=final_info,
        plan_result=plan_result,
        replan_count=replan_count,
        goal_xy=goal_xy,
    )
    # Augment with runner-tracked fields
    metrics["planned_waypoints_len"] = planned_waypoints_total
    metrics["naive_replan_count"] = naive_replan_count
    metrics["replan_attempts"] = replan_attempts
    total_replan_decisions = replan_count + naive_replan_count
    metrics["replan_storm_ratio"] = (
        naive_replan_count / total_replan_decisions
        if total_replan_decisions > 0 else 0.0
    )
    metrics["feasible_after_guardrail"] = final_info.get(
        "feasible_after_guardrail", True
    )

    # Reject reason counts from env events (EC-1)
    reject_counts: dict[str, int] = {}
    for ev in env.events:
        if ev.get("type") == "move_rejected":
            reason = ev.get("reject_reason", "")
            reason_str = reason.value if hasattr(reason, "value") else str(reason)
            if reason_str:
                reject_counts[reason_str] = reject_counts.get(reason_str, 0) + 1
    metrics["reject_reason_counts"] = reject_counts

    return EpisodeResult(
        events=env.events,
        trajectory=trajectory,
        metrics=metrics,
        frame_hashes=[],
    )


def _extract_mission_meta(config: "ScenarioConfig", info: dict) -> dict:
    """Extract mission metadata for frame rendering."""
    meta = {}
    for k in ("objective_label", "distance_to_task", "task_progress",
               "deliverable_name", "mission_domain",
               "origin_name", "destination_name", "priority"):
        if k in info:
            meta[k] = info[k]
    # Fall back to config-derived values
    if "mission_domain" not in meta:
        meta["mission_domain"] = config.mission_type.value
    # Add config-derived fields for briefing card
    meta["difficulty"] = config.difficulty.value
    # Get constraints from mission metadata
    from uavbench.missions.engine import _MISSION_META
    mt = config.mission_type.value
    if mt in _MISSION_META:
        meta["constraints"] = _MISSION_META[mt].get("constraints", [])
    return meta


def _path_to_action(
    agent_xy: tuple[int, int],
    path: list[tuple[int, int]],
    path_idx: int,
) -> int:
    """Convert path waypoint to an action integer.

    If we're at the end of the path or path is empty, use greedy
    navigation toward the goal (last waypoint).
    """
    # Action constants
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    if not path:
        return ACTION_STAY

    # Target: next waypoint on path
    target_idx = path_idx + 1 if path_idx + 1 < len(path) else len(path) - 1
    target = path[target_idx]

    ax, ay = agent_xy
    tx, ty = target

    if (ax, ay) == (tx, ty):
        # At target — try next waypoint or stay
        if target_idx + 1 < len(path):
            tx, ty = path[target_idx + 1]
        else:
            return ACTION_STAY

    dx = tx - ax
    dy = ty - ay

    if dx == 0 and dy == 0:
        return ACTION_STAY
    if abs(dx) >= abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    else:
        return ACTION_DOWN if dy > 0 else ACTION_UP
