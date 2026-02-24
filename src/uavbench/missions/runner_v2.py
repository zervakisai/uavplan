"""EXPERIMENTAL MODULE — Not used in reported benchmark results.

This module is retained for demo/stakeholder visualization only.
The canonical evaluation path is::

    cli/benchmark.py → benchmark/runner.py → envs/urban.py

Do not use ``plan_mission_v2`` for reproducible benchmark evaluation.

Enhanced mission runner with UpdateBus + PlannerAdapter + SafetyMonitor.

``plan_mission_v2()`` is the upgraded mission runner that integrates:
  - UpdateBus for unified event pipeline
  - PlannerAdapter with ReplanPolicy (cadence/event/risk/forced triggers)
  - ConflictDetector for path-obstacle intersection checking
  - SafetyMonitor for violation counting and no-ghosting enforcement
  - DynamicObstacleManager for moving obstacles with plausible kinematics
  - ForcedReplanScheduler guaranteeing ≥2 replans per episode
  - Full causal chain logging (replan_id → trigger → reason → cost delta)

The original ``plan_mission()`` in runner.py is preserved for backward
compatibility.  This module adds ``plan_mission_v2()`` alongside it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from uavbench.missions.spec import MissionID, MissionSpec, MissionProduct, ProductType
from uavbench.missions.engine import MissionEngine, InjectionEvent
from uavbench.missions.policies import MissionPolicy, GreedyPolicy, LookaheadOPTWPolicy
from uavbench.missions.builders import build_mission
from uavbench.planners.base import BasePlanner, GridPos, PlanResult
from uavbench.planners.adapter import (
    PlannerAdapter,
    ReplanPolicy,
    ReplanPolicyConfig,
    ReplanRecord,
    ReplanTrigger,
)
from uavbench.updates.bus import EventType, UpdateEvent, UpdateBus
from uavbench.updates.conflict import ConflictDetector
from uavbench.updates.safety import SafetyMonitor, SafetyConfig
from uavbench.updates.obstacles import DynamicObstacleManager
from uavbench.updates.forced_replan import ForcedReplanScheduler


# ── Enhanced result dataclass ─────────────────────────────────────────

@dataclass
class MissionResultV2:
    """Complete result of an enhanced mission episode."""
    mission_id: str
    difficulty: str
    planner_id: str
    policy_id: str
    seed: int
    success: bool
    metrics: dict[str, float]
    task_log: list[dict[str, Any]]
    segment_log: list[dict[str, Any]]
    products: dict[str, list[dict[str, Any]]]
    event_detections: list[dict[str, Any]]
    step_count: int
    # V2 additions
    replan_log: list[dict[str, Any]] = field(default_factory=list)
    event_bus_summary: dict[str, int] = field(default_factory=dict)
    safety_summary: dict[str, Any] = field(default_factory=dict)
    forced_replan_summary: dict[str, Any] = field(default_factory=dict)
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    dynamic_obstacle_entity_data: list[dict[str, Any]] = field(default_factory=list)


# ── plan_mission_v2 API ──────────────────────────────────────────────

def plan_mission_v2(
    start: GridPos,
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    *,
    mission_id: MissionID | str,
    difficulty: Literal["easy", "medium", "hard"] = "easy",
    planner_id: str = "astar",
    policy_id: str = "greedy",
    seed: int = 0,
    risk_map: np.ndarray | None = None,
    cost_map: np.ndarray | None = None,
    roads_mask: np.ndarray | None = None,
    # V2 config
    replan_cadence: int = 10,
    min_forced_replans: int = 2,
    forced_obstacle_radius: int = 5,
    safety_check_buildings: bool = True,
    safety_check_nfz: bool = True,
) -> MissionResultV2:
    """Enhanced mission runner with full dynamic replanning pipeline.

    Key differences from plan_mission():
      - Mid-path replanning via PlannerAdapter + ConflictDetector
      - Dynamic obstacles (vehicles/vessels/workzones) per mission type
      - ForcedReplanScheduler guaranteeing ≥2 replans
      - SafetyMonitor with violation counting
      - Full causal replan log
      - UpdateBus event pipeline

    Args:
        start: starting grid position (x, y)
        heightmap: 2D obstacle map
        no_fly: 2D boolean no-fly mask
        mission_id: which mission to run
        difficulty: easy/medium/hard
        planner_id: route-layer planner name
        policy_id: mission-layer policy
        seed: random seed
        risk_map: optional 2D risk field
        cost_map: optional 2D cost overlay for planner
        roads_mask: optional road network for vehicle constraint (M1)
        replan_cadence: steps between scheduled replans
        min_forced_replans: minimum guaranteed replans (default 2)
        forced_obstacle_radius: radius of forced blocking obstacles
        safety_check_buildings: enforce building collision check
        safety_check_nfz: enforce NFZ check

    Returns:
        MissionResultV2 with all metrics, replan log, safety summary
    """
    from uavbench.planners import PLANNERS

    if isinstance(mission_id, str):
        mission_id = MissionID(mission_id)

    grid_shape = (heightmap.shape[0], heightmap.shape[1])
    map_size = heightmap.shape[1]
    rng = np.random.default_rng(seed)
    spec, schedule = build_mission(mission_id, difficulty, map_size, seed)

    # Risk map
    if risk_map is None:
        risk_map = np.zeros_like(heightmap, dtype=np.float32)

    # ── Instantiate components ────────────────────────────────────

    # 1. UpdateBus
    bus = UpdateBus()

    # 2. Engine
    engine = MissionEngine(spec, rng)
    engine.set_injection_schedule(schedule)

    # 3. Policy
    policy: MissionPolicy
    if policy_id == "lookahead":
        policy = LookaheadOPTWPolicy(depth=2)
    else:
        policy = GreedyPolicy()

    # 4. Planner + Adapter
    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")
    base_planner = PLANNERS[planner_id](heightmap, no_fly)

    conflict_detector = ConflictDetector(grid_shape=grid_shape, lookahead=15)
    replan_policy = ReplanPolicy(ReplanPolicyConfig(
        cadence_interval=replan_cadence,
    ))
    adapter = PlannerAdapter(base_planner, bus, conflict_detector, replan_policy)

    # 5. SafetyMonitor
    safety = SafetyMonitor(heightmap, no_fly, bus, SafetyConfig(
        check_buildings=safety_check_buildings,
        check_nfz=safety_check_nfz,
    ))

    # 6. Dynamic obstacle manager
    obstacle_mgr = DynamicObstacleManager(
        mission_type=mission_id.value,
        grid_shape=grid_shape,
        bus=bus,
        seed=seed,
        roads_mask=roads_mask,
    )

    # 7. ForcedReplanScheduler
    forced_scheduler = ForcedReplanScheduler(
        bus, replan_policy,
        min_replans=min_forced_replans,
        obstacle_radius=forced_obstacle_radius,
    )

    # ── Episode loop ──────────────────────────────────────────────
    current_pos = start
    path: list[GridPos] = []
    path_idx = 0
    trajectory: list[GridPos] = [start]
    last_safe_pos = start

    while not engine.done:
        step = engine.step_count + 1

        # A. Advance dynamic obstacles
        dyn_obstacle_mask = obstacle_mgr.step(step)

        # B. Process forced replan injections
        forced_mask = forced_scheduler.step(step, path, path_idx)
        if forced_mask is not None:
            dyn_obstacle_mask = dyn_obstacle_mask | forced_mask

        # C. Update adapter's dynamic state
        adapter.update_dynamic_state(
            obstacle_mask=dyn_obstacle_mask,
            nfz_mask=no_fly,
            risk_map=risk_map,
            vehicle_positions=_get_vehicle_positions(obstacle_mgr),
        )

        # D. If no active path, pick next task and plan
        if not path or path_idx >= len(path) - 1:
            target = policy.select_next_task(current_pos, engine)
            if target is None:
                engine.done = True
                break

            engine.current_target = target

            plan_result = adapter.plan(current_pos, target.xy, cost_map)
            engine.record_replan(PlanResult(
                path=plan_result.path,
                success=plan_result.success,
                compute_time_ms=plan_result.compute_time_ms,
                expansions=plan_result.expansions,
            ))

            if plan_result.success and plan_result.path:
                path = list(plan_result.path)
                path_idx = 0
                # Schedule forced replans on initial path
                forced_scheduler.schedule_from_path(
                    path, spec.knobs.time_budget, grid_shape,
                )
            else:
                engine.step(current_pos, {}, risk_at_pos=0.0)
                continue

        # E. Check if replan needed (mid-path)
        if path and path_idx < len(path):
            risk_here = _risk_at(risk_map, current_pos)
            should, trigger, reason, conflicts = adapter.step_check(
                current_pos, path_idx, step, risk_here,
            )
            if should and engine.current_target is not None:
                replan_result = adapter.try_replan(
                    current_pos, step, trigger, reason, conflicts, cost_map,
                )
                engine.record_replan(PlanResult(
                    path=replan_result.path,
                    success=replan_result.success,
                    compute_time_ms=replan_result.compute_time_ms,
                    expansions=replan_result.expansions,
                ))
                if replan_result.success and replan_result.path:
                    path = list(replan_result.path)
                    path_idx = 0

        # F. Follow path one step
        if path and path_idx < len(path) - 1:
            path_idx += 1
            next_pos = path[path_idx]
        else:
            next_pos = current_pos

        # G. Safety check BEFORE moving
        violations = safety.check(
            next_pos, step,
            dynamic_obstacle_mask=dyn_obstacle_mask,
        )

        if violations:
            # Don't move into violation — stay at last safe position
            if safety.fail_safe_active:
                next_pos = safety.get_safe_position(current_pos)
            else:
                next_pos = current_pos  # hover
        else:
            last_safe_pos = next_pos
            safety.set_hover_position(next_pos)

        # H. Compute risk and step engine
        risk_here = _risk_at(risk_map, next_pos)
        nx, ny = next_pos
        in_nfz = bool(no_fly[ny, nx]) if (0 <= ny < no_fly.shape[0] and 0 <= nx < no_fly.shape[1]) else False

        events = engine.step(
            next_pos,
            {"in_nfz": in_nfz},
            risk_at_pos=risk_here,
            energy_cost=1.0,
        )

        current_pos = next_pos
        trajectory.append(current_pos)

        # I. Generate products on task completion
        if events.get("completions"):
            for tid in events["completions"]:
                _generate_products_v2(engine, spec, tid, engine.step_count)

        # J. Clear path if target completed
        if engine.current_target is None:
            path = []
            path_idx = 0

        # K. Strict compliance abort
        if spec.strict_compliance and events.get("violation"):
            engine.done = True
            break

    # ── Collect results ───────────────────────────────────────────
    metrics = engine.compute_common_metrics()
    # Override violation count with safety monitor's count
    metrics["violation_count"] = float(safety.violation_count)
    metrics["replanning_count"] = float(adapter.replan_count)

    episode_log = engine.export_episode_log()

    return MissionResultV2(
        mission_id=spec.mission_id.value,
        difficulty=difficulty,
        planner_id=planner_id,
        policy_id=policy_id,
        seed=seed,
        success=metrics["task_completion_rate"] > 0,
        metrics=metrics,
        task_log=episode_log["tasks"],
        segment_log=episode_log["segments"],
        products=episode_log["products"],
        event_detections=episode_log["event_detections"],
        step_count=engine.step_count,
        replan_log=[r.to_dict() for r in adapter.replan_log],
        event_bus_summary=bus.summary(),
        safety_summary=safety.summary(),
        forced_replan_summary=forced_scheduler.summary(),
        trajectory=trajectory,
        dynamic_obstacle_entity_data=obstacle_mgr.get_entity_data(),
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _risk_at(risk_map: np.ndarray, pos: GridPos) -> float:
    x, y = pos
    if 0 <= y < risk_map.shape[0] and 0 <= x < risk_map.shape[1]:
        return float(risk_map[y, x])
    return 0.0


def _get_vehicle_positions(mgr: DynamicObstacleManager) -> list[tuple[int, int]]:
    """Extract vehicle positions from the obstacle manager."""
    positions: list[tuple[int, int]] = []
    if mgr.vehicle_layer is not None:
        positions.extend(mgr.vehicle_layer.get_positions())
    if mgr.vessel_layer is not None:
        positions.extend(mgr.vessel_layer.get_positions())
    if mgr.workzone_layer is not None:
        for z in mgr.workzone_layer.zones:
            if z.active:
                positions.append(z.grid_center)
    return positions


def _generate_products_v2(
    engine: MissionEngine,
    spec: MissionSpec,
    task_id: str,
    step: int,
) -> None:
    """Generate operational products when a task is completed (V2)."""
    task = next((t for t in engine.tasks if t.task_id == task_id), None)
    if task is None:
        return

    cat = task.spec.category

    if spec.mission_id == MissionID.CIVIL_PROTECTION:
        if cat == "perimeter_point":
            engine.add_product(MissionProduct(
                product_type=ProductType.FIRE_PERIMETER_GEOJSON,
                timestamp_step=step,
                data={
                    "point_id": task_id,
                    "x": task.xy[0],
                    "y": task.xy[1],
                    "perimeter_validated": True,
                },
            ))
        elif cat == "corridor_checkpoint":
            engine.add_product(MissionProduct(
                product_type=ProductType.CORRIDOR_STATUS_CSV,
                timestamp_step=step,
                data={
                    "checkpoint_id": task_id,
                    "time": step,
                    "reachable": True,
                    "risk_level": engine.total_risk / max(step, 1),
                },
            ))
        engine.add_product(MissionProduct(
            product_type=ProductType.ALERT_TIMELINE_CSV,
            timestamp_step=step,
            data={
                "event_id": task_id,
                "detected_time": task.spec.injected_at,
                "first_response_time": step,
            },
        ))

    elif spec.mission_id == MissionID.MARITIME_DOMAIN:
        if cat == "patrol_waypoint":
            engine.add_product(MissionProduct(
                product_type=ProductType.CORRIDOR_COVERAGE_CSV,
                timestamp_step=step,
                data={
                    "segment_id": task_id,
                    "last_seen_time": step,
                    "coverage_score": 1.0,
                },
            ))
        elif cat == "distress_event":
            engine.add_product(MissionProduct(
                product_type=ProductType.EVENT_RESPONSE_CSV,
                timestamp_step=step,
                data={
                    "event_id": task_id,
                    "detection_time": task.spec.injected_at,
                    "localization_time": step,
                    "response_start_time": step,
                },
            ))
            engine.record_event_detection(
                event_id=task_id,
                detected_step=task.spec.injected_at,
                response_start_step=step,
                localized_step=step,
            )

    elif spec.mission_id == MissionID.CRITICAL_INFRASTRUCTURE:
        tw_ok = True
        if task.spec.time_window is not None:
            tw_ok = task.spec.time_window[0] <= step <= task.spec.time_window[1]
        engine.add_product(MissionProduct(
            product_type=ProductType.INSPECTION_LOG_CSV,
            timestamp_step=step,
            data={
                "site_id": task_id,
                "visited_time": step,
                "time_window_ok": tw_ok,
                "risk_integral_at_visit": engine.total_risk,
            },
        ))
