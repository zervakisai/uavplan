"""Mission runner: executes a mission episode end-to-end.

Provides the ``plan_mission()`` API that integrates:
  - Mission layer (policy: Greedy / LookaheadOPTW)
  - Route layer (planner: any PLANNERS entry)
  - MissionEngine (task tracking, products, metrics)

Backward compatible: ``plan(start, goal)`` still works for planners.
This adds: ``plan_mission(start, tasks, constraints, risk_layers)``.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from uavbench.missions.spec import (
    MissionID,
    MissionSpec,
    MissionProduct,
    ProductType,
    COMMON_METRICS,
)
from uavbench.missions.engine import MissionEngine, InjectionEvent
from uavbench.missions.policies import (
    MissionPolicy,
    GreedyPolicy,
    LookaheadOPTWPolicy,
)
from uavbench.missions.builders import build_mission
from uavbench.planners.base import BasePlanner, GridPos, PlanResult


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class MissionResult:
    """Complete result of a mission episode."""
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


# ── plan_mission API ──────────────────────────────────────────────────

def plan_mission(
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
) -> MissionResult:
    """High-level API: run a complete mission episode.

    This is the main entry point for the mission bank.  It:
    1. Builds the MissionSpec + injection schedule
    2. Instantiates engine, policy, planner
    3. Runs the episode step-by-step
    4. Returns MissionResult with metrics, logs, products

    Args:
        start: starting grid position (x, y)
        heightmap: 2D obstacle map
        no_fly: 2D boolean no-fly mask
        mission_id: which mission to run
        difficulty: easy/medium/hard
        planner_id: route-layer planner name
        policy_id: mission-layer policy ("greedy" or "lookahead")
        seed: random seed
        risk_map: optional 2D risk field
        cost_map: optional 2D cost overlay for planner

    Returns:
        MissionResult with all metrics, task logs, products
    """
    from uavbench.planners import PLANNERS

    if isinstance(mission_id, str):
        mission_id = MissionID(mission_id)

    map_size = heightmap.shape[1]
    rng = np.random.default_rng(seed)
    spec, schedule = build_mission(mission_id, difficulty, map_size, seed)

    # Instantiate engine
    engine = MissionEngine(spec, rng)
    engine.set_injection_schedule(schedule)

    # Instantiate policy
    policy: MissionPolicy
    if policy_id == "lookahead":
        policy = LookaheadOPTWPolicy(depth=2)
    else:
        policy = GreedyPolicy()

    # Instantiate planner
    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")
    planner = PLANNERS[planner_id](heightmap, no_fly)

    # Risk map for cost
    if risk_map is None:
        risk_map = np.zeros_like(heightmap, dtype=np.float32)

    # ── Episode loop ──────────────────────────────────────────────
    current_pos = start
    path: list[GridPos] = []
    path_idx = 0
    total_plan_calls = 0

    while not engine.done:
        # 1. If no active path, pick next task
        if not path or path_idx >= len(path) - 1:
            target = policy.select_next_task(current_pos, engine)
            if target is None:
                # No more reachable tasks
                engine.done = True
                break

            engine.current_target = target

            # Route-layer: plan path to target
            t0 = time.perf_counter()
            plan_result = planner.plan(current_pos, target.xy, cost_map)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_plan_calls += 1

            engine.record_replan(PlanResult(
                path=plan_result.path,
                success=plan_result.success,
                compute_time_ms=elapsed_ms,
                expansions=plan_result.expansions,
            ))

            if plan_result.success and plan_result.path:
                path = list(plan_result.path)
                path_idx = 0
            else:
                # Can't reach target — skip
                target.status = target.status  # keep pending, try another
                engine.step(current_pos, {}, risk_at_pos=float(risk_map[current_pos[1], current_pos[0]]))
                continue

        # 2. Follow path one step
        if path_idx < len(path) - 1:
            path_idx += 1
            next_pos = path[path_idx]
        else:
            next_pos = current_pos

        # 3. Compute risk at new position
        nx, ny = next_pos
        risk_here = float(risk_map[ny, nx]) if (0 <= ny < risk_map.shape[0] and 0 <= nx < risk_map.shape[1]) else 0.0
        in_nfz = bool(no_fly[ny, nx]) if (0 <= ny < no_fly.shape[0] and 0 <= nx < no_fly.shape[1]) else False

        # 4. Step engine
        events = engine.step(
            next_pos,
            {"in_nfz": in_nfz},
            risk_at_pos=risk_here,
            energy_cost=1.0,
        )

        current_pos = next_pos

        # 5. Generate products on task completion
        if events.get("completions"):
            for tid in events["completions"]:
                _generate_products(engine, spec, tid, engine.step_count)

        # 6. If strict compliance and we hit NFZ, abort
        if spec.strict_compliance and events.get("violation"):
            engine.done = True
            break

        # 7. Check if current path target was completed → clear path
        if engine.current_target is None:
            path = []
            path_idx = 0

    # ── Collect results ───────────────────────────────────────────
    metrics = engine.compute_common_metrics()
    episode_log = engine.export_episode_log()

    return MissionResult(
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
    )


def _generate_products(
    engine: MissionEngine,
    spec: MissionSpec,
    task_id: str,
    step: int,
) -> None:
    """Generate operational products when a task is completed."""
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


# ── CSV/JSON export helpers ───────────────────────────────────────────

def export_products_csv(
    result: MissionResult,
    output_dir: Path,
) -> list[Path]:
    """Export all products from a MissionResult as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for product_type, rows in result.products.items():
        if not rows:
            continue
        path = output_dir / product_type
        with open(path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        written.append(path)

    return written


def export_episode_json(
    result: MissionResult,
    output_path: Path,
) -> Path:
    """Export full episode log as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "mission_id": result.mission_id,
        "difficulty": result.difficulty,
        "planner_id": result.planner_id,
        "policy_id": result.policy_id,
        "seed": result.seed,
        "success": result.success,
        "step_count": result.step_count,
        "metrics": result.metrics,
        "tasks": result.task_log,
        "products": result.products,
        "event_detections": result.event_detections,
    }
    with open(output_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    return output_path
