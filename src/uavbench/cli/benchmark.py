"""Canonical evaluation entrypoint for UAVBench benchmark results.

Canonical Evaluation Path::

    cli/benchmark.py → benchmark/runner.py → envs/urban.py

All results reported in the paper are produced exclusively through this
module's ``run_planner_once``, ``run_dynamic_episode``, and
``run_mission_episode`` functions.  The ``missions/runner_v2.py`` module
is an *experimental* demo runner and is NOT used for reported results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.metrics.operational import compute_all_metrics
from uavbench.missions.spec import MissionID, DifficultyKnobs
from uavbench.missions.engine import MissionEngine
from uavbench.missions.policies import GreedyPolicy, LookaheadOPTWPolicy
from uavbench.missions.builders import build_mission
from uavbench.planners import PLANNERS
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.base import PlanResult
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.registry import list_scenarios_by_track


# ----------------- Scenario path resolution -----------------

def scenario_path(scenario_id: str) -> Path:
    # src/uavbench/cli/benchmark.py -> parents[1] = src/uavbench
    base = Path(__file__).resolve().parents[1]
    return base / "scenarios" / "configs" / f"{scenario_id}.yaml"


def _apply_protocol_variant(cfg: Any, variant: str) -> Any:
    """Return config with deterministic ablation/protocol overrides."""
    if variant == "default":
        return cfg
    extra = dict(cfg.extra or {})
    if variant == "no_interactions":
        extra["disable_interactions"] = True
        return replace(cfg, extra=extra)
    if variant == "no_forced_breaks":
        extra["disable_forced_interdictions"] = True
        return replace(cfg, force_replan_count=0, event_t1=None, event_t2=None, extra=extra)
    if variant == "risk_only":
        extra["disable_forced_interdictions"] = True
        return replace(
            cfg,
            force_replan_count=0,
            event_t1=None,
            event_t2=None,
            fire_blocks_movement=False,
            traffic_blocks_movement=False,
            extra=extra,
        )
    if variant == "blocking_only":
        extra["disable_population_risk"] = True
        extra["enable_adversarial_uav"] = False
        return replace(cfg, extra=extra)
    if variant == "no_guardrail":
        extra["disable_feasibility_guardrail"] = True
        return replace(cfg, extra=extra)
    raise ValueError(
        "protocol_variant must be one of: default, no_interactions, no_forced_breaks, "
        "risk_only, blocking_only, no_guardrail"
    )


def _apply_stress_alpha(cfg: Any, alpha: float | None) -> Any:
    if alpha is None:
        return cfg
    a = float(np.clip(alpha, 0.0, 1.0))
    extra = dict(cfg.extra or {})

    fire_ign = max(1, int(round(max(1, int(cfg.fire_ignition_points or 1)) * (0.5 + 2.5 * a))))
    nfz_rate = float(max(0.1, cfg.nfz_expansion_rate * (0.5 + 1.5 * a)))
    veh = max(1, int(round(max(4, int(cfg.num_emergency_vehicles or 4)) * (0.5 + 1.8 * a))))
    wind_speed = float(np.clip(0.2 + 0.8 * a, 0.0, 1.0))

    extra["interaction_coupling_strength"] = float(0.4 + 1.6 * a)
    extra["interdiction_radius_scale"] = float(0.7 + 1.3 * a)
    extra["stress_alpha"] = a

    # Scale risk weights with stress (more population risk under stress)
    w_pop = float(cfg.risk_weight_population) * (0.8 + 0.4 * a)
    w_adv = float(cfg.risk_weight_adversarial) * (0.7 + 0.6 * a)
    w_smoke = float(cfg.risk_weight_smoke) * (0.6 + 0.8 * a)
    w_total = max(w_pop + w_adv + w_smoke, 0.01)
    w_pop, w_adv, w_smoke = w_pop / w_total, w_adv / w_total, w_smoke / w_total

    enable_fire = bool(cfg.enable_fire or (cfg.map_source == "osm" and a > 0.05))
    return replace(
        cfg,
        enable_fire=enable_fire,
        fire_ignition_points=fire_ign,
        nfz_expansion_rate=nfz_rate,
        num_emergency_vehicles=veh,
        wind_speed=wind_speed,
        risk_weight_population=w_pop,
        risk_weight_adversarial=w_adv,
        risk_weight_smoke=w_smoke,
        extra=extra,
    )


def _apply_config_overrides(cfg: Any, overrides: dict[str, Any] | None) -> Any:
    if not overrides:
        return cfg
    extra = dict(cfg.extra or {})
    direct: dict[str, Any] = {}
    for key, value in overrides.items():
        if hasattr(cfg, key):
            direct[key] = value
        else:
            extra[key] = value
    if extra != (cfg.extra or {}):
        direct["extra"] = extra
    return replace(cfg, **direct)


def _mean_metric(history: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in history if key in row]
    return float(np.mean(vals)) if vals else 0.0


def _compute_guardrail_depth_distribution(events: list[dict[str, Any]]) -> dict[str, float]:
    """Compute distribution of guardrail depth levels from event log."""
    guardrail_events = [e for e in events if e.get("type") == "feasibility_relaxation_applied"]
    if not guardrail_events:
        return {"depth_0": 1.0, "depth_1": 0.0, "depth_2": 0.0, "depth_3": 0.0}

    depths = [int((e.get("payload", {}) or {}).get("guardrail_depth", 0)) for e in guardrail_events]
    total = max(len(depths), 1)
    d1 = sum(1 for d in depths if d == 1)
    d2 = sum(1 for d in depths if d == 2)
    d3 = sum(1 for d in depths if d == 3)
    d0 = total - d1 - d2 - d3
    return {
        "depth_0": float(d0 / total),
        "depth_1": float(d1 / total),
        "depth_2": float(d2 / total),
        "depth_3": float(d3 / total),
    }


def _path_invalidated(
    path: list[tuple[int, int]],
    path_idx: int,
    dyn: dict[str, Any],
    lookahead: int = 6,
) -> bool:
    """Planner-agnostic path invalidation check on upcoming waypoints."""
    if not path or path_idx >= len(path):
        return False

    extra = dyn.get("forced_block_mask")
    if extra is None:
        extra = dyn.get("traffic_closure_mask")
    elif dyn.get("traffic_closure_mask") is not None:
        extra = np.asarray(extra, dtype=bool) | np.asarray(dyn["traffic_closure_mask"], dtype=bool)

    fire = dyn.get("fire_mask")
    smoke = dyn.get("smoke_mask")
    intr = dyn.get("intruder_buffer")
    mt = dyn.get("moving_target_buffer")
    nfz = dyn.get("dynamic_nfz_mask")
    if extra is None and intr is not None:
        extra = np.asarray(intr, dtype=bool)
    elif extra is not None and intr is not None:
        extra = np.asarray(extra, dtype=bool) | np.asarray(intr, dtype=bool)
    if extra is None and mt is not None:
        extra = np.asarray(mt, dtype=bool)
    elif extra is not None and mt is not None:
        extra = np.asarray(extra, dtype=bool) | np.asarray(mt, dtype=bool)
    if extra is None and nfz is not None:
        extra = np.asarray(nfz, dtype=bool)
    elif extra is not None and nfz is not None:
        extra = np.asarray(extra, dtype=bool) | np.asarray(nfz, dtype=bool)

    end = min(len(path), path_idx + 1 + max(1, lookahead))
    for x, y in path[path_idx + 1 : end]:
        if extra is not None and bool(extra[y, x]):
            return True
        if fire is not None and bool(fire[y, x]):
            return True
        if smoke is not None and float(smoke[y, x]) > 0.3:
            return True
    return False


# ----------------- Path-to-action helper -----------------

def _waypoint_action(curr_xy: tuple[int, int], next_xy: tuple[int, int]) -> int:
    """Convert consecutive (x,y) waypoints to a Discrete(6) action.

    Actions: 0=up(y-1), 1=down(y+1), 2=left(x-1), 3=right(x+1).
    """
    dx = next_xy[0] - curr_xy[0]
    dy = next_xy[1] - curr_xy[1]
    if dy == -1:
        return 0  # up
    if dy == 1:
        return 1  # down
    if dx == -1:
        return 2  # left
    if dx == 1:
        return 3  # right
    return 0  # fallback (shouldn't happen with 4-connected A*)


def _expand_execution_path(
    path: list[tuple[int, int]],
    heightmap: np.ndarray,
    no_fly: np.ndarray,
) -> list[tuple[int, int]]:
    """Expand sparse/any-angle waypoints into 4-connected executable steps."""
    if len(path) < 2:
        return list(path)

    expanded: list[tuple[int, int]] = [path[0]]
    segment_planner = AStarPlanner(heightmap, no_fly)
    for waypoint in path[1:]:
        prev = expanded[-1]
        dx = abs(waypoint[0] - prev[0])
        dy = abs(waypoint[1] - prev[1])
        if dx + dy == 1:
            expanded.append(waypoint)
            continue
        seg = segment_planner.plan(prev, waypoint)
        if seg.success and len(seg.path) >= 2:
            expanded.extend(seg.path[1:])
        else:
            expanded.append(waypoint)
    return expanded


# ----------------- Static planner run -----------------

def run_planner_once(
    scenario_id: str,
    planner_id: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Plan a static path and validate against the map (no env stepping)."""
    cfg = load_scenario(scenario_path(scenario_id))
    env = UrbanEnv(cfg)

    env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")

    planner = PLANNERS[planner_id](heightmap, no_fly)
    t0 = time.perf_counter()
    plan_result = planner.plan(start_xy, goal_xy)  # Returns PlanResult object
    planning_time = time.perf_counter() - t0
    planning_time_ms = planning_time * 1000.0
    plan_budget_ms = (
        cfg.plan_budget_dynamic_ms if cfg.paper_track == "dynamic" else cfg.plan_budget_static_ms
    )
    budget_exceeded = planning_time_ms > plan_budget_ms

    path = plan_result.path  # Extract path from PlanResult
    has_path = bool(path)
    violations = 0

    # Validate path if it exists
    if has_path:
        if path[0] != start_xy or path[-1] != goal_xy:
            violations += 1

        H, W = heightmap.shape
        for (x, y) in path:
            if not (0 <= x < W and 0 <= y < H):
                violations += 1
                break
            if bool(no_fly[y, x]):
                violations += 1
                break
            # A* baseline: buildings are obstacles
            if float(heightmap[y, x]) > 0.0:
                violations += 1
                break

    success = bool(has_path and violations == 0)

    return {
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "seed": int(seed),
        "success": success,
        "constraint_violations": int(violations),
        "path_length": int(len(path)) if success else 0,
        "path": path if success else None,
        "heightmap": heightmap,
        "no_fly": no_fly,
        "start": start_xy,
        "goal": goal_xy,
        "planning_time": planning_time,
        "planning_time_ms": planning_time_ms,
        "plan_budget_ms": float(plan_budget_ms),
        "budget_exceeded": bool(budget_exceeded),
        "map_source": cfg.map_source,
        "osm_tile_id": cfg.osm_tile_id,
        "config": cfg,
    }


# ----------------- Mission POI generation for renderer -----------------


def _compute_mission_pois(
    cfg: Any,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
) -> list[dict[str, Any]]:
    """Generate mission-specific POI list for the operational renderer.

    Returns a list of dicts with keys: xy, icon, color, label.
    """
    pois: list[dict[str, Any]] = []
    mt = getattr(cfg.mission_type, "value", str(cfg.mission_type)).lower()

    try:
        from uavbench.visualization.icons.library import IconID
    except ImportError:
        return pois

    # Only place start/goal semantic icons — dynamic zone markers are
    # rendered per-zone by the renderer from restriction_zones data.
    if "civil" in mt or "protection" in mt:
        pois.append({
            "xy": goal_xy,
            "icon": IconID.HOME,
            "color": "#00CC44",
            "label": "Evacuation point",
        })
    elif "maritime" in mt or "domain" in mt:
        pois.append({
            "xy": goal_xy,
            "icon": IconID.ANCHOR,
            "color": "#0088CC",
            "label": "Return-to-base",
        })
    elif "critical" in mt or "infra" in mt:
        pois.append({
            "xy": goal_xy,
            "icon": IconID.INSPECTION,
            "color": "#00CC88",
            "label": "Inspection site",
        })

    return pois


# ----------------- Dynamic episode run -----------------

def run_dynamic_episode(
    scenario_id: str,
    planner_id: str,
    *,
    seed: int,
    protocol_variant: str = "default",
    stress_alpha: float | None = None,
    config_overrides: dict[str, Any] | None = None,
    episode_horizon_steps: int | None = None,
    renderer: Any | None = None,
    render_dir: str = "",
    render_gif: str = "",
    render_dpi: int = 120,
) -> dict[str, Any]:
    """Run a full episode stepping through the environment.

    Unlike run_planner_once (static validation), this:
    1. Plans an initial path
    2. Steps through the env action-by-action
    3. For adaptive planners: checks should_replan each step, replans if needed
    4. For static planners: follows the initial path blindly (may get stuck)

    If ``renderer`` is an OperationalRenderer instance, each step produces a
    visualization frame automatically.  Alternatively, pass ``render_dir``
    (PNG per frame) and/or ``render_gif`` (animated GIF path) to auto-create
    a renderer.
    """
    cfg = load_scenario(scenario_path(scenario_id))
    cfg = _apply_protocol_variant(cfg, protocol_variant)
    cfg = _apply_stress_alpha(cfg, stress_alpha)
    cfg = _apply_config_overrides(cfg, config_overrides)
    env = UrbanEnv(cfg)

    env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    # Auto-create renderer if render_dir or render_gif requested
    if renderer is None and (render_dir or render_gif):
        try:
            from uavbench.visualization.operational_renderer import OperationalRenderer
            mission_pois = _compute_mission_pois(cfg, start_xy, goal_xy)

            # Build incident chips for the animated title card
            incidents: list[str] = []
            if getattr(cfg, "enable_fire", False):
                if getattr(cfg, "fire_blocks_movement", False):
                    incidents.append("FIRE BLOCKING")
                else:
                    incidents.append("FIRE ACTIVE")
            if getattr(cfg, "enable_traffic", False):
                if getattr(cfg, "traffic_blocks_movement", False):
                    incidents.append("TRAFFIC BLOCKING")
                else:
                    incidents.append("TRAFFIC ACTIVE")
            if getattr(cfg, "enable_dynamic_nfz", False):
                # Mission-typed zone chip (TFR / SAR / CORDON)
                mt = str(getattr(cfg.mission_type, "value", cfg.mission_type)).lower()
                nz = getattr(cfg, "num_nfz_zones", 3)
                if "civil" in mt:
                    incidents.append(f"TFR \u00d7{nz}" if nz > 1 else "TFR")
                elif "maritime" in mt:
                    incidents.append("SAR BOX")
                elif "critical" in mt or "infra" in mt:
                    incidents.append(f"CORDON \u00d7{nz}" if nz > 1 else "CORDON")
                else:
                    incidents.append("DYNAMIC NFZ")
            wind_val = str(getattr(cfg, "wind_level", "") or "")
            if wind_val and wind_val.lower() not in ("none", ""):
                incidents.append(f"WIND: {wind_val.upper()}")
            if getattr(cfg, "emergency_corridor_enabled", False):
                incidents.append("EMRG CORRIDOR")
            extra = dict(cfg.extra or {}) if hasattr(cfg, "extra") else {}
            cdp = extra.get("comms_dropout_prob", 0)
            if cdp and float(cdp) > 0:
                incidents.append(f"COMMS DROP {int(float(cdp) * 100)}%")

            renderer = OperationalRenderer(
                heightmap, no_fly, start_xy, goal_xy,
                planner_name=planner_id,
                mode_label=protocol_variant,
                scenario_id=scenario_id,
                mission_type=getattr(cfg.mission_type, "value", str(cfg.mission_type)),
                track=cfg.paper_track,
                dpi=render_dpi,
                landuse_map=getattr(env, '_landuse_map', None),
                roads_mask=getattr(env, '_roads_mask', None),
                osm_tile_id=cfg.osm_tile_id,
                mission_pois=mission_pois,
                active_incidents=incidents,
            )
        except ImportError:
            pass  # matplotlib not available

    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")

    planner = PLANNERS[planner_id](heightmap, no_fly)
    t0 = time.perf_counter()
    plan_result = planner.plan(start_xy, goal_xy)
    planning_time = time.perf_counter() - t0
    planning_time_ms = planning_time * 1000.0
    plan_budget_ms = (
        cfg.plan_budget_dynamic_ms if cfg.paper_track == "dynamic" else cfg.plan_budget_static_ms
    )
    initial_budget_exceeded = planning_time_ms > plan_budget_ms
    if initial_budget_exceeded:
        env.log_event(
            "plan_budget_exceeded",
            call_type="initial",
            elapsed_ms=round(float(planning_time_ms), 3),
            budget_ms=float(plan_budget_ms),
        )
    path = _expand_execution_path(list(plan_result.path), heightmap, no_fly)
    env.set_plan_info(len(path), initial_budget_exceeded, "initial")

    is_adaptive = hasattr(planner, "should_replan") and hasattr(planner, "replan")

    if (not plan_result.success) or (not path):
        return {
            "scenario": scenario_id,
            "planner": planner_id,
            "seed": int(seed),
            "success": False,
            "constraint_violations": 0,
            "path_length": 0,
            "path": None,
            "heightmap": heightmap,
            "no_fly": no_fly,
            "start": start_xy,
            "goal": goal_xy,
            "planning_time": planning_time,
            "planning_time_ms": planning_time_ms,
            "plan_budget_ms": float(plan_budget_ms),
            "initial_budget_exceeded": bool(initial_budget_exceeded),
            "map_source": cfg.map_source,
            "osm_tile_id": cfg.osm_tile_id,
            "config": cfg,
            "protocol_variant": protocol_variant,
            "episode_steps": 0,
            "total_replans": 0,
            "replan_budget_violations": 0,
            "termination_reason": (
                f"planning_failed:{plan_result.reason}"
            ),
            "events": [],
        }

    # Episode execution
    path_idx = 0
    actual_trajectory = [start_xy]
    total_reward = 0.0
    violations = 0
    max_steps = 4 * int(cfg.map_size)
    if episode_horizon_steps is not None:
        max_steps = int(max(1, episode_horizon_steps))
    stuck_counter = 0
    reached_goal = False
    collision_terminated = False
    termination_reason = "timeout"
    episode_steps = 0
    replan_budget_violations = 0
    forced_replans_seen = 0
    replan_every = int(cfg.replan_every_steps)
    max_replans = int(cfg.max_replans_per_episode)
    enforced_replans = 0
    last_replan_step = 0
    snapshot_calls = 1
    replan_steps: list[int] = []
    risk_exposure_integral = 0.0
    break_pending_step: int | None = None
    break_to_replan_steps: list[int] = []
    break_to_recover_steps: list[int] = []
    forced_block_hits = 0
    traffic_closure_hits = 0
    fire_block_hits = 0
    traffic_block_hits = 0
    target_block_hits = 0
    intruder_block_hits = 0
    dyn_nfz_block_hits = 0
    trigger_counts = {
        "path_invalidation": 0,
        "forced_event": 0,
        "cadence": 0,
        "stuck_fallback": 0,
        "planner_signal": 0,
    }

    # ── P1 realism buffers ───────────────────────────────────────────
    from collections import deque as _deque

    _constraint_latency = int(getattr(cfg, "constraint_latency_steps", 0))
    _comms_dropout_prob = float(getattr(cfg, "comms_dropout_prob", 0.0))
    _gnss_sigma = float(getattr(cfg, "gnss_noise_sigma", 0.0))

    # FIFO buffer for constraint latency (stores last N snapshots)
    _latency_buffer: _deque[dict] = _deque(maxlen=max(1, _constraint_latency + 1))
    _last_good_snapshot: dict | None = None  # for comms dropout
    _realism_rng = np.random.RandomState(int(seed) ^ 0x50_1A_C1_00)  # deterministic

    for step in range(max_steps):
        episode_steps = step + 1

        # End of current path
        if path_idx >= len(path) - 1:
            break

        action = _waypoint_action(path[path_idx], path[path_idx + 1])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        risk_exposure_integral += float(info.get("risk_cost", 0.0))
        forced_block_hits += 1 if bool(info.get("attempted_forced_block", False)) else 0
        traffic_closure_hits += 1 if bool(info.get("attempted_traffic_closure", False)) else 0
        fire_block_hits += 1 if bool(info.get("attempted_fire_block", False)) else 0
        traffic_block_hits += 1 if bool(info.get("attempted_traffic_block", False)) else 0
        target_block_hits += 1 if bool(info.get("attempted_target_block", False)) else 0
        intruder_block_hits += 1 if bool(info.get("attempted_intruder_block", False)) else 0
        dyn_nfz_block_hits += 1 if bool(info.get("attempted_nfz_block", False)) else 0

        # Check if agent actually moved to expected next waypoint
        current_pos = tuple(int(v) for v in env._agent_pos)  # (x, y, z)
        current_xy = (current_pos[0], current_pos[1])

        if current_xy == path[path_idx + 1]:
            path_idx += 1
            actual_trajectory.append(current_xy)
            stuck_counter = 0
            if break_pending_step is not None:
                break_to_recover_steps.append(episode_steps - break_pending_step)
                break_pending_step = None
        else:
            stuck_counter += 1

        # ── Operational renderer hook ────────────────────────────────
        if renderer is not None:
            dyn_snap = env.get_dynamic_state()
            prev_heading = getattr(renderer, "_last_heading", 0.0)
            if len(actual_trajectory) >= 2:
                dx = actual_trajectory[-1][0] - actual_trajectory[-2][0]
                dy = actual_trajectory[-1][1] - actual_trajectory[-2][1]
                if dx != 0 or dy != 0:
                    prev_heading = math.degrees(math.atan2(dy, dx))
            renderer._last_heading = prev_heading  # type: ignore[attr-defined]
            renderer.render_frame(
                drone_pos=current_xy,
                step=episode_steps,
                fire_mask=dyn_snap.get("fire_mask"),
                smoke_mask=dyn_snap.get("smoke_mask"),
                nfz_mask=dyn_snap.get("dynamic_nfz_mask"),
                traffic_positions=dyn_snap.get("traffic_positions"),
                traffic_closure_mask=dyn_snap.get("traffic_closure_mask"),
                risk_map=dyn_snap.get("risk_cost_map"),
                forced_block_mask=dyn_snap.get("forced_block_mask"),
                restriction_zones=dyn_snap.get("restriction_zones"),
                restriction_risk_buffer=dyn_snap.get("restriction_risk_buffer"),
                trajectory=actual_trajectory,
                planned_path=path[path_idx + 1:] if path_idx + 1 < len(path) else [],
                heading_deg=prev_heading,
                replan_flash=bool(episode_steps in replan_steps),
                replans=enforced_replans,
                risk_value=float(info.get("risk_cost", 0.0)),
                risk_integral=risk_exposure_integral,
                dynamic_block_hits=forced_block_hits,
                guardrail_depth=int(info.get("guardrail_depth", 0)),
                feasible=bool(info.get("feasible", True)),
                status_text="OK" if not terminated else "TERMINATED",
                total_steps=max_steps,
                event_t1=getattr(cfg, "event_t1", None),
                event_t2=getattr(cfg, "event_t2", None),
                plan_len=int(info.get("plan_len", 0)),
                plan_stale=bool(info.get("plan_stale", False)),
                plan_reason=str(info.get("plan_reason", "none")),
                forced_block_active=bool(info.get("forced_block_active", False)),
                forced_block_cleared=bool(info.get("forced_block_cleared_by_guardrail", False)),
            )

        if terminated:
            reached_goal = info.get("reached_goal", False)
            collision_terminated = info.get("collision_terminated", False)
            termination_reason = info.get("termination_reason", "unknown")
            break

        if truncated:
            break

        # Adaptive replanning
        # ── Replanning (all planners) ────────────────────────────────
        # Adaptive planners use their native .replan() method.
        # Non-adaptive planners get harness-level replanning via .plan()
        # so that every planner has a fair chance to recover from blocks.
        if True:  # replanning gate – always enter
            dyn_raw = env.get_dynamic_state()
            snapshot_calls += 1

            # ── P1: constraint latency (FIFO buffer) ────────────────
            _latency_buffer.append(dyn_raw)
            _snapshot_age = 0
            if _constraint_latency > 0 and len(_latency_buffer) > _constraint_latency:
                dyn = _latency_buffer[0]  # delayed snapshot
                _snapshot_age = _constraint_latency
            else:
                dyn = dyn_raw

            # ── P1: comms dropout (stale-state delivery) ────────────
            if _comms_dropout_prob > 0.0 and _realism_rng.random() < _comms_dropout_prob:
                if _last_good_snapshot is not None:
                    dyn = _last_good_snapshot  # deliver stale snapshot
                    _snapshot_age = max(_snapshot_age, episode_steps - getattr(_last_good_snapshot, "_step", episode_steps))
                # else: first step, no stale available — use current
            else:
                _last_good_snapshot = dyn

            # Build merged extra_obstacles from new dynamic layers
            extra_obs = None
            for key in (
                "moving_target_buffer",
                "intruder_buffer",
                "dynamic_nfz_mask",
                "forced_block_mask",
                "traffic_closure_mask",
            ):
                layer = dyn.get(key)
                if layer is not None:
                    extra_obs = layer if extra_obs is None else (extra_obs | layer)

            forced_now = int(info.get("forced_replans_triggered", forced_replans_seen))
            forced_event = forced_now > forced_replans_seen
            forced_replans_seen = forced_now
            if forced_event and break_pending_step is None:
                break_pending_step = episode_steps
            cadence_due = (episode_steps - last_replan_step) >= replan_every
            path_invalid = _path_invalidated(path, path_idx, dyn, lookahead=6)
            stuck_trigger = stuck_counter >= 3

            # ── P1: GNSS noise (noisy position for planner) ─────────
            if _gnss_sigma > 0.0:
                sz = int(cfg.map_size)
                nx = int(round(current_pos[0] + _realism_rng.normal(0.0, _gnss_sigma)))
                ny = int(round(current_pos[1] + _realism_rng.normal(0.0, _gnss_sigma)))
                nx = int(np.clip(nx, 0, sz - 1))
                ny = int(np.clip(ny, 0, sz - 1))
                planner_pos = (nx, ny, current_pos[2]) if len(current_pos) == 3 else (nx, ny)
                planner_xy = (nx, ny)
            else:
                planner_pos = current_pos
                planner_xy = current_xy

            planner_should = False
            planner_reason = ""
            if hasattr(planner, "should_replan"):
                try:
                    planner_should, planner_reason = planner.should_replan(
                        planner_pos,
                        dyn["fire_mask"],
                        dyn["traffic_positions"],
                        smoke_mask=dyn["smoke_mask"],
                        extra_obstacles=extra_obs,
                        risk_cost_map=dyn.get("risk_cost_map"),
                    )
                except TypeError:
                    planner_should, planner_reason = False, ""

            trigger_replan = path_invalid or forced_event or cadence_due or stuck_trigger or planner_should
            allowed_now = forced_event or cadence_due or path_invalid or stuck_trigger

            if trigger_replan and allowed_now:
                if enforced_replans >= max_replans:
                    termination_reason = "max_replans_exceeded"
                    break
                if path_invalid:
                    trigger_counts["path_invalidation"] += 1
                if forced_event:
                    trigger_counts["forced_event"] += 1
                if cadence_due:
                    trigger_counts["cadence"] += 1
                if stuck_trigger:
                    trigger_counts["stuck_fallback"] += 1
                if planner_should:
                    trigger_counts["planner_signal"] += 1

                # ── Dispatch: native replan vs harness replan ────────
                new_path: list = []
                replan_ms = 0.0
                if is_adaptive and hasattr(planner, "replan"):
                    # Native adaptive replanning
                    try:
                        t_replan = time.perf_counter()
                        new_path = planner.replan(
                            planner_pos,
                            goal_xy,
                            dyn["fire_mask"],
                            dyn["traffic_positions"],
                            planner_reason or ("path_invalidated" if path_invalid else "cadence"),
                            smoke_mask=dyn["smoke_mask"],
                            extra_obstacles=extra_obs,
                            risk_cost_map=dyn.get("risk_cost_map"),
                        )
                        replan_ms = (time.perf_counter() - t_replan) * 1000.0
                    except TypeError:
                        new_path = []
                        replan_ms = 0.0
                else:
                    # Harness-level replanning for non-adaptive planners:
                    # call planner.plan() from current position to goal.
                    try:
                        t_replan = time.perf_counter()
                        harness_result = planner.plan(planner_xy, goal_xy)
                        replan_ms = (time.perf_counter() - t_replan) * 1000.0
                        new_path = list(harness_result.path) if harness_result.success else []
                    except Exception:
                        new_path = []
                        replan_ms = 0.0

                _replan_reason = (
                    "forced_event" if forced_event else
                    "path_invalidated" if path_invalid else
                    "cadence" if cadence_due else
                    "stuck" if stuck_trigger else
                    "planner_signal"
                )
                if replan_ms > plan_budget_ms:
                    replan_budget_violations += 1
                    new_path = []
                    env.log_event(
                        "plan_budget_exceeded",
                        call_type="replan",
                        elapsed_ms=round(float(replan_ms), 3),
                        budget_ms=float(plan_budget_ms),
                    )
                    env.set_plan_info(len(path) - path_idx, True, "budget_exceeded", _snapshot_age)
                else:
                    enforced_replans += 1
                    last_replan_step = episode_steps
                    replan_steps.append(int(episode_steps))
                    if break_pending_step is not None:
                        break_to_replan_steps.append(episode_steps - break_pending_step)
                if new_path:
                    path = _expand_execution_path(list(new_path), heightmap, no_fly)
                    path_idx = 0
                    stuck_counter = 0
                    env.set_plan_info(len(path), False, _replan_reason, _snapshot_age)
            elif stuck_counter >= 10:
                # All planners: give up if stuck too long AND no replan triggered
                break

    # Count real constraint violations (NFZ, building) — fire blocks are expected
    events = env.events
    nfz_violations = sum(1 for e in events if e.get("type") == "no_fly_violation_attempt")
    building_violations = sum(1 for e in events if e.get("type") == "collision_building_attempt")
    fire_blocks = sum(1 for e in events if e.get("type") == "fire_block")
    traffic_blocks = sum(1 for e in events if e.get("type") == "traffic_block")
    target_blocks = sum(1 for e in events if e.get("type") == "target_block")
    intruder_blocks = sum(1 for e in events if e.get("type") == "intruder_block")
    nfz_dyn_blocks = sum(1 for e in events if e.get("type") == "dynamic_nfz_block")
    forced_interdiction_blocks = sum(1 for e in events if e.get("type") == "forced_interdiction_block")
    traffic_closure_blocks = sum(1 for e in events if e.get("type") == "traffic_closure_block")
    violations = nfz_violations + building_violations

    # Also check if we reached the goal position even without terminated flag
    final_xy = (int(env._agent_pos[0]), int(env._agent_pos[1]))
    if final_xy == goal_xy and not reached_goal:
        reached_goal = True

    success = reached_goal

    if is_adaptive and hasattr(planner, "get_replan_metrics"):
        replan_metrics = planner.get_replan_metrics()
    else:
        replan_metrics = {"total_replans": 0}
    dyn_state = env.get_dynamic_state()
    interaction_history = dyn_state.get("interaction_metrics_history") or []
    protocol_metrics = dyn_state.get("protocol_metrics") or {}
    guardrail_status = dyn_state.get("guardrail_status") or {}
    guardrail_events = [e for e in events if e.get("type") == "feasibility_relaxation_applied"]
    guardrail_activation_count = len(guardrail_events)
    corridor_fallback_count = sum(
        1
        for e in guardrail_events
        if bool((e.get("payload", {}) or {}).get("corridor_fallback_used", False))
    )
    relax_magnitudes: list[float] = []
    for e in guardrail_events:
        payload = (e.get("payload", {}) or {}).get("relaxation_applied", {}) or {}
        mag = (
            abs(float(payload.get("nfz_rate_delta", 0.0)))
            + abs(float(payload.get("nfz_radius_delta", 0.0)))
            + float(payload.get("closures_removed", 0)) / 100.0
            + float(payload.get("forced_blocks_cleared", 0)) / 100.0
        )
        relax_magnitudes.append(float(mag))
    relaxation_magnitude = float(np.mean(relax_magnitudes)) if relax_magnitudes else 0.0

    result = {
        "scenario": scenario_id,
        "planner": planner_id,
        "seed": int(seed),
        "success": success,
        "constraint_violations": int(violations),
        "path_length": int(len(actual_trajectory)),
        "path": actual_trajectory,
        "heightmap": heightmap,
        "no_fly": no_fly,
        "start": start_xy,
        "goal": goal_xy,
        "planning_time": planning_time,
        "planning_time_ms": planning_time_ms,
        "plan_budget_ms": float(plan_budget_ms),
        "initial_budget_exceeded": bool(initial_budget_exceeded),
        "protocol_variant": protocol_variant,
        "map_source": cfg.map_source,
        "osm_tile_id": cfg.osm_tile_id,
        "config": cfg,
        "episode_steps": episode_steps,
        "total_replans": int(enforced_replans),
        "planner_total_replans_raw": int(replan_metrics.get("total_replans", 0)),
        "replan_steps": replan_steps,
        "replan_budget_violations": int(replan_budget_violations),
        "snapshot_calls": int(snapshot_calls),
        "replan_every_steps": int(cfg.replan_every_steps),
        "max_replans_per_episode": int(cfg.max_replans_per_episode),
        "replan_contract": {
            "trigger_path_invalidation": True,
            "trigger_forced_event": True,
            "trigger_cadence": True,
            "max_replans": int(cfg.max_replans_per_episode),
        },
        "replan_mode": "native" if is_adaptive else "harness_replan",
        # P1 realism parameters
        "constraint_latency_steps": _constraint_latency,
        "comms_dropout_prob": _comms_dropout_prob,
        "gnss_noise_sigma": _gnss_sigma,
        "replan_trigger_path_invalidation_count": int(trigger_counts["path_invalidation"]),
        "replan_trigger_forced_event_count": int(trigger_counts["forced_event"]),
        "replan_trigger_cadence_count": int(trigger_counts["cadence"]),
        "replan_trigger_stuck_fallback_count": int(trigger_counts["stuck_fallback"]),
        "replan_trigger_planner_signal_count": int(trigger_counts["planner_signal"]),
        "total_reward": total_reward,
        "fire_blocks": fire_blocks,
        "traffic_blocks": traffic_blocks,
        "target_blocks": target_blocks,
        "intruder_blocks": intruder_blocks,
        "dynamic_nfz_blocks": nfz_dyn_blocks,
        "forced_interdiction_blocks": forced_interdiction_blocks,
        "traffic_closure_blocks": traffic_closure_blocks,
        "total_dynamic_blocks": (
            fire_blocks
            + traffic_blocks
            + target_blocks
            + intruder_blocks
            + nfz_dyn_blocks
            + forced_interdiction_blocks
            + traffic_closure_blocks
        ),
        "dynamic_block_hits": (
            forced_block_hits
            + traffic_closure_hits
            + fire_block_hits
            + traffic_block_hits
            + target_block_hits
            + intruder_block_hits
            + dyn_nfz_block_hits
        ),
        "risk_exposure_integral": float(risk_exposure_integral),
        "time_to_recover_after_break": float(np.mean(break_to_recover_steps)) if break_to_recover_steps else float("nan"),
        "replan_latency_after_break": float(np.mean(break_to_replan_steps)) if break_to_replan_steps else float("nan"),
        "termination_reason": termination_reason,
        "collision_terminated": collision_terminated,
        "interaction_fire_nfz_overlap_ratio": _mean_metric(
            interaction_history, "interaction_fire_nfz_overlap_ratio"
        ),
        "interaction_fire_road_closure_rate": _mean_metric(
            interaction_history, "interaction_fire_road_closure_rate"
        ),
        "interaction_congestion_risk_corr": _mean_metric(
            interaction_history, "interaction_congestion_risk_corr"
        ),
        "dynamic_block_entropy": _mean_metric(interaction_history, "dynamic_block_entropy"),
        "dynamic_block_entropy_env": _mean_metric(interaction_history, "dynamic_block_entropy_env"),
        # Restriction zone metrics
        "restriction_zone_count": int(
            len(last_dyn.get("restriction_zones", []))
            if (last_dyn := env.get_dynamic_state()) else 0
        ),
        "zone_coverage_peak": float(
            getattr(env._dynamic_nfz, "peak_coverage", 0.0)
            if hasattr(env, "_dynamic_nfz") and env._dynamic_nfz is not None else 0.0
        ),
        "zone_violation_count": int(
            getattr(env._dynamic_nfz, "zone_violations", 0)
            if hasattr(env, "_dynamic_nfz") and env._dynamic_nfz is not None else 0
        ),
        "restrictions_mode": str(getattr(cfg, "restrictions_mode", "disabled")),
        "interdiction_hit_rate": float(protocol_metrics.get("interdiction_hit_rate", 0.0)),
        "interdictions_triggered": int(protocol_metrics.get("interdictions_triggered", 0)),
        "forced_replans_triggered": int(protocol_metrics.get("forced_replans_triggered", 0)),
        "interdiction_hit_rate_reference": float(protocol_metrics.get("interdiction_hit_rate", 0.0)),
        "reachability_failed_before_relax": bool(
            guardrail_status.get("reachability_failed_before_relax", False)
        ),
        "corridor_fallback_used": bool(guardrail_status.get("corridor_fallback_used", False)),
        "feasible_after_guardrail": bool(guardrail_status.get("feasible_after_guardrail", True)),
        "relaxation_applied": guardrail_status.get("relaxation_applied", {}),
        "guardrail_activation_count": int(guardrail_activation_count),
        "guardrail_activation_rate": float(guardrail_activation_count / max(episode_steps, 1)),
        "corridor_fallback_count": int(corridor_fallback_count),
        "corridor_fallback_rate": float(corridor_fallback_count / max(guardrail_activation_count, 1)),
        "relaxation_magnitude": float(relaxation_magnitude),
        "guardrail_depth_distribution": _compute_guardrail_depth_distribution(events),
        "events": events,
    }

    # ── Post-episode: export visualization artifacts ──────────────────
    if renderer is not None:
        if render_dir:
            frame_path = Path(render_dir)
            frame_path.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(renderer._frames):
                plt_path = frame_path / f"frame_{i:05d}.png"
                try:
                    import matplotlib.pyplot as _mplt
                    _mplt.imsave(str(plt_path), frame)
                except Exception:
                    pass
            # Export event-driven keyframes at high DPI
            try:
                kf_dir = frame_path / "keyframes"
                renderer.export_keyframes(kf_dir, dpi=max(200, render_dpi))
            except Exception:
                pass
        if render_gif:
            try:
                renderer.export_gif(Path(render_gif))
            except Exception:
                pass

    return result


# ----------------- Mission episode run (multi-task) -----------------

def run_mission_episode(
    scenario_id: str,
    planner_id: str,
    *,
    seed: int,
    policy_id: str = "greedy",
    protocol_variant: str = "default",
    stress_alpha: float | None = None,
    config_overrides: dict[str, Any] | None = None,
    episode_horizon_steps: int | None = None,
) -> dict[str, Any]:
    """Run a multi-task mission episode using MissionEngine.

    This extends ``run_dynamic_episode`` by integrating the mission layer:
    task sequencing, injection, utility scoring, and operational products.

    The environment still handles dynamics (fire, traffic, NFZ) exactly as
    before; the mission layer adds multi-waypoint task selection on top.

    Backward compatible: can be called from CLI with ``--mission`` flag;
    existing ``run_dynamic_episode`` behaviour is unchanged.

    Parameters
    ----------
    scenario_id : str
        Scenario identifier (e.g. 'gov_civil_protection_hard').
    planner_id : str
        Planner registry key.
    seed : int
        Random seed for reproducibility.
    policy_id : str
        Mission-layer policy: 'greedy' or 'lookahead'.
    protocol_variant, stress_alpha, config_overrides, episode_horizon_steps :
        Same semantics as ``run_dynamic_episode``.

    Returns
    -------
    dict
        Superset of ``run_dynamic_episode`` keys, plus mission-specific:
        ``mission_score``, ``task_completion_rate``, ``task_log``,
        ``products``, ``mission_id``, ``difficulty``, ``policy_id``.
    """
    cfg = load_scenario(scenario_path(scenario_id))
    cfg = _apply_protocol_variant(cfg, protocol_variant)
    cfg = _apply_stress_alpha(cfg, stress_alpha)
    cfg = _apply_config_overrides(cfg, config_overrides)
    env = UrbanEnv(cfg)

    env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")

    # ── Resolve mission parameters from scenario YAML extra field ────
    extra = dict(cfg.extra or {}) if hasattr(cfg, "extra") else {}
    mission_bank = extra.get("mission_bank", "")

    # Map mission_bank string to MissionID
    _bank_to_id = {
        "civil_protection": MissionID.CIVIL_PROTECTION,
        "maritime_domain": MissionID.MARITIME_DOMAIN,
        "critical_infrastructure": MissionID.CRITICAL_INFRASTRUCTURE,
    }
    mission_type_str = getattr(cfg.mission_type, "value", str(cfg.mission_type)).lower()
    mission_id = _bank_to_id.get(
        mission_bank,
        _bank_to_id.get(mission_type_str, MissionID.CIVIL_PROTECTION),
    )

    difficulty_raw = str(cfg.difficulty).lower()
    difficulty = difficulty_raw if difficulty_raw in ("easy", "medium", "hard") else "easy"

    # ── Build mission spec + injection schedule ──────────────────────
    rng = np.random.default_rng(seed)
    map_size = int(cfg.map_size)
    spec, schedule = build_mission(mission_id, difficulty, map_size, seed)

    engine = MissionEngine(spec, rng)
    engine.set_injection_schedule(schedule)

    # ── Instantiate policy ───────────────────────────────────────────
    if policy_id == "lookahead":
        policy = LookaheadOPTWPolicy(depth=2)
    else:
        policy = GreedyPolicy()

    # ── Instantiate planner ──────────────────────────────────────────
    planner_fn = PLANNERS[planner_id]
    planner = planner_fn(heightmap, no_fly)
    is_adaptive = hasattr(planner, "should_replan") and hasattr(planner, "replan")

    # ── P1 realism buffers ───────────────────────────────────────────
    from collections import deque as _deque

    _constraint_latency = int(getattr(cfg, "constraint_latency_steps", 0))
    _comms_dropout_prob = float(getattr(cfg, "comms_dropout_prob", 0.0))
    _gnss_sigma = float(getattr(cfg, "gnss_noise_sigma", 0.0))
    _latency_buffer: _deque[dict] = _deque(maxlen=max(1, _constraint_latency + 1))
    _last_good_snapshot: dict | None = None
    _realism_rng = np.random.RandomState(int(seed) ^ 0x50_1A_C1_00)

    plan_budget_ms = (
        cfg.plan_budget_dynamic_ms if cfg.paper_track == "dynamic" else cfg.plan_budget_static_ms
    )

    # ── Episode state ────────────────────────────────────────────────
    current_xy = start_xy
    path: list[tuple[int, int]] = []
    path_idx = 0
    actual_trajectory = [start_xy]
    total_reward = 0.0
    violations = 0
    max_steps = episode_horizon_steps or spec.knobs.time_budget or 4 * map_size
    episode_steps = 0
    total_replans = 0
    replan_budget_violations = 0
    replan_steps: list[int] = []
    risk_exposure_integral = 0.0
    stuck_counter = 0
    reached_goal = False
    collision_terminated = False
    termination_reason = "timeout"
    segment_replans = 0
    total_plan_time_ms = 0.0
    total_expansions = 0

    # ── Episode loop ─────────────────────────────────────────────────
    for step in range(max_steps):
        episode_steps = step + 1

        # 1. If no active path, pick next task from mission policy
        need_new_target = (not path) or (path_idx >= len(path) - 1)

        if need_new_target:
            target = policy.select_next_task(current_xy, engine)
            if target is None:
                # No more reachable tasks — episode complete
                termination_reason = "all_tasks_done" if engine.done else "no_reachable_tasks"
                break

            engine.current_target = target

            # Route-layer: plan path to target task
            t0 = time.perf_counter()
            plan_result = planner.plan(current_xy, target.xy)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_plan_time_ms += elapsed_ms
            total_expansions += plan_result.expansions

            if elapsed_ms > plan_budget_ms:
                replan_budget_violations += 1

            engine.record_replan(PlanResult(
                path=plan_result.path,
                success=plan_result.success,
                compute_time_ms=elapsed_ms,
                expansions=plan_result.expansions,
            ))

            if plan_result.success and plan_result.path:
                path = _expand_execution_path(list(plan_result.path), heightmap, no_fly)
                path_idx = 0
                segment_replans = 0
            else:
                # Can't reach target — step engine, try next task
                risk_map = env.get_dynamic_state().get("risk_cost_map")
                risk_at = 0.0
                if risk_map is not None:
                    x, y = current_xy
                    if 0 <= y < risk_map.shape[0] and 0 <= x < risk_map.shape[1]:
                        risk_at = float(risk_map[y, x])
                engine.step(current_xy, {}, risk_at_pos=risk_at)
                continue

        # 2. Follow path one step
        if path_idx < len(path) - 1:
            action = _waypoint_action(path[path_idx], path[path_idx + 1])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            risk_exposure_integral += float(info.get("risk_cost", 0.0))

            current_pos = tuple(int(v) for v in env._agent_pos)
            current_xy = (current_pos[0], current_pos[1])

            if current_xy == path[path_idx + 1]:
                path_idx += 1
                actual_trajectory.append(current_xy)
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            # At end of path — step env with hover action
            obs, reward, terminated, truncated, info = env.step(0)
            total_reward += reward
            risk_exposure_integral += float(info.get("risk_cost", 0.0))

        # 3. Step mission engine
        risk_map = env.get_dynamic_state().get("risk_cost_map")
        risk_at = 0.0
        if risk_map is not None:
            x, y = current_xy
            if 0 <= y < risk_map.shape[0] and 0 <= x < risk_map.shape[1]:
                risk_at = float(risk_map[y, x])

        in_nfz = bool(no_fly[current_xy[1], current_xy[0]]) if (
            0 <= current_xy[1] < no_fly.shape[0] and 0 <= current_xy[0] < no_fly.shape[1]
        ) else False

        events = engine.step(
            current_xy,
            {"in_nfz": in_nfz},
            risk_at_pos=risk_at,
            energy_cost=1.0,
        )

        # 4. Generate products on task completion
        if events.get("completions"):
            from uavbench.missions.runner import _generate_products
            for tid in events["completions"]:
                _generate_products(engine, spec, tid, engine.step_count)
            # Clear path so next iteration picks a new task
            path = []
            path_idx = 0

        # 5. Check strict compliance
        if spec.strict_compliance and events.get("violation"):
            termination_reason = "strict_compliance_violation"
            break

        if terminated:
            reached_goal = info.get("reached_goal", False)
            collision_terminated = info.get("collision_terminated", False)
            termination_reason = info.get("termination_reason", "unknown")
            break

        if truncated:
            termination_reason = "truncated"
            break

        # 6. Mid-segment replanning (same logic as run_dynamic_episode)
        if path and path_idx < len(path) - 1:
            dyn_raw = env.get_dynamic_state()

            # P1: constraint latency
            _latency_buffer.append(dyn_raw)
            if _constraint_latency > 0 and len(_latency_buffer) > _constraint_latency:
                dyn = _latency_buffer[0]
            else:
                dyn = dyn_raw

            # P1: comms dropout
            if _comms_dropout_prob > 0.0 and _realism_rng.random() < _comms_dropout_prob:
                if _last_good_snapshot is not None:
                    dyn = _last_good_snapshot
            else:
                _last_good_snapshot = dyn

            path_invalid = _path_invalidated(path, path_idx, dyn, lookahead=6)

            if path_invalid or stuck_counter >= 3:
                # P1: GNSS noise
                if _gnss_sigma > 0.0:
                    sz = map_size
                    nx = int(round(current_xy[0] + _realism_rng.normal(0.0, _gnss_sigma)))
                    ny = int(round(current_xy[1] + _realism_rng.normal(0.0, _gnss_sigma)))
                    nx = int(np.clip(nx, 0, sz - 1))
                    ny = int(np.clip(ny, 0, sz - 1))
                    planner_xy = (nx, ny)
                else:
                    planner_xy = current_xy

                # Replan to current target
                target_xy = engine.current_target.xy if engine.current_target else goal_xy
                t0 = time.perf_counter()
                if is_adaptive and hasattr(planner, "replan"):
                    extra_obs = None
                    for key in ("moving_target_buffer", "intruder_buffer",
                                "dynamic_nfz_mask", "forced_block_mask", "traffic_closure_mask"):
                        layer = dyn.get(key)
                        if layer is not None:
                            extra_obs = layer if extra_obs is None else (extra_obs | layer)
                    try:
                        pos_3d = (planner_xy[0], planner_xy[1], 0)
                        new_path = planner.replan(
                            pos_3d, target_xy,
                            dyn.get("fire_mask"), dyn.get("traffic_positions"),
                            "path_invalidated" if path_invalid else "stuck",
                            smoke_mask=dyn.get("smoke_mask"),
                            extra_obstacles=extra_obs,
                            risk_cost_map=dyn.get("risk_cost_map"),
                        )
                    except TypeError:
                        new_path = []
                else:
                    harness_result = planner.plan(planner_xy, target_xy)
                    new_path = list(harness_result.path) if harness_result.success else []

                replan_ms = (time.perf_counter() - t0) * 1000.0
                total_plan_time_ms += replan_ms

                if replan_ms <= plan_budget_ms and new_path:
                    path = _expand_execution_path(list(new_path), heightmap, no_fly)
                    path_idx = 0
                    total_replans += 1
                    segment_replans += 1
                    replan_steps.append(episode_steps)
                    stuck_counter = 0
                elif replan_ms > plan_budget_ms:
                    replan_budget_violations += 1

        if stuck_counter >= 10:
            termination_reason = "stuck"
            break

        if engine.done:
            termination_reason = "mission_complete"
            break

    # ── Collect mission metrics ──────────────────────────────────────
    mission_metrics = engine.compute_common_metrics()
    episode_log = engine.export_episode_log()

    # Count constraint violations
    env_events = env.events
    nfz_violations = sum(1 for e in env_events if e.get("type") == "no_fly_violation_attempt")
    building_violations = sum(1 for e in env_events if e.get("type") == "collision_building_attempt")
    violations = nfz_violations + building_violations

    final_xy = (int(env._agent_pos[0]), int(env._agent_pos[1]))

    result = {
        # Core fields (compatible with run_dynamic_episode)
        "scenario": scenario_id,
        "planner": planner_id,
        "seed": int(seed),
        "success": mission_metrics.get("task_completion_rate", 0) > 0,
        "constraint_violations": violations,
        "path_length": len(actual_trajectory),
        "path": actual_trajectory,
        "heightmap": heightmap,
        "no_fly": no_fly,
        "start": start_xy,
        "goal": goal_xy,
        "planning_time": total_plan_time_ms / 1000.0,
        "planning_time_ms": total_plan_time_ms,
        "plan_budget_ms": float(plan_budget_ms),
        "protocol_variant": protocol_variant,
        "map_source": cfg.map_source,
        "osm_tile_id": cfg.osm_tile_id,
        "config": cfg,
        "episode_steps": episode_steps,
        "total_replans": total_replans,
        "replan_steps": replan_steps,
        "replan_budget_violations": replan_budget_violations,
        "risk_exposure_integral": risk_exposure_integral,
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "collision_terminated": collision_terminated,
        "events": env_events,
        # P1 realism
        "constraint_latency_steps": _constraint_latency,
        "comms_dropout_prob": _comms_dropout_prob,
        "gnss_noise_sigma": _gnss_sigma,
        "replan_mode": "native" if is_adaptive else "harness_replan",
        # ── Mission-specific fields ──────────────────────────────────
        "mission_id": spec.mission_id.value,
        "difficulty": difficulty,
        "policy_id": policy_id,
        "mission_score": mission_metrics.get("mission_score", 0.0),
        "task_completion_rate": mission_metrics.get("task_completion_rate", 0.0),
        "completion_time": mission_metrics.get("completion_time", 0),
        "energy_used": mission_metrics.get("energy_used", 0.0),
        "mission_risk_integral": mission_metrics.get("risk_integral", 0.0),
        "mission_violation_count": mission_metrics.get("violation_count", 0),
        "replanning_cost_ms": mission_metrics.get("replanning_cost_ms", 0.0),
        "replanning_count": mission_metrics.get("replanning_count", 0),
        "replanning_expansions": mission_metrics.get("replanning_expansions", 0),
        "robustness": mission_metrics.get("robustness", 0.0),
        "product_latency": mission_metrics.get("product_latency", 0.0),
        "mission_metrics": mission_metrics,
        "task_log": episode_log.get("tasks", []),
        "segment_log": episode_log.get("segments", []),
        "products": episode_log.get("products", {}),
        "event_detections": episode_log.get("event_detections", []),
    }

    return result


# ----------------- Metrics aggregation -----------------

def aggregate(results: list[dict[str, Any]], _metric_ids: list[str] | None = None) -> dict[str, float]:
    """Aggregate per-trial results into metrics.

    Uses compute_all_metrics() for per-trial operational metrics, then
    averages across trials.
    """
    if not results:
        return {}

    # Per-trial operational metrics
    per_trial_metrics = [compute_all_metrics(r) for r in results]

    out: dict[str, float] = {}

    # Core metrics (always computed for backward compat)
    successes = np.array([1.0 if r["success"] else 0.0 for r in results], dtype=float)
    out["success_rate"] = float(successes.mean())

    # Collision and timeout rates (dynamic episodes)
    collision_flags = [r.get("collision_terminated", False) for r in results]
    if any(collision_flags):
        out["collision_rate"] = float(np.mean([1.0 if c else 0.0 for c in collision_flags]))

    timeout_flags = [r.get("termination_reason") == "timeout" for r in results]
    if any(timeout_flags):
        out["timeout_rate"] = float(np.mean([1.0 if t else 0.0 for t in timeout_flags]))

    lengths = np.array([float(r["path_length"]) for r in results if r["success"]], dtype=float)
    out["avg_path_length"] = float(lengths.mean()) if len(lengths) else float("nan")

    violations = np.array([float(r["constraint_violations"]) for r in results], dtype=float)
    out["avg_constraint_violations"] = float(violations.mean())

    # Replanning metrics (if present)
    replans = [r.get("total_replans", 0) for r in results]
    if any(r > 0 for r in replans):
        out["avg_replans"] = round(float(np.mean(replans)), 1)

    ep_steps = [r.get("episode_steps", 0) for r in results]
    if any(s > 0 for s in ep_steps):
        out["avg_episode_steps"] = round(float(np.mean(ep_steps)), 0)

    # Dynamic blocking metrics
    dyn_blocks = [r.get("total_dynamic_blocks", 0) for r in results]
    if any(b > 0 for b in dyn_blocks):
        out["avg_dynamic_blocks"] = round(float(np.mean(dyn_blocks)), 1)

    f_blocks = [r.get("fire_blocks", 0) for r in results]
    if any(f > 0 for f in f_blocks):
        out["avg_fire_blocks"] = round(float(np.mean(f_blocks)), 1)

    t_blocks = [r.get("traffic_blocks", 0) for r in results]
    if any(t > 0 for t in t_blocks):
        out["avg_traffic_blocks"] = round(float(np.mean(t_blocks)), 1)

    dyn_hits = [float(r.get("dynamic_block_hits", 0.0)) for r in results]
    if any(v > 0 for v in dyn_hits):
        out["avg_dynamic_block_hits"] = float(np.mean(dyn_hits))

    risk_vals = [float(r.get("risk_exposure_integral", 0.0)) for r in results]
    if any(v > 0.0 for v in risk_vals):
        out["avg_risk_exposure_integral"] = float(np.mean(risk_vals))

    recover_vals = [float(r["time_to_recover_after_break"]) for r in results if "time_to_recover_after_break" in r and not np.isnan(float(r["time_to_recover_after_break"]))]
    if recover_vals:
        out["avg_time_to_recover_after_break"] = float(np.mean(recover_vals))

    repl_lat_vals = [float(r["replan_latency_after_break"]) for r in results if "replan_latency_after_break" in r and not np.isnan(float(r["replan_latency_after_break"]))]
    if repl_lat_vals:
        out["avg_replan_latency_after_break"] = float(np.mean(repl_lat_vals))

    budget_flags = [1.0 if r.get("initial_budget_exceeded", False) else 0.0 for r in results]
    if any(budget_flags):
        out["plan_budget_violation_rate"] = float(np.mean(budget_flags))

    replan_budget = [float(r.get("replan_budget_violations", 0)) for r in results]
    if any(v > 0 for v in replan_budget):
        out["avg_replan_budget_violations"] = float(np.mean(replan_budget))

    for metric_key in (
        "interaction_fire_nfz_overlap_ratio",
        "interaction_fire_road_closure_rate",
        "interaction_congestion_risk_corr",
        "dynamic_block_entropy",
        "interdiction_hit_rate",
        "interdiction_hit_rate_reference",
    ):
        vals = [float(r[metric_key]) for r in results if metric_key in r]
        if vals:
            out[f"avg_{metric_key}"] = float(np.mean(vals))

    guardrail_rate = [float(r.get("guardrail_activation_rate", 0.0)) for r in results]
    if any(v > 0.0 for v in guardrail_rate):
        out["avg_guardrail_activation_rate"] = float(np.mean(guardrail_rate))
    corridor_rate = [float(r.get("corridor_fallback_rate", 0.0)) for r in results]
    if any(v > 0.0 for v in corridor_rate):
        out["avg_corridor_fallback_rate"] = float(np.mean(corridor_rate))
    relax_mag = [float(r.get("relaxation_magnitude", 0.0)) for r in results]
    if any(v > 0.0 for v in relax_mag):
        out["avg_relaxation_magnitude"] = float(np.mean(relax_mag))

    # Operational metrics (averaged across trials)
    all_keys: set[str] = set()
    for m in per_trial_metrics:
        all_keys.update(m.keys())

    for key in sorted(all_keys):
        vals = [m[key] for m in per_trial_metrics if key in m]
        if vals:
            out[f"avg_{key}"] = round(float(np.mean(vals)), 4)

    return out


# ----------------- CLI -----------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="uavbench",
        description="Run UAVBench benchmark experiments.",
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        default="urban_easy",
        help="Comma-separated list of scenario IDs to run (e.g. urban_easy,urban_medium).",
    )
    parser.add_argument(
        "--planners",
        type=str,
        default="astar",
        help="Comma-separated list of planner IDs to run (e.g. astar,periodic_replan,grid_mppi).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="path_length,success_rate,constraint_violations",
        help="Comma-separated list of metrics to compute "
             "(e.g. path_length,success_rate,constraint_violations).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per (scenario, planner) pair.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--play",
        type=str,
        default="",
        choices=["", "best", "worst"],
        help="Open a window to play the best or worst successful path (per scenario/planner).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Playback speed for --play (frames per second).",
    )
    parser.add_argument(
        "--save-videos",
        type=str,
        default="",
        choices=["", "best", "worst", "both"],
        help="Save best/worst/both successful paths as MP4/GIF to 'videos/' directory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on any exception (V&V mode).",
    )
    parser.add_argument(
        "--with-dynamics",
        action="store_true",
        help="Include fire/traffic dynamics overlays in visualizations (requires OSM tile).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for optional benchmark artifacts/reports.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save per-run summary JSON into --output-dir.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-run summary CSV into --output-dir.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Optional cap on dynamic episode steps (0 keeps scenario default).",
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=0,
        help="Reserved for oracle-regret protocol compatibility.",
    )
    parser.add_argument(
        "--oracle-planner",
        type=str,
        default="oracle",
        help="Reserved oracle planner identifier for metadata/reporting.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Tag run as deterministic in saved metadata.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print scenario registry and exit.",
    )
    parser.add_argument(
        "--list-planners",
        action="store_true",
        help="Print available planners and exit.",
    )
    parser.add_argument(
        "--strict-scenario-validation",
        action="store_true",
        help="Fail immediately when scenario validation fails.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="all",
        choices=["all", "static", "dynamic"],
        help="Filter scenarios by paper track. If not 'all', replaces --scenarios list.",
    )
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Enable fixed paper scoring protocol labels in output.",
    )
    parser.add_argument(
        "--protocol-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "no_interactions",
            "no_forced_breaks",
            "risk_only",
            "blocking_only",
            "no_guardrail",
        ],
        help="Protocol/ablation variant (affects dynamic runs).",
    )
    parser.add_argument(
        "--render-frames",
        type=str,
        default="",
        help="Directory to export per-step PNG frames for dynamic episodes (e.g. 'frames/').",
    )
    parser.add_argument(
        "--render-gif",
        type=str,
        default="",
        help="Path to export animated GIF of dynamic episodes (e.g. 'episode.gif').",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=120,
        help="DPI for rendered visualization frames (default: 120).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="nav",
        choices=["nav", "mission"],
        help="Execution mode: 'nav' for navigation (default), 'mission' for multi-task mission episodes.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="greedy",
        choices=["greedy", "lookahead"],
        help="Mission-layer task-selection policy (only used with --mode mission).",
    )

    args = parser.parse_args()

    if args.list_scenarios:
        from uavbench.scenarios.registry import print_scenario_registry
        print_scenario_registry()
        return
    if args.list_planners:
        from uavbench.planners import PAPER_PLANNERS, DEPRECATED_ALIASES, DEPRECATED_PLANNERS
        print("Paper-suite planners:", ", ".join(PAPER_PLANNERS))
        deprecated_keys = sorted(set(DEPRECATED_ALIASES) | DEPRECATED_PLANNERS)
        print(f"Deprecated (removal planned v2.0): {', '.join(deprecated_keys)}")
        return

    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if args.track in {"static", "dynamic"}:
        scenario_ids = list_scenarios_by_track(args.track)
    planner_ids = [p.strip() for p in args.planners.split(",") if p.strip()]
    metric_ids = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print(
        "[UAVBench] "
        f"scenarios={scenario_ids}, "
        f"planners={planner_ids}, "
        f"metrics={metric_ids}, "
        f"trials={args.trials}, "
        f"seed_base={args.seed_base}, "
        f"track={args.track}, "
        f"mode={args.mode}"
    )

    # Run grid experiment
    for scenario_id in scenario_ids:
        sp = scenario_path(scenario_id)
        if not sp.exists():
            raise FileNotFoundError(f"Scenario not found: {sp}")

        # Check if scenario needs dynamic episode execution
        cfg = load_scenario(sp)
        use_dynamic = (cfg.paper_track == "dynamic"
                       or cfg.fire_blocks_movement or cfg.traffic_blocks_movement
                       or cfg.enable_moving_target or cfg.enable_intruders
                       or cfg.enable_dynamic_nfz
                       or int(cfg.force_replan_count) > 0)

        for planner_id in planner_ids:
            per_trial: list[dict[str, Any]] = []

            for t in range(args.trials):
                seed = args.seed_base + (hash(scenario_id) & 0xFFFF) + (hash(planner_id) & 0x0FFF) + t
                try:
                    if args.mode == "mission":
                        r = run_mission_episode(
                            scenario_id,
                            planner_id,
                            seed=seed,
                            policy_id=args.policy,
                            protocol_variant=args.protocol_variant,
                        )
                    elif use_dynamic:
                        # Build per-trial render paths if requested
                        _rd = ""
                        _rg = ""
                        if args.render_frames:
                            _rd = str(Path(args.render_frames) / f"{scenario_id}_{planner_id}_s{seed}")
                        if args.render_gif:
                            _rg = str(Path(args.render_gif).parent / f"{scenario_id}_{planner_id}_s{seed}.gif") if args.trials > 1 else args.render_gif
                        r = run_dynamic_episode(
                            scenario_id,
                            planner_id,
                            seed=seed,
                            protocol_variant=args.protocol_variant,
                            render_dir=_rd,
                            render_gif=_rg,
                            render_dpi=args.render_dpi,
                        )
                    else:
                        r = run_planner_once(scenario_id, planner_id, seed=seed)
                except Exception as e:
                    if args.fail_fast:
                        raise
                    r = {
                        "scenario": scenario_id,
                        "planner": planner_id,
                        "seed": int(seed),
                        "success": False,
                        "constraint_violations": 1,
                        "path_length": 0,
                        "path": None,
                        "heightmap": None,
                        "no_fly": None,
                        "start": None,
                        "goal": None,
                        "planning_time": 0.0,
                        "error": str(e),
                    }
                per_trial.append(r)

            # Print metrics
            metrics = aggregate(per_trial, metric_ids)

            print("\n------------------------------")
            print(f"Scenario: {scenario_id}")
            print(f"Planner : {planner_id}")
            print(f"Trials  : {args.trials}")
            if args.paper_protocol:
                print(
                    "Protocol: paper_v1 "
                    f"({'track='+args.track if args.track!='all' else 'track=mixed'}, "
                    f"variant={args.protocol_variant})"
                )
            if use_dynamic:
                print(f"Mode    : dynamic episode")
            if args.mode == "mission":
                print(f"Mode    : mission (policy={args.policy})")
            for k, v in metrics.items():
                if isinstance(v, float) and np.isnan(v):
                    print(f"{k:>24}: n/a")
                else:
                    print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")
            print("------------------------------")

            # Optional artifact export
            if args.save_json or args.save_csv:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                base_name = f"{scenario_id}_{planner_id}"
                if args.save_json:
                    json_path = out_dir / f"{base_name}_summary.json"
                    json_path.write_text(json.dumps({
                        "scenario": scenario_id,
                        "planner": planner_id,
                        "trials": args.trials,
                        "seed_base": args.seed_base,
                        "deterministic": bool(args.deterministic),
                        "oracle_horizon": int(args.oracle_horizon),
                        "oracle_planner": args.oracle_planner,
                        "metrics": metrics,
                    }, indent=2), encoding="utf-8")
                if args.save_csv:
                    csv_path = out_dir / f"{base_name}_summary.csv"
                    with csv_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["metric", "value"])
                        for k, v in sorted(metrics.items()):
                            writer.writerow([k, v])

            # AFTER TRIALS: visualization (play / save-videos)
            successful = [r for r in per_trial if r.get("success", False)]
            want_viz = args.play or args.save_videos

            if want_viz and successful:
                try:
                    from uavbench.viz.player import play_path_window, save_path_video
                except ModuleNotFoundError as e:
                    print(
                        f"\n[ERROR] Cannot load visualization: {e}\n"
                        "  Install: pip install uavbench[viz]\n"
                    )
                    return

                # Determine which paths to visualize
                to_visualize: dict[str, dict] = {}
                if args.play in ("best",) or args.save_videos in ("best", "both"):
                    to_visualize["best"] = min(successful, key=lambda r: r["path_length"])
                if args.play == "worst" or args.save_videos in ("worst", "both"):
                    to_visualize["worst"] = max(successful, key=lambda r: r["path_length"])

                for vis_type, chosen in to_visualize.items():
                    title = (
                        f"{scenario_id} - {planner_id} - {vis_type.upper()} "
                        f"(seed={chosen['seed']}, len={chosen['path_length']})"
                    )

                    # Prepare dynamics overlays if requested
                    dynamics_kwargs: dict[str, Any] = {}
                    if args.with_dynamics and chosen.get("map_source") == "osm":
                        try:
                            from uavbench.viz.dynamics_sim import simulate_dynamics_along_path
                            tile_path = Path(__file__).resolve().parents[3] / "data" / "maps" / f"{chosen['osm_tile_id']}.npz"
                            viz_cfg = chosen["config"]
                            sim = simulate_dynamics_along_path(
                                tile_path=tile_path,
                                path=chosen["path"],
                                enable_fire=viz_cfg.enable_fire,
                                enable_traffic=viz_cfg.enable_traffic,
                                fire_ignition_points=viz_cfg.fire_ignition_points,
                                wind_direction=viz_cfg.wind_direction,
                                wind_speed=viz_cfg.wind_speed,
                                num_vehicles=viz_cfg.num_emergency_vehicles,
                                seed=chosen["seed"],
                            )
                            dynamics_kwargs = {
                                "fire_states": sim["fire_states"],
                                "burned_states": sim["burned_states"],
                                "traffic_states": sim["traffic_states"],
                                "roads_mask": sim["roads_mask"],
                                "risk_map": sim["risk_map"],
                            }
                        except Exception as e:
                            print(f"[WARNING] Dynamics overlay failed: {e}")

                    # Play in window
                    if args.play and vis_type == args.play:
                        play_path_window(
                            chosen["heightmap"], chosen["no_fly"],
                            chosen["start"], chosen["goal"], chosen["path"],
                            title=title, fps=args.fps,
                            **dynamics_kwargs,
                        )

                    # Save video
                    if args.save_videos:
                        video_dir = Path("videos")
                        video_name = f"{scenario_id}_{planner_id}_{vis_type}_seed{chosen['seed']}.mp4"
                        save_path_video(
                            chosen["heightmap"], chosen["no_fly"],
                            chosen["start"], chosen["goal"], chosen["path"],
                            video_dir / video_name,
                            title=title, fps=args.fps,
                            **dynamics_kwargs,
                        )


if __name__ == "__main__":
    main()
