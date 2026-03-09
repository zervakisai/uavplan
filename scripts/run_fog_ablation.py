#!/usr/bin/env python3
"""Limited-Visibility ablation study for IEEE RA-L paper.

Compares fog-ON (partial observability, sensor_radius=50) vs fog-OFF
(full visibility) across all planners and scenarios.

  5 planners x 3 scenarios x 10 seeds x 2 conditions = 300 episodes

Uses ProcessPoolExecutor for parallel execution.

Usage:
    python scripts/run_fog_ablation.py                # default 10 seeds
    python scripts/run_fog_ablation.py --seeds 30     # 30 seeds
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import fields, replace
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/ablation_results"
MAX_WORKERS = 6

MEDIUM_SCENARIO_IDS = [
    "osm_penteli_pharma_delivery_medium",
    "osm_piraeus_flood_rescue_medium",
    "osm_downtown_fire_surveillance_medium",
]

ALL_PLANNERS = ["astar", "periodic_replan", "aggressive_replan", "dstar_lite", "apf"]

CSV_COLUMNS = [
    "scenario_id", "planner_id", "seed", "fog_condition",
    "success", "path_length", "executed_steps", "replans",
    "computation_time_ms",
]


# ---------------------------------------------------------------------------
# Worker function (module-level for pickling with spawn)
# ---------------------------------------------------------------------------


def _run_episode_worker(args: tuple) -> dict:
    """Worker: run a single (config, planner, seed) episode.

    Uses the full runner logic (fog filter, risk cost map, mission POI,
    energy budget) rather than the simplified inline loop, to ensure
    the fog condition is faithfully applied.

    Args is a tuple: (config_dict, planner_id, seed, fog_label, scenario_id)
    """
    import time as _time
    import traceback
    from dataclasses import fields as _fields

    config_dict, planner_id, seed, fog_label, scenario_id = args

    # Local imports for clean subprocess
    from uavbench.blocking import compute_risk_cost_map
    from uavbench.dynamics.limited_visibility import LimitedVisibility
    from uavbench.envs.base import TerminationReason
    from uavbench.envs.urban import UrbanEnvV2
    from uavbench.metrics.compute import compute_episode_metrics
    from uavbench.planners import PLANNERS
    from uavbench.scenarios.schema import ScenarioConfig

    try:
        # Reconstruct config from dict
        valid_fields = {f.name for f in _fields(ScenarioConfig)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = ScenarioConfig(**filtered)

        t0 = _time.perf_counter()

        # Create environment
        env = UrbanEnvV2(config)
        obs, info = env.reset(seed=seed)
        heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

        # Mission POI (MC-2)
        mission_poi = info.get("objective_poi", goal_xy)
        mission_task_done = (mission_poi == goal_xy)

        # Create planner
        planner_cls = PLANNERS[planner_id]
        planner = planner_cls(heightmap, no_fly, config)
        planner.set_seed(seed)

        # Limited visibility (LV-1)
        fog = (
            LimitedVisibility((config.map_size, config.map_size), config.sensor_radius)
            if config.enable_limited_visibility
            else None
        )

        # Initial risk cost map
        initial_dyn = env.get_dynamic_state()
        if fog is not None:
            initial_dyn = fog.observe(start_xy, initial_dyn, 0)
        risk_map = compute_risk_cost_map(heightmap, no_fly, config, initial_dyn)

        # Initial plan: start -> goal (FC-1)
        plan_result = planner.plan(start_xy, goal_xy, cost_map=risk_map)

        # Handle infeasible episodes (CC-4)
        if not plan_result.success and not env.bfs_corridor:
            elapsed_ms = (_time.perf_counter() - t0) * 1000.0
            return {
                "status": "ok",
                "row": {
                    "scenario_id": scenario_id,
                    "planner_id": planner_id,
                    "seed": seed,
                    "fog_condition": fog_label,
                    "success": False,
                    "path_length": 0,
                    "executed_steps": 1,
                    "replans": 0,
                    "computation_time_ms": round(elapsed_ms, 2),
                },
            }

        current_target = goal_xy if mission_task_done else mission_poi

        # Track path execution
        path = plan_result.path if plan_result.success else []
        path_idx = 0
        trajectory = [start_xy]

        # Step loop (RU-4)
        step_idx = 0
        replan_count = 0
        _failed_plan_cooldown = 0
        terminated = truncated = False
        final_info = info

        # POI stuck detection
        _poi_stuck_counter = 0
        _poi_best_dist = float("inf")
        _POI_STUCK_LIMIT = 30

        # Wall-clock timeout
        _wall_start = _time.perf_counter()
        _WALL_TIMEOUT_S = 300.0

        while not terminated and not truncated:
            step_idx += 1

            # Mission POI: STAY at POI until task completed
            if not mission_task_done and env.agent_xy == current_target:
                action = 4  # STAY
            else:
                action = _path_to_action(env.agent_xy, path, path_idx)

            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info

            # Wall-clock timeout
            if _time.perf_counter() - _wall_start > _WALL_TIMEOUT_S:
                final_info = dict(final_info)
                final_info["termination_reason"] = TerminationReason.WALL_TIMEOUT
                final_info["objective_completed"] = False
                truncated = True
            trajectory.append(env.agent_xy)

            # Mission POI completion -> switch to goal
            if not mission_task_done:
                task_progress = info.get("task_progress", "0/0")
                if task_progress.startswith("1/"):
                    mission_task_done = True
                    current_target = goal_xy
                    _poi_stuck_counter = 0
                    if not path or path[-1] != goal_xy:
                        pr = planner.plan(env.agent_xy, goal_xy, cost_map=risk_map)
                        if pr.success:
                            path = pr.path
                            path_idx = 0
                            replan_count += 1
                else:
                    poi_dist = (abs(env.agent_xy[0] - mission_poi[0])
                                + abs(env.agent_xy[1] - mission_poi[1]))
                    if poi_dist < _poi_best_dist:
                        _poi_best_dist = poi_dist
                        _poi_stuck_counter = 0
                    else:
                        _poi_stuck_counter += 1
                    if _poi_stuck_counter >= _POI_STUCK_LIMIT:
                        mission_task_done = True
                        current_target = goal_xy
                        _poi_stuck_counter = 0

            # Advance path index
            if path_idx < len(path) - 1:
                if env.agent_xy == path[path_idx + 1]:
                    path_idx += 1

            # Update planner with fog-filtered dynamic state
            dyn_state = env.get_dynamic_state()
            observed = fog.observe(env.agent_xy, dyn_state, step_idx) if fog else dyn_state
            risk_map = compute_risk_cost_map(heightmap, no_fly, config, observed)
            planner.update(observed)

            # Replan logic
            if _failed_plan_cooldown > 0:
                _failed_plan_cooldown -= 1
            else:
                should, reason = planner.should_replan(
                    env.agent_xy, path, dyn_state, step_idx
                )
                if should:
                    pr = planner.plan(env.agent_xy, current_target, cost_map=risk_map)
                    if pr.success:
                        path = pr.path
                        path_idx = 0
                        replan_count += 1
                        _failed_plan_cooldown = 0
                    else:
                        _failed_plan_cooldown = 15

        # Compute metrics
        metrics = compute_episode_metrics(
            scenario_id=config.name,
            planner_id=planner_id,
            seed=seed,
            trajectory=trajectory,
            events=env.events,
            final_info=final_info,
            plan_result=plan_result,
            replan_count=replan_count,
            goal_xy=goal_xy,
        )

        elapsed_ms = (_time.perf_counter() - t0) * 1000.0

        return {
            "status": "ok",
            "row": {
                "scenario_id": scenario_id,
                "planner_id": planner_id,
                "seed": seed,
                "fog_condition": fog_label,
                "success": metrics.get("success", False),
                "path_length": metrics.get("path_length", 0),
                "executed_steps": metrics.get("executed_steps_len", 0),
                "replans": metrics.get("replans", 0),
                "computation_time_ms": round(elapsed_ms, 2),
            },
        }

    except Exception:
        return {
            "status": "error",
            "scenario_id": scenario_id,
            "planner_id": planner_id,
            "seed": seed,
            "fog_condition": fog_label,
            "traceback": traceback.format_exc(),
        }


def _path_to_action(agent_xy, path, path_idx):
    """Convert path waypoint to action integer."""
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY = 0, 1, 2, 3, 4
    if not path:
        return ACTION_STAY
    target_idx = path_idx + 1 if path_idx + 1 < len(path) else len(path) - 1
    target = path[target_idx]
    ax, ay = agent_xy
    tx, ty = target
    if (ax, ay) == (tx, ty):
        if target_idx + 1 < len(path):
            tx, ty = path[target_idx + 1]
        else:
            return ACTION_STAY
    dx, dy = tx - ax, ty - ay
    if dx == 0 and dy == 0:
        return ACTION_STAY
    if abs(dx) >= abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    return ACTION_DOWN if dy > 0 else ACTION_UP


# ---------------------------------------------------------------------------
# Config serialization (for pickling across processes)
# ---------------------------------------------------------------------------


def _config_to_dict(config) -> dict:
    """Convert ScenarioConfig dataclass to a plain dict for pickling."""
    from dataclasses import fields as _fields
    return {f.name: getattr(config, f.name) for f in _fields(config)}


# ---------------------------------------------------------------------------
# Build fog ablation configs
# ---------------------------------------------------------------------------


def _build_fog_ablation_configs():
    """Build fog-ON and fog-OFF variants for all 3 medium scenarios.

    fog_off: enable_limited_visibility=False (full visibility)
    fog_on:  enable_limited_visibility=True, sensor_radius=50
    """
    from uavbench.scenarios.loader import load_scenario

    variants = []
    for sid in MEDIUM_SCENARIO_IDS:
        base = load_scenario(sid)

        # fog OFF: full visibility (baseline)
        cfg_off = replace(
            base,
            name=f"{sid}__fog_off",
            enable_limited_visibility=False,
        )
        variants.append(("fog_off", sid, cfg_off))

        # fog ON: partial observability
        cfg_on = replace(
            base,
            name=f"{sid}__fog_on",
            enable_limited_visibility=True,
            sensor_radius=50,
        )
        variants.append(("fog_on", sid, cfg_on))

    return variants


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def run_fog_ablation(n_seeds: int) -> str:
    """Run fog ablation using ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_path = os.path.join(OUTPUT_DIR, "fog_ablation.csv")
    error_log = os.path.join(OUTPUT_DIR, "fog_ablation_errors.log")

    variants = _build_fog_ablation_configs()

    # Build work units: one per (config, planner, seed)
    work_units = []
    for fog_label, scenario_id, config in variants:
        cfg_dict = _config_to_dict(config)
        for pid in ALL_PLANNERS:
            for seed in range(n_seeds):
                work_units.append((
                    cfg_dict, pid, seed, fog_label, scenario_id,
                ))

    total = len(work_units)
    print(f"\n{'='*70}")
    print(f"Limited-Visibility Ablation Study")
    print(f"  Conditions: fog_off (full visibility), fog_on (sensor_radius=50)")
    print(f"  Planners:   {', '.join(ALL_PLANNERS)}")
    print(f"  Scenarios:  {len(MEDIUM_SCENARIO_IDS)}")
    print(f"  Seeds:      {n_seeds}")
    print(f"  Episodes:   {total}")
    print(f"  Workers:    {MAX_WORKERS}")
    print(f"  Output:     {csv_path}")
    print(f"{'='*70}\n")

    done = 0
    errors = []
    wall_start = time.perf_counter()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_run_episode_worker, wu): wu
                for wu in work_units
            }
            for future in as_completed(futures):
                wu = futures[future]
                done += 1
                try:
                    result = future.result()
                except Exception as e:
                    _, pid, seed, fog_label, sid = wu
                    errors.append({
                        "scenario_id": sid,
                        "planner_id": pid,
                        "seed": seed,
                        "fog_condition": fog_label,
                        "traceback": str(e),
                    })
                    continue

                if result["status"] == "ok":
                    writer.writerow(result["row"])
                    f.flush()
                else:
                    errors.append(result)

                elapsed = time.perf_counter() - wall_start
                pct = done / total if total else 1
                eta = (elapsed / done * (total - done)) if done > 0 else 0
                _, pid, seed, fog_label, sid = wu
                print(
                    f"\r[{done}/{total}] ({100*pct:.0f}%) "
                    f"| Elapsed: {elapsed/60:.1f}m "
                    f"| ETA: {eta/60:.0f}m "
                    f"| Last: {fog_label}/{pid}/s{seed}",
                    end="", flush=True,
                )

    print()

    if errors:
        with open(error_log, "w") as ef:
            for err in errors:
                ef.write(f"--- {err.get('scenario_id','?')}/{err.get('planner_id','?')}"
                         f"/seed={err.get('seed','?')}/{err.get('fog_condition','?')} ---\n")
                ef.write(err.get("traceback", "unknown") + "\n")
        print(f"  Errors logged to: {error_log}")

    ok = done - len(errors)
    print(f"\n  Completed: {ok}/{total}, Errors: {len(errors)}")
    print(f"  Results:   {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_fog_summary(csv_path: str) -> None:
    """Print summary table: success rate fog-on vs fog-off per planner."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    print(f"\n{'='*70}")
    print("LIMITED-VISIBILITY ABLATION — SUCCESS RATE (%)")
    print(f"{'='*70}\n")

    # Overall by planner x fog condition
    pivot = df.pivot_table(
        index="planner_id",
        columns="fog_condition",
        values="success",
        aggfunc="mean",
    )
    # Reorder columns and rows
    col_order = [c for c in ["fog_off", "fog_on"] if c in pivot.columns]
    pivot = pivot[col_order]
    row_order = [p for p in ALL_PLANNERS if p in pivot.index]
    pivot = pivot.reindex(row_order)

    # Add delta column
    if "fog_off" in pivot.columns and "fog_on" in pivot.columns:
        pivot["delta"] = pivot["fog_on"] - pivot["fog_off"]

    print(f"  {'Planner':<22s} | {'Fog OFF':>8s} | {'Fog ON':>8s} | {'Delta':>8s}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for pid in row_order:
        off_val = pivot.loc[pid, "fog_off"] * 100 if "fog_off" in pivot.columns else float("nan")
        on_val = pivot.loc[pid, "fog_on"] * 100 if "fog_on" in pivot.columns else float("nan")
        delta = pivot.loc[pid, "delta"] * 100 if "delta" in pivot.columns else float("nan")
        sign = "+" if delta > 0 else ""
        print(f"  {pid:<22s} | {off_val:7.1f}% | {on_val:7.1f}% | {sign}{delta:6.1f}pp")

    # Per-scenario breakdown
    print(f"\n{'='*70}")
    print("PER-SCENARIO BREAKDOWN")
    print(f"{'='*70}\n")

    for sid in MEDIUM_SCENARIO_IDS:
        short = sid.replace("osm_", "").replace("_medium", "")
        print(f"  --- {short} ---")
        sub = df[df["scenario_id"] == sid]
        piv = sub.pivot_table(
            index="planner_id",
            columns="fog_condition",
            values="success",
            aggfunc="mean",
        )
        col_order = [c for c in ["fog_off", "fog_on"] if c in piv.columns]
        piv = piv[col_order]
        row_order = [p for p in ALL_PLANNERS if p in piv.index]
        piv = piv.reindex(row_order)
        for pid in row_order:
            off_v = piv.loc[pid, "fog_off"] * 100 if "fog_off" in piv.columns else float("nan")
            on_v = piv.loc[pid, "fog_on"] * 100 if "fog_on" in piv.columns else float("nan")
            print(f"    {pid:<22s} | OFF: {off_v:5.1f}% | ON: {on_v:5.1f}%")
        print()

    # Mean replans comparison
    print(f"{'='*70}")
    print("MEAN REPLANS PER EPISODE")
    print(f"{'='*70}\n")
    rp = df.pivot_table(
        index="planner_id",
        columns="fog_condition",
        values="replans",
        aggfunc="mean",
    )
    col_order = [c for c in ["fog_off", "fog_on"] if c in rp.columns]
    rp = rp[col_order]
    row_order = [p for p in ALL_PLANNERS if p in rp.index]
    rp = rp.reindex(row_order)
    print(f"  {'Planner':<22s} | {'Fog OFF':>8s} | {'Fog ON':>8s}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}")
    for pid in row_order:
        off_r = rp.loc[pid, "fog_off"] if "fog_off" in rp.columns else float("nan")
        on_r = rp.loc[pid, "fog_on"] if "fog_on" in rp.columns else float("nan")
        print(f"  {pid:<22s} | {off_r:8.1f} | {on_r:8.1f}")

    # Mean computation time
    print(f"\n{'='*70}")
    print("MEAN COMPUTATION TIME (ms)")
    print(f"{'='*70}\n")
    ct = df.pivot_table(
        index="planner_id",
        columns="fog_condition",
        values="computation_time_ms",
        aggfunc="mean",
    )
    col_order = [c for c in ["fog_off", "fog_on"] if c in ct.columns]
    ct = ct[col_order]
    row_order = [p for p in ALL_PLANNERS if p in ct.index]
    ct = ct.reindex(row_order)
    print(f"  {'Planner':<22s} | {'Fog OFF':>10s} | {'Fog ON':>10s}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}")
    for pid in row_order:
        off_t = ct.loc[pid, "fog_off"] if "fog_off" in ct.columns else float("nan")
        on_t = ct.loc[pid, "fog_on"] if "fog_on" in ct.columns else float("nan")
        print(f"  {pid:<22s} | {off_t:9.0f}ms | {on_t:9.0f}ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    global OUTPUT_DIR
    p = argparse.ArgumentParser(description="Limited-Visibility ablation study for UAVBench.")
    p.add_argument("--seeds", type=int, default=10,
                   help="Number of seeds per (scenario, planner, condition)")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = p.parse_args()
    OUTPUT_DIR = args.output_dir
    n_seeds = args.seeds

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wall_start = time.perf_counter()
    print(f"UAVBench Limited-Visibility Ablation (parallel, {MAX_WORKERS} workers)")
    print(f"  Seeds: {n_seeds}")

    csv_path = run_fog_ablation(n_seeds)

    # Summary
    print_fog_summary(csv_path)

    wall_total = time.perf_counter() - wall_start
    m, s = divmod(int(wall_total), 60)
    print(f"\nTotal wall time: {m}m {s:02d}s")
    print(f"Results in: {csv_path}")


if __name__ == "__main__":
    main()
