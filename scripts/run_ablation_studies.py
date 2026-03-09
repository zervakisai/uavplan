#!/usr/bin/env python3
"""Ablation study runner for UAVBench paper — multiprocessing version.

Runs three ablation studies with ProcessPoolExecutor (6 workers):

  Ablation 1 — Dynamics Isolation (600 episodes)
  Ablation 2 — Replan Frequency (150 episodes)
  Ablation 3 — Fire Intensity (250 episodes)

Usage:
    python scripts/run_ablation_studies.py                  # 10 seeds, all
    python scripts/run_ablation_studies.py --seeds 30       # 30 seeds
    python scripts/run_ablation_studies.py --ablation 1     # dynamics only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import replace
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/ablation_results"
MAX_WORKERS = 6

MEDIUM_SCENARIO_IDS = [
    "osm_penteli_pharma_delivery_medium",
    "osm_downtown_fire_surveillance_medium",
    "osm_piraeus_flood_rescue_medium",
]

ALL_PLANNERS = ["astar", "periodic_replan", "aggressive_replan", "dstar_lite", "apf"]

BASE_COLUMNS = [
    "ablation", "variant", "scenario_id", "planner_id", "seed",
    "success", "path_length", "executed_steps", "replans",
    "termination_reason", "objective_completed", "computation_time_ms",
]


# ---------------------------------------------------------------------------
# Worker function (module-level for pickling)
# ---------------------------------------------------------------------------


def _run_block(args: tuple) -> list[dict]:
    """Worker: runs one (config, planner_id) x n_seeds block.

    Args is a tuple: (config_dict, planner_id, n_seeds, ablation_name,
                       variant_label, scenario_id)

    config_dict is used instead of ScenarioConfig to avoid pickling issues
    with spawn. We reconstruct the config inside the worker.
    """
    import time
    import traceback
    from dataclasses import fields

    config_dict, planner_id, n_seeds, ablation_name, variant_label, scenario_id = args

    # Local imports for clean subprocess
    from uavbench.benchmark.runner import EpisodeResult
    from uavbench.envs.urban import UrbanEnvV2
    from uavbench.metrics.compute import compute_episode_metrics
    from uavbench.planners import PLANNERS
    from uavbench.scenarios.schema import ScenarioConfig

    # Reconstruct config from dict
    valid_fields = {f.name for f in fields(ScenarioConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ScenarioConfig(**filtered)

    results = []
    for seed in range(n_seeds):
        try:
            t0 = time.perf_counter()

            # --- Run episode with config (inline) ---
            env = UrbanEnvV2(config)
            obs, info = env.reset(seed=seed)
            heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

            planner_cls = PLANNERS[planner_id]
            planner = planner_cls(heightmap, no_fly, config)
            planner.set_seed(seed)

            plan_result = planner.plan(start_xy, goal_xy)
            path = plan_result.path if plan_result.success else []
            path_idx = 0
            trajectory = [start_xy]
            step_idx = 0
            replan_count = 0
            terminated = truncated = False
            final_info = info

            while not terminated and not truncated:
                step_idx += 1
                action = _path_to_action(env.agent_xy, path, path_idx)
                obs, reward, terminated, truncated, info = env.step(action)
                final_info = info
                trajectory.append(env.agent_xy)

                if path_idx < len(path) - 1:
                    if env.agent_xy == path[path_idx + 1]:
                        path_idx += 1

                dyn_state = env.get_dynamic_state()
                planner.update(dyn_state)

                should, reason = planner.should_replan(
                    env.agent_xy, path, dyn_state, step_idx
                )
                if should:
                    pr = planner.plan(env.agent_xy, goal_xy)
                    if pr.success:
                        path = pr.path
                        path_idx = 0
                        replan_count += 1

            metrics = compute_episode_metrics(
                scenario_id=config.name,
                planner_id=planner_id,
                seed=seed,
                trajectory=trajectory,
                events=env.events,
                final_info=final_info,
                plan_result=plan_result,
                replan_count=replan_count,
            )
            # --- End episode ---

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            m = metrics
            row = {
                "ablation": ablation_name,
                "variant": variant_label,
                "scenario_id": scenario_id,
                "planner_id": planner_id,
                "seed": seed,
                "success": m.get("success", False),
                "path_length": m.get("path_length", 0),
                "executed_steps": m.get("executed_steps_len", 0),
                "replans": m.get("replans", 0),
                "termination_reason": m.get("termination_reason", "unknown"),
                "objective_completed": m.get("objective_completed", False),
                "computation_time_ms": round(elapsed_ms, 2),
            }
            results.append({"status": "ok", "row": row})

        except Exception:
            results.append({
                "status": "error",
                "variant": variant_label,
                "scenario_id": scenario_id,
                "planner_id": planner_id,
                "seed": seed,
                "traceback": traceback.format_exc(),
            })
    return results


def _path_to_action(agent_xy, path, path_idx):
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
    """Convert ScenarioConfig dataclass to a plain dict for pickling.

    Enums are kept as-is (str enums pickle fine with spawn).
    """
    from dataclasses import fields
    return {f.name: getattr(config, f.name) for f in fields(config)}


# ---------------------------------------------------------------------------
# Ablation config builders
# ---------------------------------------------------------------------------


def _build_dynamics_ablation_configs():
    from uavbench.scenarios.loader import load_scenario
    variants = []
    for sid in MEDIUM_SCENARIO_IDS:
        base = load_scenario(sid)
        cfg_fire = replace(base, name=f"{sid}__fire_only",
                           enable_fire=True, enable_traffic=False,
                           enable_dynamic_nfz=False, fire_blocks_movement=True,
                           traffic_blocks_movement=False,
                           num_emergency_vehicles=0, num_nfz_zones=0)
        variants.append(("fire_only", sid, cfg_fire))

        cfg_traffic = replace(base, name=f"{sid}__traffic_only",
                              enable_fire=False, enable_traffic=True,
                              enable_dynamic_nfz=False, fire_blocks_movement=False,
                              traffic_blocks_movement=True,
                              fire_ignition_points=0, num_nfz_zones=0)
        variants.append(("traffic_only", sid, cfg_traffic))

        cfg_nfz = replace(base, name=f"{sid}__nfz_only",
                          enable_fire=False, enable_traffic=False,
                          enable_dynamic_nfz=True, fire_blocks_movement=False,
                          traffic_blocks_movement=False,
                          fire_ignition_points=0, num_emergency_vehicles=0)
        variants.append(("nfz_only", sid, cfg_nfz))

        cfg_all = replace(base, name=f"{sid}__all_dynamics")
        variants.append(("all_dynamics", sid, cfg_all))
    return variants


def _build_replan_frequency_configs():
    from uavbench.scenarios.loader import load_scenario
    cadences = [3, 6, 12, 24, 48]
    variants = []
    for sid in MEDIUM_SCENARIO_IDS:
        base = load_scenario(sid)
        for c in cadences:
            cfg = replace(base, name=f"{sid}__replan_{c}", replan_every_steps=c)
            variants.append((f"replan_{c}", sid, cfg))
    return variants


def _build_fire_intensity_configs():
    from uavbench.scenarios.loader import load_scenario
    ignition_counts = [1, 2, 4, 6, 8]
    sid = "osm_penteli_pharma_delivery_medium"
    base = load_scenario(sid)
    variants = []
    for n in ignition_counts:
        cfg = replace(base, name=f"{sid}__ignitions_{n}", fire_ignition_points=n)
        variants.append((f"ignitions_{n}", sid, cfg))
    return variants


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def _run_ablation_parallel(
    ablation_name: str,
    variants: list,
    planners: list[str],
    n_seeds: int,
) -> str:
    """Run an ablation study using ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    csv_path = os.path.join(OUTPUT_DIR, f"{ablation_name}.csv")
    error_log = os.path.join(OUTPUT_DIR, f"{ablation_name}_errors.log")

    # Build work units: (config_dict, planner_id, n_seeds, ...)
    work_units = []
    for variant_label, scenario_id, config in variants:
        cfg_dict = _config_to_dict(config)
        for pid in planners:
            work_units.append((
                cfg_dict, pid, n_seeds,
                ablation_name, variant_label, scenario_id,
            ))

    total = len(work_units) * n_seeds
    print(f"\n{'='*70}")
    print(f"Ablation: {ablation_name}")
    print(f"  Work units: {len(work_units)} blocks x {n_seeds} seeds = {total} episodes")
    print(f"  Workers:    {MAX_WORKERS}")
    print(f"  Output:     {csv_path}")
    print(f"{'='*70}\n")

    done = 0
    errors = []
    wall_start = time.perf_counter()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_run_block, wu): wu for wu in work_units
            }
            for future in as_completed(futures):
                wu = futures[future]
                try:
                    block_results = future.result()
                except Exception as e:
                    _, pid, _, _, vl, sid = wu
                    errors.append({
                        "variant": vl, "scenario_id": sid,
                        "planner_id": pid, "seed": "ALL",
                        "traceback": str(e),
                    })
                    done += n_seeds
                    continue

                for item in block_results:
                    done += 1
                    if item["status"] == "ok":
                        writer.writerow(item["row"])
                        f.flush()
                    else:
                        errors.append(item)

                elapsed = time.perf_counter() - wall_start
                pct = done / total if total else 1
                eta = (elapsed / done * (total - done)) if done > 0 else 0
                _, pid, _, _, vl, sid = wu
                print(
                    f"\r[{done}/{total}] ({100*pct:.0f}%) "
                    f"| Elapsed: {elapsed/60:.1f}m "
                    f"| ETA: {eta/60:.0f}m "
                    f"| Last: {vl}/{pid}",
                    end="", flush=True,
                )

    print()
    if errors:
        with open(error_log, "w") as ef:
            for err in errors:
                ef.write(f"--- {err.get('variant','?')}/{err.get('planner_id','?')}"
                         f"/seed={err.get('seed','?')} ---\n")
                ef.write(err.get("traceback", "unknown") + "\n")

    ok = done - len(errors)
    print(f"  Completed: {ok}/{total}, Errors: {len(errors)}")
    print(f"  Results:   {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def print_dynamics_summary(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n--- Ablation 1: Dynamics Isolation Summary ---\n")
    pivot = df.pivot_table(index="variant", columns="planner_id",
                           values="success", aggfunc="mean")
    order = ["fire_only", "traffic_only", "nfz_only", "all_dynamics"]
    pivot = pivot.reindex(order)
    pivot = pivot[[p for p in ALL_PLANNERS if p in pivot.columns]]
    print("Success Rate (%):")
    for var in pivot.index:
        vals = " | ".join(
            f"{100*pivot.loc[var, p]:5.1f}" if p in pivot.columns else "  -- "
            for p in ALL_PLANNERS
        )
        print(f"  {var:16s} | {vals}")


def print_replan_frequency_summary(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n--- Ablation 2: Replan Frequency Summary ---\n")
    df["cadence"] = df["variant"].str.extract(r"replan_(\d+)").astype(int)
    grouped = df.groupby("cadence").agg(
        success_rate=("success", "mean"),
        mean_replans=("replans", "mean"),
    )
    print(f"  {'Cadence':>7s} | {'SR%':>6s} | {'Replans':>8s}")
    for cadence, row in grouped.iterrows():
        print(f"  {cadence:7d} | {100*row['success_rate']:5.1f}% | {row['mean_replans']:8.1f}")


def print_fire_intensity_summary(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n--- Ablation 3: Fire Intensity Summary ---\n")
    df["ignitions"] = df["variant"].str.extract(r"ignitions_(\d+)").astype(int)
    pivot = df.pivot_table(index="ignitions", columns="planner_id",
                           values="success", aggfunc="mean")
    pivot = pivot[[p for p in ALL_PLANNERS if p in pivot.columns]]
    print("Success Rate (%):")
    print(f"  {'Ign':>4s} | " + " | ".join(f"{p[:8]:>8s}" for p in ALL_PLANNERS))
    for n in sorted(pivot.index):
        vals = " | ".join(
            f"{100*pivot.loc[n, p]:7.1f}%" if p in pivot.columns else "     -- "
            for p in ALL_PLANNERS
        )
        print(f"  {n:4d} | {vals}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    global OUTPUT_DIR
    p = argparse.ArgumentParser(description="Run UAVBench ablation studies (parallel).")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--ablation", type=int, default=None, choices=[1, 2, 3])
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = p.parse_args()
    OUTPUT_DIR = args.output_dir
    n_seeds = args.seeds

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_all = args.ablation is None
    wall_start = time.perf_counter()
    csv_paths = {}

    print(f"UAVBench Ablation Studies (parallel, {MAX_WORKERS} workers)")
    print(f"  Seeds: {n_seeds}")

    if run_all or args.ablation == 1:
        variants = _build_dynamics_ablation_configs()
        csv_paths[1] = _run_ablation_parallel(
            "dynamics_isolation", variants, ALL_PLANNERS, n_seeds)

    if run_all or args.ablation == 2:
        variants = _build_replan_frequency_configs()
        csv_paths[2] = _run_ablation_parallel(
            "replan_frequency", variants, ["periodic_replan"], n_seeds)

    if run_all or args.ablation == 3:
        variants = _build_fire_intensity_configs()
        csv_paths[3] = _run_ablation_parallel(
            "fire_intensity", variants, ALL_PLANNERS, n_seeds)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLES")
    print(f"{'='*70}")
    if 1 in csv_paths:
        print_dynamics_summary(csv_paths[1])
    if 2 in csv_paths:
        print_replan_frequency_summary(csv_paths[2])
    if 3 in csv_paths:
        print_fire_intensity_summary(csv_paths[3])

    wall_total = time.perf_counter() - wall_start
    m, s = divmod(int(wall_total), 60)
    print(f"\nTotal wall time: {m}m {s:02d}s")
    print(f"Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
