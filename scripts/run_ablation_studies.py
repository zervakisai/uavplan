#!/usr/bin/env python3
"""Ablation study runner for UAVBench paper.

Runs three ablation studies to isolate the effects of individual design choices:

  Ablation 1 — Dynamics Isolation
    For each medium-difficulty scenario, toggle dynamics layers independently:
      (a) fire only   (b) traffic only   (c) NFZ only   (d) all (baseline)
    Shows which dynamic obstacle type impacts each planner most.

  Ablation 2 — Replan Frequency
    For periodic_replan on all medium scenarios, vary replan_every_steps
    across [3, 6, 12, 24, 48].  Shows the cadence sweet spot.

  Ablation 3 — Fire Intensity
    For fire_delivery_medium, vary fire_ignition_points in [1, 2, 4, 6, 8].
    Shows the phase transition where planners begin to fail.

All experiments use `dataclasses.replace()` on frozen ScenarioConfig objects
to produce modified configs.  Episodes are run via the same runner loop as
the main paper experiments (benchmark/runner.py internals).

Usage:
    python scripts/run_ablation_studies.py                  # 10 seeds, all ablations
    python scripts/run_ablation_studies.py --seeds 30       # 30 seeds
    python scripts/run_ablation_studies.py --ablation 1     # dynamics only
    python scripts/run_ablation_studies.py --ablation 2     # replan freq only
    python scripts/run_ablation_studies.py --ablation 3     # fire intensity only

Results are saved to outputs/ablation_results/{ablation_name}.csv.
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
# Project imports
# ---------------------------------------------------------------------------

from uavbench.benchmark.runner import EpisodeResult
from uavbench.envs.urban import UrbanEnvV2
from uavbench.metrics.compute import compute_episode_metrics
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.schema import ScenarioConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/ablation_results"

# The 3 medium scenarios used as base configs for ablations.
MEDIUM_SCENARIO_IDS = [
    "gov_fire_delivery_medium",
    "gov_fire_surveillance_medium",
    "gov_flood_rescue_medium",
]

# All 6 paper planners.
ALL_PLANNERS = ["astar", "theta_star", "periodic_replan", "aggressive_replan", "dstar_lite", "apf"]

# Shared CSV columns across all ablation outputs.
BASE_COLUMNS = [
    "ablation",
    "variant",
    "scenario_id",
    "planner_id",
    "seed",
    "success",
    "path_length",
    "executed_steps",
    "replans",
    "termination_reason",
    "objective_completed",
    "computation_time_ms",
]


# ---------------------------------------------------------------------------
# Episode runner (accepts a ScenarioConfig directly)
# ---------------------------------------------------------------------------


def _run_episode_with_config(
    config: ScenarioConfig,
    planner_id: str,
    seed: int,
) -> EpisodeResult:
    """Run a single episode given an already-constructed ScenarioConfig.

    This mirrors the logic in benchmark/runner.py:run_episode() but accepts
    a config object instead of a scenario_id string, which allows running
    episodes with modified (ablated) configs that do not correspond to any
    on-disk YAML file.
    """
    # Create environment from config
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)

    # Export planner inputs
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    # Create planner
    planner_cls = PLANNERS[planner_id]
    planner = planner_cls(heightmap, no_fly, config)
    planner.set_seed(seed)

    # Initial plan
    plan_result = planner.plan(start_xy, goal_xy)

    # Track path execution
    path = plan_result.path if plan_result.success else []
    path_idx = 0
    trajectory: list[tuple[int, int]] = [start_xy]

    # Step loop
    step_idx = 0
    replan_count = 0
    terminated = False
    truncated = False
    final_info = info

    while not terminated and not truncated:
        step_idx += 1

        # Determine action from path
        action = _path_to_action(env.agent_xy, path, path_idx)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        trajectory.append(env.agent_xy)

        # Advance path index
        if path_idx < len(path) - 1:
            next_wp = path[path_idx + 1]
            if env.agent_xy == next_wp:
                path_idx += 1

        # Update planner with dynamic state
        dyn_state = env.get_dynamic_state()
        planner.update(dyn_state)

        # Check if replan needed
        should, reason = planner.should_replan(
            env.agent_xy, path, dyn_state, step_idx
        )
        if should:
            plan_result_new = planner.plan(env.agent_xy, goal_xy)
            if plan_result_new.success:
                path = plan_result_new.path
                path_idx = 0
                replan_count += 1

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
    )

    return EpisodeResult(
        events=env.events,
        trajectory=trajectory,
        metrics=metrics,
        frame_hashes=[],
    )


def _path_to_action(
    agent_xy: tuple[int, int],
    path: list[tuple[int, int]],
    path_idx: int,
) -> int:
    """Convert path waypoint to action integer (same as runner.py)."""
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

    dx = tx - ax
    dy = ty - ay

    if dx == 0 and dy == 0:
        return ACTION_STAY
    if abs(dx) >= abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    else:
        return ACTION_DOWN if dy > 0 else ACTION_UP


# ---------------------------------------------------------------------------
# Ablation config builders
# ---------------------------------------------------------------------------


def _build_dynamics_ablation_configs() -> list[tuple[str, str, ScenarioConfig]]:
    """Ablation 1: dynamics isolation.

    For each medium scenario, produce 4 variants:
      fire_only, traffic_only, nfz_only, all_dynamics (baseline).

    Returns list of (variant_label, scenario_id, modified_config).
    """
    variants: list[tuple[str, str, ScenarioConfig]] = []

    for sid in MEDIUM_SCENARIO_IDS:
        base = load_scenario(sid)

        # (a) fire only
        cfg_fire = replace(
            base,
            name=f"{sid}__fire_only",
            enable_fire=True,
            enable_traffic=False,
            enable_dynamic_nfz=False,
            fire_blocks_movement=True,
            traffic_blocks_movement=False,
            num_emergency_vehicles=0,
            num_nfz_zones=0,
        )
        variants.append(("fire_only", sid, cfg_fire))

        # (b) traffic only
        cfg_traffic = replace(
            base,
            name=f"{sid}__traffic_only",
            enable_fire=False,
            enable_traffic=True,
            enable_dynamic_nfz=False,
            fire_blocks_movement=False,
            traffic_blocks_movement=True,
            fire_ignition_points=0,
            num_nfz_zones=0,
        )
        variants.append(("traffic_only", sid, cfg_traffic))

        # (c) NFZ only
        cfg_nfz = replace(
            base,
            name=f"{sid}__nfz_only",
            enable_fire=False,
            enable_traffic=False,
            enable_dynamic_nfz=True,
            fire_blocks_movement=False,
            traffic_blocks_movement=False,
            fire_ignition_points=0,
            num_emergency_vehicles=0,
        )
        variants.append(("nfz_only", sid, cfg_nfz))

        # (d) all dynamics (baseline)
        cfg_all = replace(base, name=f"{sid}__all_dynamics")
        variants.append(("all_dynamics", sid, cfg_all))

    return variants


def _build_replan_frequency_configs() -> list[tuple[str, str, ScenarioConfig]]:
    """Ablation 2: replan cadence sweep for periodic_replan.

    Vary replan_every_steps across [3, 6, 12, 24, 48] on all medium scenarios.

    Returns list of (variant_label, scenario_id, modified_config).
    """
    cadences = [3, 6, 12, 24, 48]
    variants: list[tuple[str, str, ScenarioConfig]] = []

    for sid in MEDIUM_SCENARIO_IDS:
        base = load_scenario(sid)
        for cadence in cadences:
            cfg = replace(
                base,
                name=f"{sid}__replan_{cadence}",
                replan_every_steps=cadence,
            )
            variants.append((f"replan_{cadence}", sid, cfg))

    return variants


def _build_fire_intensity_configs() -> list[tuple[str, str, ScenarioConfig]]:
    """Ablation 3: fire ignition intensity sweep.

    Vary fire_ignition_points in [1, 2, 4, 6, 8] on fire_delivery_medium.

    Returns list of (variant_label, scenario_id, modified_config).
    """
    ignition_counts = [1, 2, 4, 6, 8]
    sid = "gov_fire_delivery_medium"
    base = load_scenario(sid)
    variants: list[tuple[str, str, ScenarioConfig]] = []

    for n_ign in ignition_counts:
        cfg = replace(
            base,
            name=f"{sid}__ignitions_{n_ign}",
            fire_ignition_points=n_ign,
        )
        variants.append((f"ignitions_{n_ign}", sid, cfg))

    return variants


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


def _progress_line(
    done: int, total: int, elapsed: float, label: str, ep_sec: float,
) -> str:
    pct = done / total if total else 1
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    parts = [f"[{bar}] {done}/{total} ({100 * pct:.0f}%)"]
    parts.append(f"Elapsed: {_fmt_duration(elapsed)}")
    if done > 0:
        avg = elapsed / done
        remaining = avg * (total - done)
        parts.append(f"ETA: {_fmt_duration(remaining)}")
    if label:
        parts.append(f"Last: {label} ({ep_sec:.1f}s)")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------


def _extract_row(
    ablation_name: str,
    variant: str,
    scenario_id: str,
    planner_id: str,
    seed: int,
    result: EpisodeResult,
    elapsed_ms: float,
) -> dict:
    """Extract a CSV row dict from an EpisodeResult."""
    m = result.metrics
    return {
        "ablation": ablation_name,
        "variant": variant,
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


def run_ablation_1(n_seeds: int) -> str:
    """Ablation 1: Dynamics isolation.

    Runs all 6 planners x 3 medium scenarios x 4 variants x n_seeds.
    Returns path to output CSV.
    """
    ablation_name = "dynamics_isolation"
    csv_path = os.path.join(OUTPUT_DIR, f"{ablation_name}.csv")
    error_log = os.path.join(OUTPUT_DIR, f"{ablation_name}_errors.log")

    variants = _build_dynamics_ablation_configs()
    planners = ALL_PLANNERS
    total = len(variants) * len(planners) * n_seeds

    print(f"\n{'='*70}")
    print(f"Ablation 1: Dynamics Isolation")
    print(f"  Variants:  {len(variants)} (3 scenarios x 4 dynamics combos)")
    print(f"  Planners:  {len(planners)} -- {', '.join(planners)}")
    print(f"  Seeds:     {n_seeds}")
    print(f"  Total:     {total} episodes")
    print(f"  Output:    {csv_path}")
    print(f"{'='*70}\n")

    errors: list[dict] = []
    done = 0
    wall_start = time.perf_counter()
    last_label = ""
    last_sec = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()

        for variant_label, scenario_id, config in variants:
            for planner_id in planners:
                for seed in range(n_seeds):
                    try:
                        t0 = time.perf_counter()
                        result = _run_episode_with_config(config, planner_id, seed)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        last_sec = elapsed_ms / 1000.0

                        row = _extract_row(
                            ablation_name, variant_label,
                            scenario_id, planner_id, seed,
                            result, elapsed_ms,
                        )
                        writer.writerow(row)
                        f.flush()

                    except Exception:
                        tb = traceback.format_exc()
                        last_sec = time.perf_counter() - t0
                        errors.append({
                            "variant": variant_label,
                            "scenario_id": scenario_id,
                            "planner_id": planner_id,
                            "seed": seed,
                            "traceback": tb,
                        })

                    done += 1
                    last_label = f"{variant_label}/{planner_id}/seed={seed}"
                    elapsed = time.perf_counter() - wall_start
                    print(
                        f"\r{_progress_line(done, total, elapsed, last_label, last_sec)}",
                        end="", flush=True,
                    )

    print()
    _write_error_log(error_log, errors)
    _print_ablation_footer(done, total, len(errors), csv_path, error_log)
    return csv_path


def run_ablation_2(n_seeds: int) -> str:
    """Ablation 2: Replan frequency sweep (periodic_replan only).

    Runs periodic_replan x 3 medium scenarios x 5 cadences x n_seeds.
    Returns path to output CSV.
    """
    ablation_name = "replan_frequency"
    csv_path = os.path.join(OUTPUT_DIR, f"{ablation_name}.csv")
    error_log = os.path.join(OUTPUT_DIR, f"{ablation_name}_errors.log")

    variants = _build_replan_frequency_configs()
    planner_id = "periodic_replan"
    total = len(variants) * n_seeds

    print(f"\n{'='*70}")
    print(f"Ablation 2: Replan Frequency Sweep")
    print(f"  Variants:  {len(variants)} (3 scenarios x 5 cadences)")
    print(f"  Planner:   {planner_id}")
    print(f"  Seeds:     {n_seeds}")
    print(f"  Total:     {total} episodes")
    print(f"  Output:    {csv_path}")
    print(f"{'='*70}\n")

    errors: list[dict] = []
    done = 0
    wall_start = time.perf_counter()
    last_label = ""
    last_sec = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()

        for variant_label, scenario_id, config in variants:
            for seed in range(n_seeds):
                try:
                    t0 = time.perf_counter()
                    result = _run_episode_with_config(config, planner_id, seed)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    last_sec = elapsed_ms / 1000.0

                    row = _extract_row(
                        ablation_name, variant_label,
                        scenario_id, planner_id, seed,
                        result, elapsed_ms,
                    )
                    writer.writerow(row)
                    f.flush()

                except Exception:
                    tb = traceback.format_exc()
                    last_sec = time.perf_counter() - t0
                    errors.append({
                        "variant": variant_label,
                        "scenario_id": scenario_id,
                        "planner_id": planner_id,
                        "seed": seed,
                        "traceback": tb,
                    })

                done += 1
                last_label = f"{variant_label}/{scenario_id}/seed={seed}"
                elapsed = time.perf_counter() - wall_start
                print(
                    f"\r{_progress_line(done, total, elapsed, last_label, last_sec)}",
                    end="", flush=True,
                )

    print()
    _write_error_log(error_log, errors)
    _print_ablation_footer(done, total, len(errors), csv_path, error_log)
    return csv_path


def run_ablation_3(n_seeds: int) -> str:
    """Ablation 3: Fire intensity sweep (all planners, fire_delivery_medium).

    Runs all 6 planners x 5 ignition counts x n_seeds.
    Returns path to output CSV.
    """
    ablation_name = "fire_intensity"
    csv_path = os.path.join(OUTPUT_DIR, f"{ablation_name}.csv")
    error_log = os.path.join(OUTPUT_DIR, f"{ablation_name}_errors.log")

    variants = _build_fire_intensity_configs()
    planners = ALL_PLANNERS
    total = len(variants) * len(planners) * n_seeds

    print(f"\n{'='*70}")
    print(f"Ablation 3: Fire Intensity Sweep")
    print(f"  Variants:  {len(variants)} (5 ignition counts)")
    print(f"  Planners:  {len(planners)} -- {', '.join(planners)}")
    print(f"  Seeds:     {n_seeds}")
    print(f"  Total:     {total} episodes")
    print(f"  Output:    {csv_path}")
    print(f"{'='*70}\n")

    errors: list[dict] = []
    done = 0
    wall_start = time.perf_counter()
    last_label = ""
    last_sec = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()

        for variant_label, scenario_id, config in variants:
            for planner_id in planners:
                for seed in range(n_seeds):
                    try:
                        t0 = time.perf_counter()
                        result = _run_episode_with_config(config, planner_id, seed)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        last_sec = elapsed_ms / 1000.0

                        row = _extract_row(
                            ablation_name, variant_label,
                            scenario_id, planner_id, seed,
                            result, elapsed_ms,
                        )
                        writer.writerow(row)
                        f.flush()

                    except Exception:
                        tb = traceback.format_exc()
                        last_sec = time.perf_counter() - t0
                        errors.append({
                            "variant": variant_label,
                            "scenario_id": scenario_id,
                            "planner_id": planner_id,
                            "seed": seed,
                            "traceback": tb,
                        })

                    done += 1
                    last_label = f"{variant_label}/{planner_id}/seed={seed}"
                    elapsed = time.perf_counter() - wall_start
                    print(
                        f"\r{_progress_line(done, total, elapsed, last_label, last_sec)}",
                        end="", flush=True,
                    )

    print()
    _write_error_log(error_log, errors)
    _print_ablation_footer(done, total, len(errors), csv_path, error_log)
    return csv_path


# ---------------------------------------------------------------------------
# Summary tables (printed to stdout after each ablation)
# ---------------------------------------------------------------------------


def print_dynamics_summary(csv_path: str) -> None:
    """Print summary table for Ablation 1: dynamics isolation."""
    try:
        import pandas as pd
    except ImportError:
        print("  [pandas not installed -- skipping summary table]")
        return

    df = pd.read_csv(csv_path)
    print(f"\n--- Ablation 1: Dynamics Isolation Summary ---\n")

    # Pivot: variant x planner -> success rate
    pivot = df.pivot_table(
        index="variant",
        columns="planner_id",
        values="success",
        aggfunc="mean",
    )
    # Reorder
    variant_order = ["fire_only", "traffic_only", "nfz_only", "all_dynamics"]
    pivot = pivot.reindex(variant_order)
    pivot = pivot[[p for p in ALL_PLANNERS if p in pivot.columns]]

    print("Success Rate (%):")
    for var in pivot.index:
        vals = " | ".join(f"{100*pivot.loc[var, p]:5.1f}" if p in pivot.columns else "  -- " for p in ALL_PLANNERS)
        print(f"  {var:16s} | {vals}")

    header = "                   | " + " | ".join(f"{p:>5s}" for p in ALL_PLANNERS)
    print(f"\n  {'':16s} | " + " | ".join(f"{p[:5]:>5s}" for p in ALL_PLANNERS))

    # Also show mean path length
    print("\nMean Path Length:")
    pivot_pl = df.pivot_table(
        index="variant",
        columns="planner_id",
        values="path_length",
        aggfunc="mean",
    ).reindex(variant_order)
    for var in pivot_pl.index:
        vals = " | ".join(f"{pivot_pl.loc[var, p]:7.0f}" if p in pivot_pl.columns else "    -- " for p in ALL_PLANNERS)
        print(f"  {var:16s} | {vals}")


def print_replan_frequency_summary(csv_path: str) -> None:
    """Print summary table for Ablation 2: replan frequency."""
    try:
        import pandas as pd
    except ImportError:
        print("  [pandas not installed -- skipping summary table]")
        return

    df = pd.read_csv(csv_path)
    print(f"\n--- Ablation 2: Replan Frequency Summary ---\n")

    # Extract cadence from variant label
    df["cadence"] = df["variant"].str.extract(r"replan_(\d+)").astype(int)

    # Group by cadence
    grouped = df.groupby("cadence").agg(
        success_rate=("success", "mean"),
        mean_path_len=("path_length", "mean"),
        mean_replans=("replans", "mean"),
        mean_time_ms=("computation_time_ms", "mean"),
    )

    print(f"  {'Cadence':>7s} | {'SR%':>6s} | {'PathLen':>8s} | {'Replans':>8s} | {'Time(ms)':>9s}")
    print(f"  {'-'*7} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*9}")
    for cadence, row in grouped.iterrows():
        print(
            f"  {cadence:7d} | {100*row['success_rate']:5.1f}% | "
            f"{row['mean_path_len']:8.0f} | {row['mean_replans']:8.1f} | "
            f"{row['mean_time_ms']:9.0f}"
        )

    # Per-scenario breakdown
    print(f"\n  Per-scenario success rate (%):")
    pivot = df.pivot_table(
        index="cadence",
        columns="scenario_id",
        values="success",
        aggfunc="mean",
    )
    scns = sorted(pivot.columns)
    short = [s.replace("gov_", "").replace("_medium", "") for s in scns]
    print(f"  {'Cadence':>7s} | " + " | ".join(f"{s:>16s}" for s in short))
    print(f"  {'-'*7} | " + " | ".join(f"{'-'*16}" for _ in short))
    for cadence in sorted(pivot.index):
        vals = " | ".join(
            f"{100*pivot.loc[cadence, scn]:15.1f}%" for scn in scns
        )
        print(f"  {cadence:7d} | {vals}")


def print_fire_intensity_summary(csv_path: str) -> None:
    """Print summary table for Ablation 3: fire intensity."""
    try:
        import pandas as pd
    except ImportError:
        print("  [pandas not installed -- skipping summary table]")
        return

    df = pd.read_csv(csv_path)
    print(f"\n--- Ablation 3: Fire Intensity Summary ---\n")

    # Extract ignition count from variant label
    df["ignitions"] = df["variant"].str.extract(r"ignitions_(\d+)").astype(int)

    # Pivot: ignitions x planner -> success rate
    pivot = df.pivot_table(
        index="ignitions",
        columns="planner_id",
        values="success",
        aggfunc="mean",
    )
    pivot = pivot[[p for p in ALL_PLANNERS if p in pivot.columns]]

    print("Success Rate (%):")
    print(f"  {'Ignitions':>9s} | " + " | ".join(f"{p[:8]:>8s}" for p in ALL_PLANNERS))
    print(f"  {'-'*9} | " + " | ".join(f"{'-'*8}" for _ in ALL_PLANNERS))
    for n_ign in sorted(pivot.index):
        vals = " | ".join(
            f"{100*pivot.loc[n_ign, p]:7.1f}%" if p in pivot.columns else "     -- "
            for p in ALL_PLANNERS
        )
        print(f"  {n_ign:9d} | {vals}")

    # Mean path length
    print("\nMean Path Length (successful episodes only):")
    successful = df[df["success"] == True]  # noqa: E712
    if not successful.empty:
        pivot_pl = successful.pivot_table(
            index="ignitions",
            columns="planner_id",
            values="path_length",
            aggfunc="mean",
        )
        pivot_pl = pivot_pl[[p for p in ALL_PLANNERS if p in pivot_pl.columns]]
        print(f"  {'Ignitions':>9s} | " + " | ".join(f"{p[:8]:>8s}" for p in ALL_PLANNERS))
        print(f"  {'-'*9} | " + " | ".join(f"{'-'*8}" for _ in ALL_PLANNERS))
        for n_ign in sorted(pivot_pl.index):
            vals = " | ".join(
                f"{pivot_pl.loc[n_ign, p]:8.0f}" if p in pivot_pl.columns else "      -- "
                for p in ALL_PLANNERS
            )
            print(f"  {n_ign:9d} | {vals}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_error_log(path: str, errors: list[dict]) -> None:
    if not errors:
        return
    with open(path, "w") as f:
        for err in errors:
            f.write(f"--- {err.get('variant', '?')} / {err.get('scenario_id', '?')} / "
                    f"{err.get('planner_id', '?')} / seed={err.get('seed', '?')} ---\n")
            f.write(err["traceback"])
            f.write("\n")


def _print_ablation_footer(
    done: int, total: int, n_errors: int,
    csv_path: str, error_log: str,
) -> None:
    print(f"\n  Completed: {done - n_errors}/{total}")
    print(f"  Errors:    {n_errors}")
    if n_errors > 0:
        print(f"  Error log: {error_log}")
    print(f"  Results:   {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run UAVBench ablation studies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ablation IDs:\n"
            "  1  Dynamics isolation (fire / traffic / NFZ / all)\n"
            "  2  Replan frequency sweep (periodic_replan cadences)\n"
            "  3  Fire intensity sweep (ignition point counts)\n"
        ),
    )
    p.add_argument(
        "--seeds", type=int, default=10,
        help="Number of seeds per experiment (default: 10)",
    )
    p.add_argument(
        "--ablation", type=int, default=None, choices=[1, 2, 3],
        help="Run only this ablation (default: run all)",
    )
    p.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return p.parse_args()


def main() -> None:
    global OUTPUT_DIR  # noqa: PLW0603
    args = _parse_args()
    OUTPUT_DIR = args.output_dir
    n_seeds = args.seeds

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_all = args.ablation is None
    wall_start = time.perf_counter()

    print("UAVBench Ablation Studies")
    print(f"  Seeds:      {n_seeds}")
    print(f"  Output dir: {OUTPUT_DIR}")
    if not run_all:
        print(f"  Ablation:   {args.ablation}")

    csv_paths: dict[int, str] = {}

    # --- Ablation 1 ---
    if run_all or args.ablation == 1:
        csv_paths[1] = run_ablation_1(n_seeds)

    # --- Ablation 2 ---
    if run_all or args.ablation == 2:
        csv_paths[2] = run_ablation_2(n_seeds)

    # --- Ablation 3 ---
    if run_all or args.ablation == 3:
        csv_paths[3] = run_ablation_3(n_seeds)

    # --- Summary tables ---
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
    print(f"\n{'='*70}")
    print(f"All ablation studies complete. Wall time: {_fmt_duration(wall_total)}")
    print(f"Results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
