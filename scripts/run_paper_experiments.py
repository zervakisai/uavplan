#!/usr/bin/env python3
"""Paper experiment runner — 5 planners x 3 OSM scenarios x 30 seeds = 450 episodes.

Saves per-episode results to outputs/paper_results/all_episodes.csv.
Episodes that crash are logged and skipped (never stops the whole run).
Uses multiprocessing (6 workers) for Apple M1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/paper_results"
ERROR_LOG = os.path.join(OUTPUT_DIR, "errors.log")

COLUMNS = [
    "scenario_id",
    "planner_id",
    "seed",
    "success",
    "path_length",
    "executed_steps",
    "planned_waypoints_len",
    "replans",
    "replan_storm_ratio",
    "service_time_steps",
    "termination_reason",
    "objective_completed",
    "computation_time_ms",
    "guardrail_activated",
    "guardrail_depth",
    "infeasible",
    "feasible_after_guardrail",
    "collision_count",
    "nfz_violations",
    "mission_type",
    "domain",
    "difficulty",
    "track",
    "reject_reason_counts",
    "mission_score",
    "tasks_completed",
    "tasks_total",
]


# ---------------------------------------------------------------------------
# Helpers (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


def _extract_row(
    scenario_id: str,
    planner_id: str,
    seed: int,
    result: "EpisodeResult",
    elapsed_ms: float,
    config: "ScenarioConfig",
) -> dict:
    """Extract a CSV row dict from an EpisodeResult."""
    m = result.metrics

    # Extract mission briefing from events
    briefing = next(
        (e for e in result.events if e.get("type") == "mission_briefing"),
        {},
    )

    # Detect guardrail activation from events
    guardrail_events = [
        e for e in result.events if "guardrail" in str(e.get("type", ""))
    ]
    guardrail_depth = 0
    feasible_after = True
    for ge in guardrail_events:
        d = ge.get("depth", 0)
        if d > guardrail_depth:
            guardrail_depth = d
        if ge.get("feasible_after") is False:
            feasible_after = False

    tr = m.get("termination_reason", "unknown")

    # Reject reason counts from events (EC-1)
    reject_counts: dict[str, int] = {}
    for ev in result.events:
        if ev.get("type") == "move_rejected":
            reason = ev.get("reject_reason", "")
            reason_str = reason.value if hasattr(reason, "value") else str(reason)
            if reason_str:
                reject_counts[reason_str] = reject_counts.get(reason_str, 0) + 1

    return {
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "seed": seed,
        "success": m.get("success", False),
        "path_length": m.get("path_length", 0),
        "executed_steps": m.get("executed_steps_len", 0),
        "planned_waypoints_len": m.get("planned_waypoints_len", 0),
        "replans": m.get("replans", 0),
        "replan_storm_ratio": round(m.get("replan_storm_ratio", 0.0), 4),
        "service_time_steps": briefing.get("service_time_steps", 0),
        "termination_reason": tr,
        "objective_completed": m.get("objective_completed", False),
        "computation_time_ms": round(elapsed_ms, 2),
        "guardrail_activated": len(guardrail_events) > 0,
        "guardrail_depth": guardrail_depth,
        "infeasible": tr == "infeasible",
        "feasible_after_guardrail": feasible_after,
        "collision_count": m.get("collision_count", 0),
        "nfz_violations": m.get("nfz_violations", 0),
        "mission_type": config.mission_type.value,
        "domain": config.domain.value,
        "difficulty": config.difficulty.value,
        "track": config.paper_track,
        "reject_reason_counts": json.dumps(reject_counts) if reject_counts else "{}",
        "mission_score": round(m.get("mission_score", 0.0), 4),
        "tasks_completed": m.get("tasks_completed", 0),
        "tasks_total": m.get("tasks_total", 0),
    }


def _run_scenario_planner_block(args: tuple) -> list[dict]:
    """Worker function: runs one (scenario_id, planner_id) x n_seeds block.

    Must be at module level for multiprocessing pickling on macOS.
    """
    scenario_id, planner_id, n_seeds = args
    # Local imports inside worker for clean subprocess initialization
    import time
    import traceback

    from uavbench.benchmark.runner import run_episode
    from uavbench.scenarios.loader import load_scenario

    config = load_scenario(scenario_id)
    results = []
    for seed in range(n_seeds):
        try:
            t0 = time.perf_counter()
            result = run_episode(scenario_id, planner_id, seed)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            row = _extract_row(
                scenario_id, planner_id, seed, result, elapsed_ms, config
            )
            results.append({"status": "ok", "row": row})
        except Exception:
            results.append({
                "status": "error",
                "scenario_id": scenario_id,
                "planner_id": planner_id,
                "seed": seed,
                "traceback": traceback.format_exc(),
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run UAVBench v2 paper experiments.")
    p.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario IDs (default: all 3 OSM)",
    )
    p.add_argument(
        "--planners", type=str, default=None,
        help="Comma-separated planner IDs (default: all 5)",
    )
    p.add_argument(
        "--seeds", type=int, default=30,
        help="Number of seeds per (scenario, planner) pair (default: 30)",
    )
    p.add_argument(
        "--output", type=str, default=os.path.join(OUTPUT_DIR, "all_episodes.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--ablation", action="store_true",
        help="Run ablation studies after the main experiment",
    )
    return p.parse_args()


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    # Required for macOS/M1 — spawn instead of fork
    multiprocessing.set_start_method("spawn", force=True)

    from uavbench.planners import PLANNERS
    from uavbench.scenarios.loader import load_scenario
    from uavbench.scenarios.registry import list_scenarios

    args = _parse_args()

    scenarios = (
        [s.strip() for s in args.scenarios.split(",")]
        if args.scenarios else list_scenarios()
    )
    # Exclude backward-compat aliases (dstar_lite → incremental_astar)
    _PAPER_PLANNERS = ["astar", "periodic_replan", "aggressive_replan",
                       "incremental_astar", "apf"]
    planners = (
        [p.strip() for p in args.planners.split(",")]
        if args.planners else _PAPER_PLANNERS
    )
    n_seeds = args.seeds
    output_file = args.output
    total = len(scenarios) * len(planners) * n_seeds

    print(f"UAVBench v2 Paper Experiments")
    print(f"  Scenarios:  {len(scenarios)}")
    print(f"  Planners:   {len(planners)} — {', '.join(planners)}")
    print(f"  Seeds:      {n_seeds}")
    print(f"  Total:      {total} episodes")
    print(f"  Workers:    6 (Apple M1)")
    print(f"  Output:     {output_file}")
    print()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Build work units: (scenario_id, planner_id, n_seeds)
    work_units = [
        (scenario_id, planner_id, n_seeds)
        for scenario_id in scenarios
        for planner_id in planners
    ]

    attempted = 0
    completed = 0
    errors: list[dict] = []
    wall_start = time.perf_counter()

    MAX_WORKERS = 6  # Apple M1: 4 perf + 2 efficiency, leaves 2 for OS

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_run_scenario_planner_block, wu): wu
                for wu in work_units
            }
            for future in as_completed(futures):
                wu = futures[future]
                try:
                    block_results = future.result()
                except Exception as e:
                    # Entire block failed
                    scenario_id, planner_id, _ = wu
                    errors.append({
                        "scenario_id": scenario_id,
                        "planner_id": planner_id,
                        "seed": "ALL",
                        "traceback": str(e),
                    })
                    attempted += n_seeds
                    continue

                for item in block_results:
                    attempted += 1
                    if item["status"] == "ok":
                        writer.writerow(item["row"])
                        f.flush()
                        completed += 1
                    else:
                        errors.append(item)

                # Progress update
                wall_elapsed = time.perf_counter() - wall_start
                pct = attempted / total
                eta_s = (wall_elapsed / attempted * (total - attempted)) if attempted > 0 else 0
                eta_m = eta_s / 60
                scenario_id, planner_id, _ = wu
                print(
                    f"\r[{attempted}/{total}] ({100*pct:.0f}%) "
                    f"| Elapsed: {wall_elapsed/60:.1f}m "
                    f"| ETA: {eta_m:.0f}m "
                    f"| Last: {planner_id}/{scenario_id.replace('osm_', '')}",
                    end="", flush=True,
                )

    print()  # newline after progress

    wall_total = time.perf_counter() - wall_start

    # Error log
    if errors:
        with open(ERROR_LOG, "w") as ef:
            for err in errors:
                ef.write(
                    f"--- {err['scenario_id']} / {err['planner_id']} / seed={err['seed']} ---\n"
                )
                ef.write(err.get("traceback", "unknown error"))
                ef.write("\n")

    print(f"\nCompleted: {completed}/{total}")
    print(f"Errors:    {len(errors)}")
    if errors:
        print(f"Error log: {ERROR_LOG}")
    print(f"Results:   {output_file}")

    # Write runtime profile
    import datetime
    import platform

    profile = {
        "total_wall_clock_seconds": round(wall_total, 2),
        "total_episodes": total,
        "completed_episodes": completed,
        "failed_episodes": len(errors),
        "avg_time_per_episode_ms": (
            round(wall_total * 1000.0 / completed, 2) if completed else 0.0
        ),
        "seeds": n_seeds,
        "planners": planners,
        "scenarios": scenarios,
        "max_workers": MAX_WORKERS,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    profile_path = os.path.join(OUTPUT_DIR, "runtime_profile.json")
    with open(profile_path, "w") as pf:
        json.dump(profile, pf, indent=2)
    print(f"Profile:   {profile_path}")

    # Ablation studies (optional, after main run)
    if args.ablation:
        import subprocess
        print(f"\n{'='*70}")
        print("ABLATION STUDIES")
        print(f"{'='*70}")
        subprocess.run(
            [sys.executable, "scripts/run_ablation_studies.py",
             "--seeds", str(n_seeds)],
            check=True,
        )


if __name__ == "__main__":
    main()
