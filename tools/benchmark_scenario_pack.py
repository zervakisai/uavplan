#!/usr/bin/env python3
"""Run the full UAVBench scenario pack and produce comparison tables.

Runs all 20 OSM Athens scenarios with specified planners, aggregates
metrics, prints a comparison table, and optionally saves CSV results.

Usage:
  python tools/benchmark_scenario_pack.py --planners astar --trials 3 --output outputs/
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from uavbench.cli.benchmark import run_planner_once, scenario_path, aggregate

# All 20 OSM scenarios in the pack
SCENARIO_PACK = [
    # 1. Wildfire WUI
    "osm_athens_wildfire_easy",
    "osm_athens_wildfire_medium",
    "osm_athens_wildfire_hard",
    # 2. Emergency Response
    "osm_athens_emergency_easy",
    "osm_athens_emergency_medium",
    "osm_athens_emergency_hard",
    # 3. Port Security
    "osm_athens_port_easy",
    "osm_athens_port_medium",
    "osm_athens_port_hard",
    # 4. Combined Crisis
    "osm_athens_crisis_hard",
    # 5. Search & Rescue
    "osm_athens_sar_easy",
    "osm_athens_sar_medium",
    "osm_athens_sar_hard",
    # 6. Infrastructure Patrol
    "osm_athens_infrastructure_easy",
    "osm_athens_infrastructure_medium",
    "osm_athens_infrastructure_hard",
    # 7. Border Surveillance
    "osm_athens_border_easy",
    "osm_athens_border_medium",
    "osm_athens_border_hard",
    # 8. Communications-Denied
    "osm_athens_comms_denied_hard",
]

METRIC_IDS = ["path_length", "success_rate", "constraint_violations"]


def run_scenario_pack(
    planner_ids: list[str],
    trials: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    """Run all scenarios × planners × trials and collect results."""
    all_rows: list[dict[str, Any]] = []

    total = len(SCENARIO_PACK) * len(planner_ids)
    done = 0

    for scenario_id in SCENARIO_PACK:
        sp = scenario_path(scenario_id)
        if not sp.exists():
            print(f"  [SKIP] {scenario_id}: config not found at {sp}")
            done += len(planner_ids)
            continue

        for planner_id in planner_ids:
            done += 1
            print(f"  [{done}/{total}] {scenario_id} x {planner_id} ({trials} trials)...",
                  end="", flush=True)

            per_trial: list[dict[str, Any]] = []
            t0 = time.time()

            for t in range(trials):
                seed = seed_base + (hash(scenario_id) & 0xFFFF) + (hash(planner_id) & 0x0FFF) + t
                try:
                    r = run_planner_once(scenario_id, planner_id, seed=seed)
                except Exception as e:
                    r = {
                        "scenario": scenario_id,
                        "planner": planner_id,
                        "seed": int(seed),
                        "success": False,
                        "constraint_violations": 1,
                        "path_length": 0,
                        "path": None,
                        "error": str(e),
                    }
                per_trial.append(r)

            elapsed = time.time() - t0
            metrics = aggregate(per_trial, METRIC_IDS)

            row = {
                "scenario": scenario_id,
                "planner": planner_id,
                "trials": trials,
                "success_rate": metrics.get("success_rate", 0.0),
                "avg_path_length": metrics.get("avg_path_length", float("nan")),
                "avg_violations": metrics.get("avg_constraint_violations", 0.0),
                "time_s": round(elapsed, 2),
            }
            all_rows.append(row)

            sr = row["success_rate"]
            pl = row["avg_path_length"]
            pl_str = f"{pl:.0f}" if not np.isnan(pl) else "n/a"
            print(f" SR={sr:.0%} PL={pl_str} ({elapsed:.1f}s)")

    return all_rows


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Scenario':<40} {'Planner':<10} {'SR':>6} {'Path':>8} {'Viol':>6} {'Time':>7}")
    print("-" * 90)

    for row in rows:
        pl = row["avg_path_length"]
        pl_str = f"{pl:.0f}" if not np.isnan(pl) else "n/a"
        print(
            f"{row['scenario']:<40} "
            f"{row['planner']:<10} "
            f"{row['success_rate']:>5.0%} "
            f"{pl_str:>8} "
            f"{row['avg_violations']:>6.2f} "
            f"{row['time_s']:>6.1f}s"
        )

    print("=" * 90)


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "planner", "trials", "success_rate",
                   "avg_path_length", "avg_violations", "time_s"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full UAVBench scenario pack.",
    )
    parser.add_argument(
        "--planners", type=str, default="astar",
        help="Comma-separated planner IDs (default: astar).",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Trials per scenario/planner pair (default: 1).",
    )
    parser.add_argument(
        "--seed-base", type=int, default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output", type=str, default="",
        metavar="DIR",
        help="Save CSV results to DIR/scenario_pack_results.csv.",
    )
    args = parser.parse_args()

    planner_ids = [p.strip() for p in args.planners.split(",") if p.strip()]

    print(f"[UAVBench Scenario Pack]")
    print(f"  Scenarios: {len(SCENARIO_PACK)}")
    print(f"  Planners: {planner_ids}")
    print(f"  Trials: {args.trials}")
    print(f"  Total runs: {len(SCENARIO_PACK) * len(planner_ids) * args.trials}")
    print()

    rows = run_scenario_pack(planner_ids, args.trials, args.seed_base)
    print_comparison_table(rows)

    if args.output:
        save_csv(rows, Path(args.output) / "scenario_pack_results.csv")


if __name__ == "__main__":
    main()
