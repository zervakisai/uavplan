#!/usr/bin/env python3
"""Verification: confirm fire+traffic+NFZ all block movement after YAML fix.

Uses run_dynamic_episode to get proper metrics including zone_violation_count.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavbench.cli.benchmark import run_dynamic_episode

SCENARIO = "gov_civil_protection_medium"
PLANNERS = ["astar", "aggressive_replan"]
SEEDS = [0, 1, 2]

def main():
    results = []
    for planner_id in PLANNERS:
        for seed in SEEDS:
            print(f"\n{'='*60}", flush=True)
            print(f"Running: {SCENARIO} / {planner_id} / seed={seed}", flush=True)
            print(f"{'='*60}", flush=True)

            r = run_dynamic_episode(SCENARIO, planner_id, seed=seed)

            summary = {
                "planner": planner_id,
                "seed": seed,
                "success": r.get("success", False),
                "termination": r.get("termination_reason", "?"),
                "steps": r.get("episode_steps", 0),
                "path_length": r.get("path_length", 0),
                "replans": r.get("total_replans", 0),
                "zone_violations": r.get("zone_violation_count", 0),
                "planning_ms": round(r.get("planning_time_ms", 0), 1),
                "collision_terminated": r.get("collision_terminated", False),
                "fire_blocks": r.get("fire_blocks_movement", "?"),
                "traffic_blocks": r.get("traffic_blocks_movement", "?"),
            }
            results.append(summary)

            print(f"\n  Result: {summary['termination']}", flush=True)
            print(f"  Steps={summary['steps']}, Replans={summary['replans']}, "
                  f"Zone violations={summary['zone_violations']}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Planner':>20s} {'Seed':>4s} {'Success':>8s} {'Term':>15s} {'Steps':>6s} "
          f"{'Replans':>7s} {'Violations':>10s} {'Path':>6s}", flush=True)
    print("-"*70, flush=True)
    for s in results:
        print(f"{s['planner']:>20s} {s['seed']:>4d} {str(s['success']):>8s} "
              f"{s['termination']:>15s} {s['steps']:>6d} {s['replans']:>7d} "
              f"{s['zone_violations']:>10d} {s['path_length']:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
