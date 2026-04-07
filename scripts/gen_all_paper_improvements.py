#!/usr/bin/env python3
"""Generate ALL paper improvement figures and tables.

Usage:
    python scripts/gen_all_paper_improvements.py                  # all tasks
    python scripts/gen_all_paper_improvements.py --skip-long      # skip heatmap + Shapley
    python scripts/gen_all_paper_improvements.py --seeds 5        # fewer seeds
    python scripts/gen_all_paper_improvements.py --only 1 3 5     # specific tasks
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

TASKS = {
    1:  ("Risk profile table",           "scripts/gen_risk_profile_table.py"),
    2:  ("Planner characteristics table", "scripts/gen_planner_characteristics_table.py"),
    3:  ("Mission score decomposition",   "scripts/gen_mission_score_decomposition.py"),
    4:  ("Mission impact scatter v2",     "scripts/gen_mission_impact_scatter_v2.py"),
    5:  ("Risk perception 5-panel",       "scripts/gen_risk_perception_figure.py"),
    6:  ("Corridor blockage dramatic",    "scripts/gen_corridor_blockage_figure.py"),
    7:  ("Wind-driven fire",              "scripts/gen_wind_fire_figure.py"),
    8:  ("Collapse cascade",              "scripts/gen_collapse_cascade_figure.py"),
    9:  ("Triage under fire",             "scripts/gen_triage_figure.py"),
    10: ("Trajectory heatmap",            "scripts/gen_trajectory_heatmap_v2.py"),
    11: ("Shapley attribution",           "scripts/gen_shapley_figure.py"),
}
SLOW_TASKS = {10, 11}


def main():
    ap = argparse.ArgumentParser(description="Generate all paper improvement figures.")
    ap.add_argument("--skip-long", action="store_true",
                    help="Skip slow tasks (heatmap + Shapley)")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Seeds for heavy-compute tasks (default: 10)")
    ap.add_argument("--only", type=int, nargs="+",
                    help="Run only specific task numbers")
    args = ap.parse_args()

    targets = args.only or sorted(TASKS)
    if args.skip_long:
        targets = [t for t in targets if t not in SLOW_TASKS]

    print(f"{'=' * 60}")
    print(f"  FLARE Paper Improvements — {len(targets)} tasks")
    print(f"{'=' * 60}\n")

    results = {}
    total_t0 = time.perf_counter()

    for tid in targets:
        name, script = TASKS[tid]
        print(f"[{tid:2d}/11] {name}...")
        t0 = time.perf_counter()
        cmd = [sys.executable, script]
        if tid in SLOW_TASKS:
            cmd += ["--seeds", str(args.seeds)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        ok = r.returncode == 0
        results[tid] = ok
        status = "Done" if ok else "FAILED"
        marker = "\u2713" if ok else "\u2717"
        print(f"        {marker} {status} ({elapsed:.1f}s)")
        if not ok and r.stderr:
            for line in r.stderr.strip().split("\n")[-3:]:
                print(f"        {line}")
        if ok and r.stdout:
            # Print save lines
            for line in r.stdout.strip().split("\n"):
                if "Saved:" in line or "ERROR" in line:
                    print(f"        {line.strip()}")

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    print(f"\n{'=' * 60}")
    passed = sum(results.values())
    print(f"  {passed}/{len(results)} tasks succeeded ({total_elapsed:.0f}s total)\n")

    for d in ["outputs/paper_tables", "outputs/paper_figures"]:
        p = Path(d)
        if p.exists():
            for f in sorted(p.iterdir()):
                size = f.stat().st_size
                if size < 1_000_000:
                    val = size // 1024
                    unit = "KB"
                else:
                    val = size // (1024 * 1024)
                    unit = "MB"
                print(f"    {f.name:45s} {val:>5}{unit}")

    print(f"{'=' * 60}")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
