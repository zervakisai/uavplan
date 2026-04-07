#!/usr/bin/env python3
"""Generate ALL final paper outputs: tables + figure fixes + new figures.

Usage:
    python scripts/gen_paper_final.py                # everything
    python scripts/gen_paper_final.py --tables-only  # just tables (seconds)
    python scripts/gen_paper_final.py --skip-slow    # skip 30-seed heatmap
    python scripts/gen_paper_final.py --only C1 C3   # specific figures
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

# Task registry: (id, label, module_name, is_slow)
TASKS = {
    # Tables
    "T1": ("Planner profiles (merged)", "gen_table_planner_profiles", False),
    "T2": ("Mission scorecard (per-scenario)", "gen_table_mission_scorecard", False),
    "T3": ("Risk-mission trade-off (NEW)", "gen_table_risk_tradeoff", False),
    "T4": ("Scenario overview (NEW)", "gen_table_scenario_overview", False),
    # Figure fixes
    "F1": ("Collapse cascade (enable_collapse!)", "gen_collapse_cascade_figure", False),
    "F2": ("Triage map (Renderer background)", "gen_triage_figure", False),
    "F3": ("Trajectory heatmap (30 seeds)", "gen_trajectory_heatmap_v2", True),
    # New figures
    "C1": ("5-planner trajectory comparison", "gen_trajectory_comparison_5panel", False),
    "C2": ("Risk perception 5-panel", "gen_risk_perception_figure", False),
    "C3": ("Traffic dynamics 3-panel", "gen_traffic_dynamics_figure", False),
    "C4": ("Coupled hazard timeline", "gen_coupled_hazard_timeline", False),
    "C5": ("Mission score decay curves", "gen_mission_score_decomposition", False),
}

TABLE_IDS = ["T1", "T2", "T3", "T4"]
FIX_IDS = ["F1", "F2", "F3"]
NEW_IDS = ["C1", "C2", "C3", "C4", "C5"]

# Expected outputs
TABLE_OUTPUTS = {
    "T1": "outputs/paper_tables/planner_profiles.tex",
    "T2": "outputs/paper_tables/mission_scorecard.tex",
    "T3": "outputs/paper_tables/risk_tradeoff.tex",
    "T4": "outputs/paper_tables/scenario_overview.tex",
}

FIGURE_OUTPUTS = {
    "F1": "outputs/paper_figures/collapse_cascade.png",
    "F2": "outputs/paper_figures/triage_under_fire.png",
    "F3": "outputs/paper_figures/trajectory_heatmap_5panel.png",
    "C1": "outputs/paper_figures/trajectory_comparison_5panel.png",
    "C2": "outputs/paper_figures/risk_perception_5panel.png",
    "C3": "outputs/paper_figures/traffic_dynamics.png",
    "C4": "outputs/paper_figures/coupled_hazard_timeline.png",
    "C5": "outputs/paper_figures/mission_score_decomposition.png",
}


def run_task(task_id: str, label: str, module_name: str) -> tuple[bool, float]:
    """Run a generation task. Returns (success, elapsed_seconds)."""
    print(f"\n{'=' * 60}")
    print(f"  [{task_id}] {label}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    try:
        # Clear sys.argv so sub-module argparse doesn't see master args
        saved_argv = sys.argv
        sys.argv = [module_name]
        try:
            if module_name in sys.modules:
                mod = importlib.reload(sys.modules[module_name])
            else:
                mod = importlib.import_module(module_name)
            mod.main()
        finally:
            sys.argv = saved_argv
        elapsed = time.perf_counter() - t0
        print(f"  [{task_id}] DONE in {elapsed:.1f}s")
        return True, elapsed
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"  [{task_id}] FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper outputs.")
    parser.add_argument("--tables-only", action="store_true",
                        help="Only generate tables (fast)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow tasks (30-seed heatmap)")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="Run only specific tasks (e.g., C1 C3 T1)")
    args = parser.parse_args()

    # Determine which tasks to run
    if args.only:
        task_ids = [t.upper() for t in args.only]
        # Validate
        for t in task_ids:
            if t not in TASKS:
                print(f"ERROR: Unknown task ID '{t}'. Valid: {sorted(TASKS.keys())}")
                sys.exit(1)
    elif args.tables_only:
        task_ids = TABLE_IDS
    else:
        task_ids = TABLE_IDS + FIX_IDS + NEW_IDS

    # Filter slow tasks
    if args.skip_slow:
        task_ids = [t for t in task_ids if not TASKS[t][2]]

    print(f"FLARE Paper Final Generation")
    print(f"  Tasks: {len(task_ids)} ({', '.join(task_ids)})")
    print()

    # Add scripts/ to path so imports work
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    total_t0 = time.perf_counter()
    results: dict[str, tuple[bool, float]] = {}

    for task_id in task_ids:
        label, module_name, is_slow = TASKS[task_id]
        success, elapsed = run_task(task_id, label, module_name)
        results[task_id] = (success, elapsed)

    total_elapsed = time.perf_counter() - total_t0

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  === Final Paper Outputs ===")
    print(f"{'=' * 60}")

    # Tables
    table_ok = sum(1 for t in TABLE_IDS if t in results and results[t][0])
    table_total = sum(1 for t in TABLE_IDS if t in results)
    if table_total > 0:
        print(f"  Tables:  {table_ok}/{table_total} "
              f"{'OK' if table_ok == table_total else 'PARTIAL'}")

    # Fixes
    fix_ok = sum(1 for t in FIX_IDS if t in results and results[t][0])
    fix_total = sum(1 for t in FIX_IDS if t in results)
    if fix_total > 0:
        print(f"  Fixes:   {fix_ok}/{fix_total} "
              f"{'OK' if fix_ok == fix_total else 'PARTIAL'}")

    # New figures
    new_ok = sum(1 for t in NEW_IDS if t in results and results[t][0])
    new_total = sum(1 for t in NEW_IDS if t in results)
    if new_total > 0:
        print(f"  New:     {new_ok}/{new_total} "
              f"{'OK' if new_ok == new_total else 'PARTIAL'}")

    mins, secs = divmod(int(total_elapsed), 60)
    print(f"  Time:    {mins}m {secs}s")

    # File listing
    print(f"\n  outputs/paper_tables/")
    for tid, path in TABLE_OUTPUTS.items():
        if tid not in results:
            continue
        p = ROOT / path
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name:<40s} {size_kb:.1f} KB  OK")
        else:
            print(f"    {p.name:<40s} MISSING")

    print(f"\n  outputs/paper_figures/")
    for tid, path in FIGURE_OUTPUTS.items():
        if tid not in results:
            continue
        p = ROOT / path
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name:<40s} {size_kb:.1f} KB  OK")
        else:
            print(f"    {p.name:<40s} MISSING")

    # Exit code
    all_ok = all(results[t][0] for t in results)
    if not all_ok:
        failed = [t for t in results if not results[t][0]]
        print(f"\n  FAILED tasks: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  All tasks completed successfully!")


if __name__ == "__main__":
    main()
