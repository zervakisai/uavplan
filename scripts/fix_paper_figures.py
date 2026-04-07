#!/usr/bin/env python3
"""Fix all paper figure issues identified in post-review.

Runs individual figure-generation scripts in order, with options to
select specific fixes or skip slow tasks.

Usage:
    python scripts/fix_paper_figures.py              # all fixes
    python scripts/fix_paper_figures.py --only 1 2 3 # specific fixes
    python scripts/fix_paper_figures.py --skip-slow   # skip heatmap (Fix 5)
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

FIXES = {
    1: ("Collapse cascade 'No data' bug",
        "scripts/gen_collapse_cascade_figure.py", []),
    2: ("Triage map redesign",
        "scripts/gen_triage_figure.py", []),
    3: ("NEW 5-panel planner comparison",
        "scripts/gen_planner_comparison_5panel.py", []),
    4: ("Corridor blockage visibility",
        "scripts/gen_corridor_blockage_figure.py", []),
    5: ("Trajectory heatmap 10+ seeds",
        "scripts/gen_trajectory_heatmap_v2.py", ["--seeds"]),
}

POLISH = {
    6: ("Risk perception fire contours",
        "scripts/gen_risk_perception_figure.py", []),
    7: ("Mission score all planners",
        "scripts/gen_mission_score_decomposition.py", []),
    8: ("Scenario overview (start/goal markers)",
        "scripts/generate_paper_snapshots.py",
        ["--skip-families", "--skip-comparison"]),
}

SLOW_FIXES = {5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="+",
                    help="Run only specific fix numbers (1-8)")
    ap.add_argument("--skip-slow", action="store_true",
                    help="Skip slow tasks (heatmap)")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Seeds for trajectory heatmap (default: 10)")
    args = ap.parse_args()

    all_tasks = {**FIXES, **POLISH}
    targets = args.only or sorted(all_tasks)
    if args.skip_slow:
        targets = [t for t in targets if t not in SLOW_FIXES]

    print(f"{'=' * 60}")
    print(f"  FLARE Paper Figure Fixes — {len(targets)} tasks")
    print(f"{'=' * 60}\n")

    results = {}
    total_t0 = time.perf_counter()

    for tid in sorted(targets):
        if tid not in all_tasks:
            print(f"[{tid}] Unknown fix number, skipping")
            continue

        name, script, extra_args = all_tasks[tid]
        print(f"[Fix {tid}] {name}...")

        cmd = [sys.executable, script] + extra_args
        if tid == 5:
            cmd += [str(args.seeds)]

        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        ok = r.returncode == 0
        results[tid] = ok
        marker = "DONE" if ok else "FAILED"
        print(f"        {marker} ({elapsed:.1f}s)")

        if ok and r.stdout:
            for line in r.stdout.strip().split("\n"):
                if "Saved:" in line or "ERROR" in line:
                    print(f"        {line.strip()}")
        if not ok and r.stderr:
            for line in r.stderr.strip().split("\n")[-5:]:
                print(f"        {line}")

    # Copy updated figures to paper/figures/ for LaTeX
    out_dir = Path("outputs/paper_figures")
    paper_dir = Path("paper/figures")
    if out_dir.exists() and paper_dir.exists():
        print("\n  Syncing to paper/figures/...")
        for pdf in out_dir.glob("*.pdf"):
            dst = paper_dir / pdf.name
            shutil.copy2(pdf, dst)
            print(f"    {pdf.name} -> paper/figures/")

    # Summary
    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'=' * 60}")
    passed = sum(results.values())
    print(f"  {passed}/{len(results)} fixes succeeded ({total_elapsed:.0f}s total)")

    expected = [
        "collapse_cascade", "triage_under_fire",
        "planner_comparison_5panel", "corridor_blockage_dramatic",
        "trajectory_heatmap_5panel", "risk_perception_5panel",
        "mission_score_decomposition", "scenario_overview_3panel",
    ]
    print(f"\n  Output files:")
    for name in expected:
        pdf = out_dir / f"{name}.pdf"
        png = out_dir / f"{name}.png"
        if pdf.exists():
            sz = pdf.stat().st_size // 1024
            print(f"    {name}.pdf  ({sz} KB)")
        elif png.exists():
            sz = png.stat().st_size // 1024
            print(f"    {name}.png  ({sz} KB)")
        else:
            print(f"    {name}  MISSING")

    print(f"{'=' * 60}")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
