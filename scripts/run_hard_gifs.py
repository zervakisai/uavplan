#!/usr/bin/env python3
"""Run all hard scenarios with adaptive planners and export animated GIFs.

Produces styled basemap GIFs with:
  - Title card (mission name, agency, planner)
  - OSM basemap (landuse fills, 3-tier roads, hillshade, water coastlines)
  - Mission POI icons (fire zone, evac corridor, distress, inspection)
  - ODbL attribution
  - Frame sub-sampling for manageable file sizes

Usage:
    python scripts/run_hard_gifs.py

Outputs GIFs to: outputs/hard_gifs/<scenario>_<planner>.gif
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure we run from the project root (data/maps paths are relative)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# Ensure project root is importable
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavbench.cli.benchmark import run_dynamic_episode

HARD_SCENARIOS = [
    "gov_civil_protection_hard",
    "gov_maritime_domain_hard",
    "gov_critical_infrastructure_hard",
]

# Adaptive planners that support replanning (best suited for hard/dynamic scenarios)
PLANNERS = ["periodic_replan", "aggressive_replan"]

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "hard_gifs"
RENDER_DPI = 80  # good balance: readable text, manageable file size


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(HARD_SCENARIOS) * len(PLANNERS)
    done = 0

    for scenario_id in HARD_SCENARIOS:
        for planner_id in PLANNERS:
            done += 1
            gif_path = OUT_DIR / f"{scenario_id}_{planner_id}.gif"
            print(
                f"\n{'='*70}\n"
                f"[{done}/{total}] {scenario_id} × {planner_id}\n"
                f"{'='*70}"
            )

            t0 = time.perf_counter()
            try:
                result = run_dynamic_episode(
                    scenario_id,
                    planner_id,
                    seed=SEED,
                    render_gif=str(gif_path),
                    render_dpi=RENDER_DPI,
                )
                elapsed = time.perf_counter() - t0

                success = result.get("success", False)
                steps = result.get("episode_steps", 0)
                replans = result.get("total_replans", 0)
                violations = result.get("constraint_violations", 0)
                reason = result.get("termination_reason", "")

                status = "✅ SUCCESS" if success else f"❌ FAIL ({reason})"
                print(
                    f"  {status}  |  steps={steps}  replans={replans}  "
                    f"violations={violations}  time={elapsed:.1f}s"
                )

                if gif_path.exists():
                    size_mb = gif_path.stat().st_size / (1024 * 1024)
                    print(f"  📎 GIF saved: {gif_path.name} ({size_mb:.1f} MB)")
                else:
                    print(f"  ⚠️  GIF not created")

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"  💥 ERROR after {elapsed:.1f}s: {exc}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"All GIFs → {OUT_DIR}")
    gifs = sorted(OUT_DIR.glob("*.gif"))
    for g in gifs:
        size_mb = g.stat().st_size / (1024 * 1024)
        print(f"  {g.name}  ({size_mb:.1f} MB)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
