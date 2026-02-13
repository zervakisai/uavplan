#!/usr/bin/env python3
"""Quick benchmark demonstration: run 3 scenarios × 2 planners × 2 seeds.

Usage:
    python scripts/demo_benchmark.py

This script shows the full benchmark pipeline end-to-end.
"""

import sys
from pathlib import Path

# Ensure uavbench is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uavbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner


def _demo_scenarios() -> list[str]:
    """Return demo scenarios that are runnable in the current checkout.

    If OSM raster tiles are present, include dynamic OSM scenarios.
    Otherwise, use synthetic scenarios only so demo succeeds out-of-the-box.
    """
    maps_dir = Path("data/maps")
    has_penteli = (maps_dir / "Penteli.npz").exists()
    has_downtown = (maps_dir / "Downtown.npz").exists()

    if has_penteli and has_downtown:
        return [
            "urban_easy",
            "osm_athens_wildfire_wui_easy_penteli",
            "osm_athens_emergency_response_easy_downtown",
        ]

    return ["urban_easy", "urban_medium", "urban_hard"]


def main():
    print("=" * 80)
    print("UAVBench Benchmark Demo")
    print("=" * 80)

    scenario_ids = _demo_scenarios()

    config = BenchmarkConfig(
        scenario_ids=scenario_ids,
        planner_ids=[
            "astar",
            "theta_star",
        ],
        seeds=[0, 1],
        output_dir=Path("./demo_results"),
        verbose=True,
    )

    print(f"\nBenchmark Configuration:")
    print(f"  Scenarios: {len(config.scenario_ids)} ({', '.join(config.scenario_ids)})")
    print(f"  Planners: {len(config.planner_ids)} ({', '.join(config.planner_ids)})")
    print(f"  Seeds: {config.seeds}")
    print(f"  Total runs: {len(config.scenario_ids) * len(config.planner_ids) * len(config.seeds)}")
    print(f"  Output: {config.output_dir}")

    if scenario_ids[1].startswith("urban_"):
        print("  Note: OSM tile rasters not found in data/maps; using synthetic scenarios only.")

    runner = BenchmarkRunner(config)
    runner.run()

    print(f"\n✓ Demo complete!")
    print(f"  Results saved to: {config.output_dir}")
    print(f"  Episodes: {config.output_dir / 'episodes.jsonl'}")
    print(f"  Aggregates: {config.output_dir / 'aggregates.csv'}")


if __name__ == "__main__":
    main()
