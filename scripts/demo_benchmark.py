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


def main():
    print("=" * 80)
    print("UAVBench Benchmark Demo")
    print("=" * 80)
    
    config = BenchmarkConfig(
        scenario_ids=[
            "urban_easy",
            "osm_athens_wildfire_easy",
            "osm_athens_emergency_easy",
        ],
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
    
    runner = BenchmarkRunner(config)
    aggregates = runner.run()
    
    print(f"\n✓ Demo complete!")
    print(f"  Results saved to: {config.output_dir}")
    print(f"  Episodes: {config.output_dir / 'episodes.jsonl'}")
    print(f"  Aggregates: {config.output_dir / 'aggregates.csv'}")


if __name__ == "__main__":
    main()
