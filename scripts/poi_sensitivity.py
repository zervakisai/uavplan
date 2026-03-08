"""POI stuck threshold sensitivity analysis.

Tests _POI_STUCK_LIMIT at values 20, 30, 50 to verify benchmark results
are robust to threshold choice. Patches the source, runs episodes, restores.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

THRESHOLDS = [20, 30, 50]
SEEDS = list(range(5))
SCENARIOS = [
    "osm_penteli_pharma_delivery_medium",
    "osm_piraeus_flood_rescue_medium",
    "osm_downtown_fire_surveillance_medium",
]
PLANNERS_TO_TEST = ["astar", "aggressive_replan"]


def main() -> int:
    import uavbench.benchmark.runner as runner_mod

    src_path = Path(runner_mod.__file__)
    original_src = src_path.read_text()

    print("POI Stuck Threshold Sensitivity Analysis")
    print("=" * 60)
    print(f"Seeds: {SEEDS}, Planners: {PLANNERS_TO_TEST}")

    all_results: dict[int, dict[str, float]] = {}

    try:
        for threshold in THRESHOLDS:
            print(f"\n--- Threshold = {threshold} ---")

            # Patch _POI_STUCK_LIMIT
            patched = original_src.replace(
                "_POI_STUCK_LIMIT = 30",
                f"_POI_STUCK_LIMIT = {threshold}",
            )
            src_path.write_text(patched)
            importlib.reload(runner_mod)

            results: dict[str, float] = {}
            for sc in SCENARIOS:
                short = sc.split("_")[1]  # penteli/piraeus/downtown
                for planner in PLANNERS_TO_TEST:
                    successes = 0
                    for seed in SEEDS:
                        result = runner_mod.run_episode(sc, planner, seed=seed)
                        if result.metrics["termination_reason"] == "success":
                            successes += 1
                    rate = successes / len(SEEDS)
                    results[f"{short}_{planner}"] = rate
                    print(f"  {short}/{planner}: {rate:.0%} ({successes}/{len(SEEDS)})")

            all_results[threshold] = results
    finally:
        # Always restore original source
        src_path.write_text(original_src)
        importlib.reload(runner_mod)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("-" * 60)
    all_ok = True
    for threshold in THRESHOLDS:
        astar_max = max(
            v for k, v in all_results[threshold].items() if "astar" in k
        )
        adaptive_min = min(
            v for k, v in all_results[threshold].items()
            if "aggressive" in k
        )
        ok = astar_max == 0 and adaptive_min > 0
        if not ok:
            all_ok = False
        print(
            f"  T={threshold}: A* max={astar_max:.0%}, "
            f"adaptive min={adaptive_min:.0%} "
            f"[{'PASS' if ok else 'FAIL'}]"
        )

    print(f"\nVerdict: {'ROBUST — threshold choice does not affect SC-1' if all_ok else 'SENSITIVE — threshold affects results'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
