#!/usr/bin/env python3
"""validate_release.py — Pre-release validation for UAVBench.

Run this script before tagging a release to verify structural integrity:
  python tools/validate_release.py

Checks:
  1. All 3 OSM scenario YAMLs parse successfully
  2. All 5 planner registry keys instantiate
  3. Key documentation files exist
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    # -- 1. Scenario YAMLs --
    print("Checking scenario YAMLs...")
    from uavbench.scenarios.loader import load_scenario
    from uavbench.scenarios.registry import list_scenarios

    all_scenarios = list_scenarios()
    expected_scenarios = [
        "osm_penteli_pharma_delivery_medium",
        "osm_piraeus_urban_rescue_medium",
        "osm_downtown_fire_surveillance_medium",
    ]

    for sid in expected_scenarios:
        if sid not in all_scenarios:
            errors.append(f"Missing scenario: {sid}")
            continue
        try:
            cfg = load_scenario(sid)
            print(f"  OK {sid}")
        except Exception as e:
            errors.append(f"Failed to parse {sid}: {e}")

    extra = set(all_scenarios) - set(expected_scenarios)
    if extra:
        warnings.append(f"Unexpected scenarios: {extra}")

    # -- 2. Planner registry --
    print("Checking planner registry...")
    from uavbench.planners import PLANNERS

    expected_planners = {"astar", "periodic_replan", "aggressive_replan", "dstar_lite", "apf"}

    for key in sorted(expected_planners):
        if key not in PLANNERS:
            errors.append(f"Missing planner registry key: {key}")
        else:
            print(f"  OK {key}")

    extra_keys = set(PLANNERS.keys()) - expected_planners
    if extra_keys:
        warnings.append(f"Unexpected planner keys: {extra_keys}")

    # -- 3. Documentation files --
    print("Checking documentation...")
    docs_dir = ROOT / "docs"
    for doc in ["CONTRACTS.md", "DECISIONS.md"]:
        p = docs_dir / doc
        if not p.exists():
            errors.append(f"Missing doc: {doc}")
        else:
            print(f"  OK {doc}")

    for root_file in ["README.md", "pyproject.toml"]:
        p = ROOT / root_file
        if not p.exists():
            errors.append(f"Missing root file: {root_file}")
        else:
            print(f"  OK {root_file}")

    # -- Summary --
    print()
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARNING: {w}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  ERROR: {e}")
        print("\nRELEASE VALIDATION FAILED")
        return 1
    else:
        print("RELEASE VALIDATION PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
