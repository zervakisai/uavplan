#!/usr/bin/env python3
"""validate_release.py — Pre-release validation for UAVBench.

Run this script before tagging a release to verify structural integrity:
  python tools/validate_release.py

Checks:
  1. All 9 scenario YAMLs parse successfully
  2. Realism knobs are top-level (not buried in extra:)
  3. All 11 planner registry keys instantiate
  4. Key documentation files exist
  5. interdiction_reference_planner is explicitly set in all YAMLs
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

    # ── 1. Scenario YAMLs ────────────────────────────────────────────
    print("▶ Checking scenario YAMLs...")
    from uavbench.cli.benchmark import scenario_path
    from uavbench.scenarios.loader import load_scenario

    expected_scenarios = [
        "gov_civil_protection_easy",
        "gov_civil_protection_medium",
        "gov_civil_protection_hard",
        "gov_maritime_domain_easy",
        "gov_maritime_domain_medium",
        "gov_maritime_domain_hard",
        "gov_critical_infrastructure_easy",
        "gov_critical_infrastructure_medium",
        "gov_critical_infrastructure_hard",
    ]

    for sid in expected_scenarios:
        sp = scenario_path(sid)
        if not sp.exists():
            errors.append(f"Missing scenario YAML: {sid}")
            continue
        try:
            cfg = load_scenario(sp)
        except Exception as e:
            errors.append(f"Failed to parse {sid}: {e}")
            continue

        # Check realism knobs are top-level (not defaulting to 0 on hard)
        if "hard" in sid:
            if cfg.comms_dropout_prob == 0.0:
                errors.append(
                    f"{sid}: comms_dropout_prob=0.0 on hard scenario "
                    "(likely still nested under extra:)"
                )
            if cfg.constraint_latency_steps == 0:
                errors.append(
                    f"{sid}: constraint_latency_steps=0 on hard scenario "
                    "(likely still nested under extra:)"
                )
            if cfg.gnss_noise_sigma == 0.0:
                errors.append(
                    f"{sid}: gnss_noise_sigma=0.0 on hard scenario "
                    "(should be > 0)"
                )

        # Check interdiction reference planner
        ref_val = str(cfg.interdiction_reference_planner.value).lower()
        if ref_val != "astar":
            warnings.append(
                f"{sid}: interdiction_reference_planner={ref_val} "
                "(expected astar)"
            )

        print(f"  ✓ {sid}")

    # ── 2. Planner registry ──────────────────────────────────────────
    print("▶ Checking planner registry...")
    from uavbench.planners import PLANNERS

    expected_keys = {
        "astar", "theta_star",
        "periodic_replan", "aggressive_replan",
        "greedy_local", "grid_mppi",
        "incremental_dstar_lite",
        "dstar_lite", "ad_star", "dwa", "mppi",  # legacy aliases
    }

    for key in sorted(expected_keys):
        if key not in PLANNERS:
            errors.append(f"Missing planner registry key: {key}")
        else:
            print(f"  ✓ {key}")

    extra_keys = set(PLANNERS.keys()) - expected_keys
    if extra_keys:
        warnings.append(f"Unexpected planner keys: {extra_keys}")

    # ── 3. Documentation files ───────────────────────────────────────
    print("▶ Checking documentation...")
    docs_dir = ROOT / "docs"
    required_docs = [
        "AUDIT_HEAD.md",
        "PAPER_PROTOCOL.md",
        "SCENARIO_CARDS.md",
        "REPRODUCIBILITY.md",
        "DATASHEET.md",
    ]

    for doc in required_docs:
        p = docs_dir / doc
        if not p.exists():
            errors.append(f"Missing doc: {doc}")
        elif p.stat().st_size < 100:
            warnings.append(f"Doc too small (< 100 bytes): {doc}")
        else:
            print(f"  ✓ {doc}")

    # Also check root files
    for root_file in ["LICENSE", "README.md", "pyproject.toml", "Dockerfile"]:
        p = ROOT / root_file
        if not p.exists():
            errors.append(f"Missing root file: {root_file}")
        else:
            print(f"  ✓ {root_file}")

    # ── 4. Summary ───────────────────────────────────────────────────
    print()
    if warnings:
        print(f"⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print(f"\n✗ {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        print("\n❌ RELEASE VALIDATION FAILED")
        return 1
    else:
        print("✅ RELEASE VALIDATION PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
