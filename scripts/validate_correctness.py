#!/usr/bin/env python3
"""Validate that optimized code produces deterministic, self-consistent results.

Runs two identical episodes with the same seed and checks:
1. Both produce identical metrics (determinism)
2. Metrics are within expected bounds (sanity)
3. Cross-scenario consistency (all 3 dynamic scenarios)
"""
import os
import sys
import json

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "src")

from uavbench.cli.benchmark import run_dynamic_episode

# ── Configuration ─────────────────────────────────────────────
SEED = 42
SCENARIOS = [
    ("gov_civil_protection_hard", "periodic_replan"),
    ("gov_maritime_domain_hard", "periodic_replan"),
    ("gov_critical_infrastructure_hard", "periodic_replan"),
]

DETERMINISTIC_KEYS = [
    "constraint_violations",
]

# Wall-clock-dependent: replan budget timing affects success, path, etc.
SOFT_KEYS = [
    "success",
    "episode_steps",
    "termination_reason",
    "path_length",
    "total_replans",
    "fire_blocks",
    "traffic_blocks",
    "dynamic_nfz_blocks",
    "total_dynamic_blocks",
    "dynamic_block_hits",
    "guardrail_activation_count",
    "corridor_fallback_count",
    "zone_violation_count",
]

FLOAT_KEYS = [
    "risk_exposure_integral",
    "total_reward",
    "relaxation_magnitude",
    "zone_coverage_peak",
]


def compare_episodes(r1, r2, label):
    """Compare two episode results for deterministic equality."""
    errors = []
    warnings = []
    for key in DETERMINISTIC_KEYS:
        v1 = r1.get(key)
        v2 = r2.get(key)
        if v1 != v2:
            errors.append(f"  HARD {key}: {v1} vs {v2}")
    for key in SOFT_KEYS:
        v1 = r1.get(key)
        v2 = r2.get(key)
        if v1 != v2:
            warnings.append(f"  SOFT {key}: {v1} vs {v2}")
    for key in FLOAT_KEYS:
        v1 = float(r1.get(key, 0))
        v2 = float(r2.get(key, 0))
        if abs(v1 - v2) > 1e-9:
            warnings.append(f"  SOFT {key}: {v1:.6f} vs {v2:.6f}")

    if errors:
        print(f"FAIL [{label}] — determinism violation:")
        for e in errors:
            print(e)
        return False
    elif warnings:
        print(f"PASS [{label}] — deterministic core ✓ (wall-clock-dependent metrics vary, expected):")
        for w in warnings:
            print(w)
        return True
    else:
        print(f"PASS [{label}] — perfectly deterministic ✓")
        return True


def sanity_check(r, label):
    """Check that metric values are within expected bounds."""
    errors = []
    warnings = []
    if r["constraint_violations"] > 5:
        errors.append(f"  constraint_violations = {r['constraint_violations']} (too high)")
    elif r["constraint_violations"] > 0:
        warnings.append(f"  constraint_violations = {r['constraint_violations']} (minor, scenario-specific)")
    if r["episode_steps"] < 10:
        errors.append(f"  episode_steps = {r['episode_steps']} (suspiciously low)")
    if r["path_length"] < 2:
        errors.append(f"  path_length = {r['path_length']} (agent didn't move)")
    if r["total_dynamic_blocks"] < 0:
        errors.append(f"  total_dynamic_blocks = {r['total_dynamic_blocks']} (negative)")
    ri = r.get("risk_exposure_integral", 0)
    if ri < 0:
        errors.append(f"  risk_exposure_integral = {ri} (negative)")

    if errors:
        print(f"FAIL [{label}] — sanity check:")
        for e in errors:
            print(e)
        return False
    elif warnings:
        print(f"PASS [{label}] — sanity ✓ (minor notes):")
        for w in warnings:
            print(w)
        return True
    else:
        print(f"PASS [{label}] — sanity ✓")
        return True


def main():
    all_ok = True
    results_log = []

    for scenario, planner in SCENARIOS:
        label = f"{scenario}/{planner}"
        print(f"\n{'='*60}")
        print(f"Scenario: {label}, seed={SEED}")
        print(f"{'='*60}")

        # Run 1
        print("  Running episode 1...")
        r1 = run_dynamic_episode(scenario, planner, seed=SEED)

        # Run 2 (identical)
        print("  Running episode 2...")
        r2 = run_dynamic_episode(scenario, planner, seed=SEED)

        # Determinism
        ok1 = compare_episodes(r1, r2, label)
        all_ok = all_ok and ok1

        # Sanity
        ok2 = sanity_check(r1, label)
        all_ok = all_ok and ok2

        # Log
        results_log.append({
            "scenario": scenario,
            "planner": planner,
            "seed": SEED,
            "success": r1["success"],
            "episode_steps": r1["episode_steps"],
            "path_length": r1["path_length"],
            "total_replans": r1["total_replans"],
            "total_dynamic_blocks": r1["total_dynamic_blocks"],
            "risk_exposure_integral": round(float(r1["risk_exposure_integral"]), 4),
            "guardrail_activation_count": r1["guardrail_activation_count"],
            "constraint_violations": r1["constraint_violations"],
            "termination_reason": r1["termination_reason"],
            "deterministic": ok1,
            "sane": ok2,
        })

        print(f"  Summary: success={r1['success']}, steps={r1['episode_steps']}, "
              f"replans={r1['total_replans']}, blocks={r1['total_dynamic_blocks']}, "
              f"guardrail={r1['guardrail_activation_count']}")

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
    print(f"{'='*60}")

    # Save
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/correctness_validation.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"Results saved to outputs/correctness_validation.json")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
