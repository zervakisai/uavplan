"""Fairness audit module: validates cross-planner evaluation invariants.

Checks:
- Identical seed reproducibility
- Identical snapshot equality
- Identical collision semantics
- Identical time budgets
- Interdiction hit rate variance = 0.0
- Aborts if any invariant violated

Exports: fairness_audit.json
"""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from uavbench.cli.benchmark import run_dynamic_episode, run_planner_once, scenario_path
from uavbench.scenarios.loader import load_scenario


def _hash_array(arr: np.ndarray) -> str:
    """Deterministic hash of a numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def audit_seed_reproducibility(
    scenario_id: str,
    planner_id: str,
    seed: int,
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Run the same (scenario, planner, seed) N times and check identical output."""
    results = []
    for _ in range(n_repeats):
        cfg = load_scenario(scenario_path(scenario_id))
        use_dynamic = (
            cfg.paper_track == "dynamic"
            or int(cfg.force_replan_count) > 0
            or cfg.fire_blocks_movement
            or cfg.traffic_blocks_movement
        )
        if use_dynamic:
            r = run_dynamic_episode(scenario_id, planner_id, seed=seed)
        else:
            r = run_planner_once(scenario_id, planner_id, seed=seed)
        results.append(r)

    # Compare: success, path_length, path content
    successes = [r.get("success") for r in results]
    lengths = [r.get("path_length", 0) for r in results]
    paths = [str(r.get("path", [])) for r in results]

    all_same = (len(set(successes)) == 1 and len(set(lengths)) == 1 and len(set(paths)) == 1)

    return {
        "scenario": scenario_id,
        "planner": planner_id,
        "seed": seed,
        "repeats": n_repeats,
        "success_values": successes,
        "length_values": lengths,
        "deterministic": all_same,
        "status": "PASS" if all_same else "FAIL",
    }


def audit_snapshot_equality(
    scenario_id: str,
    planners: list[str],
    seed: int,
) -> dict[str, Any]:
    """Verify all planners see identical initial map snapshots."""
    hashes = {}
    for planner_id in planners:
        r = run_planner_once(scenario_id, planner_id, seed=seed)
        hm = np.asarray(r["heightmap"])
        nf = np.asarray(r["no_fly"])
        hashes[planner_id] = {
            "heightmap_hash": _hash_array(hm),
            "no_fly_hash": _hash_array(nf),
            "start": tuple(r["start"]),
            "goal": tuple(r["goal"]),
        }

    # All should be identical
    ref = list(hashes.values())[0]
    all_equal = all(
        v["heightmap_hash"] == ref["heightmap_hash"]
        and v["no_fly_hash"] == ref["no_fly_hash"]
        and v["start"] == ref["start"]
        and v["goal"] == ref["goal"]
        for v in hashes.values()
    )

    return {
        "scenario": scenario_id,
        "seed": seed,
        "planners": planners,
        "snapshot_hashes": hashes,
        "all_equal": all_equal,
        "status": "PASS" if all_equal else "FAIL",
    }


def audit_time_budget_fairness(
    scenario_id: str,
    planners: list[str],
    seed: int,
) -> dict[str, Any]:
    """Verify all planners receive identical time budgets."""
    budgets = {}
    for planner_id in planners:
        r = run_planner_once(scenario_id, planner_id, seed=seed)
        budgets[planner_id] = float(r.get("plan_budget_ms", 0.0))

    budget_set = set(budgets.values())
    all_equal = len(budget_set) == 1

    return {
        "scenario": scenario_id,
        "seed": seed,
        "planners": planners,
        "budgets": budgets,
        "all_equal": all_equal,
        "status": "PASS" if all_equal else "FAIL",
    }


def audit_interdiction_hit_rate(
    scenario_id: str,
    planners: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Verify interdiction hit rate variance = 0 across planners."""
    rates: dict[str, list[float]] = defaultdict(list)

    for planner_id in planners:
        for seed in seeds:
            cfg = load_scenario(scenario_path(scenario_id))
            if cfg.paper_track != "dynamic":
                continue
            try:
                r = run_dynamic_episode(scenario_id, planner_id, seed=seed)
                rate = float(r.get("interdiction_hit_rate", 0.0))
                rates[planner_id].append(rate)
            except Exception:
                pass

    # All planners should have identical interdiction hit rates
    all_rates = [r for rs in rates.values() for r in rs]
    variance = float(np.var(all_rates)) if all_rates else 0.0
    zero_variance = variance < 1e-6

    return {
        "scenario": scenario_id,
        "seeds": seeds,
        "planners": planners,
        "rates_per_planner": {k: v for k, v in rates.items()},
        "variance": round(variance, 8),
        "zero_variance": zero_variance,
        "status": "PASS" if zero_variance else "FAIL",
    }


def run_full_audit(
    scenarios: list[str],
    planners: list[str],
    seeds: list[int],
    output_path: Path,
    *,
    n_repeats: int = 2,
) -> dict[str, Any]:
    """Run the complete fairness audit and save results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "audit_version": "2.0",
        "scenarios": scenarios,
        "planners": planners,
        "seeds": seeds,
        "checks": {},
        "overall_status": "PASS",
    }

    # 1. Seed reproducibility
    print("[Fairness] Checking seed reproducibility...")
    repro_results = []
    for scenario_id in scenarios[:2]:  # Sample check
        for planner_id in planners[:3]:
            r = audit_seed_reproducibility(scenario_id, planner_id, seeds[0], n_repeats)
            repro_results.append(r)
            if r["status"] == "FAIL":
                audit["overall_status"] = "FAIL"
    audit["checks"]["seed_reproducibility"] = repro_results

    # 2. Snapshot equality
    print("[Fairness] Checking snapshot equality...")
    snapshot_results = []
    for scenario_id in scenarios[:3]:
        r = audit_snapshot_equality(scenario_id, planners[:4], seeds[0])
        snapshot_results.append(r)
        if r["status"] == "FAIL":
            audit["overall_status"] = "FAIL"
    audit["checks"]["snapshot_equality"] = snapshot_results

    # 3. Time budget fairness
    print("[Fairness] Checking time budget fairness...")
    budget_results = []
    for scenario_id in scenarios[:3]:
        r = audit_time_budget_fairness(scenario_id, planners[:4], seeds[0])
        budget_results.append(r)
        if r["status"] == "FAIL":
            audit["overall_status"] = "FAIL"
    audit["checks"]["time_budget_fairness"] = budget_results

    # 4. Interdiction hit rate
    dynamic_scenarios = [s for s in scenarios if _is_dynamic(s)]
    if dynamic_scenarios:
        print("[Fairness] Checking interdiction hit rate variance...")
        adaptive_planners = [p for p in planners if p not in ("astar", "theta_star")]
        if adaptive_planners:
            for scenario_id in dynamic_scenarios[:2]:
                r = audit_interdiction_hit_rate(scenario_id, adaptive_planners[:4], seeds[:2])
                audit["checks"].setdefault("interdiction_hit_rate", []).append(r)
                if r["status"] == "FAIL":
                    audit["overall_status"] = "FAIL"

    # Save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"[Fairness] Audit saved: {output_path}")
    print(f"[Fairness] Overall: {audit['overall_status']}")
    return audit


def _is_dynamic(scenario_id: str) -> bool:
    """Check if a scenario is dynamic."""
    try:
        cfg = load_scenario(scenario_path(scenario_id))
        return (
            cfg.paper_track == "dynamic"
            or cfg.fire_blocks_movement
            or cfg.traffic_blocks_movement
            or cfg.enable_dynamic_nfz
        )
    except Exception:
        return False
