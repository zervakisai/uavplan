#!/usr/bin/env python3
"""Unified-ρ factorial sweep (reviewer revision, Comment 1).

Isolates the risk coefficient ρ from planner architecture by evaluating each
risk-consuming planner across a RANGE of ρ values, holding the algorithm and
replan trigger fixed. Every planner applies the same cost-inflation law
w(x)=1+ρ·R(x) (edge cost for the graph-search planners, repulsive gain for APF),
so ρ is comparable across families.

Two passes, both written to one CSV (column `risk_rho`):
  1. Baseline  — risk_rho=None (each planner at its historical per-family
     coefficient α/β/γ/δ, A*=0). Reproduces the paper's reported configuration
     as the factorial "diagonal", under the corrected fire-coupled scoring.
  2. Factorial — the 4 risk-consuming planners × ρ∈{0,1,2,5,10}.
     A* is EXCLUDED from the ρ>0 sweep by default: a risk-aware static A* plans
     off the reference corridor and dodges FC-1 corridor interdictions
     (BUG-1/Theta*-style bypass artifact), so it stays the fixed ρ=0 anchor.
     Pass --include-astar to add it as a supplementary (bypass-caveated) arm.

Each row also carries `task_events_json` (per-task step_idx/weight/d_fire) so the
Comment-2 mission-model sensitivity (κ, decay horizon, λ scale) can be evaluated
post-hoc by re-scoring, with no re-simulation.

Determinism: ρ is deterministic and the RNG root is seeded only by the episode
seed, so every (scenario, planner, ρ, seed) is bit-identically reproducible and
the baseline diagonal reproduces the historical trajectories exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

OUTPUT_DIR = "outputs/rho_sweep"
ERROR_LOG = os.path.join(OUTPUT_DIR, "errors.log")
MAX_WORKERS = 6

SWEEP_PLANNERS = ["periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
BASELINE_PLANNERS = ["astar", "periodic_replan", "aggressive_replan",
                     "incremental_astar", "apf"]
RHO_VALUES = [0.0, 1.0, 2.0, 5.0, 10.0]

COLUMNS = [
    "scenario_id", "planner_id", "risk_rho", "seed",
    "success", "termination_reason", "objective_completed",
    "executed_steps", "path_length", "planned_waypoints_len", "replans",
    "replan_storm_ratio", "collision_count", "nfz_violations",
    "infeasible", "feasible_after_guardrail",
    "mission_type", "domain", "difficulty", "track",
    "mission_score", "tasks_completed", "tasks_total",
    "task_events_json", "computation_time_ms",
]


def _extract_row(scenario_id, planner_id, rho_label, seed, result, elapsed_ms, config):
    m = result.metrics
    task_events = [
        {"step_idx": e.get("step_idx"), "weight": e.get("weight"),
         "d_fire": round(float(e.get("d_fire", 999.0)), 3)}
        for e in result.events if e.get("type") == "task_completed"
    ]
    tr = m.get("termination_reason", "unknown")
    return {
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "risk_rho": rho_label,
        "seed": seed,
        "success": m.get("success", False),
        "termination_reason": tr,
        "objective_completed": m.get("objective_completed", False),
        "executed_steps": m.get("executed_steps_len", 0),
        "path_length": m.get("path_length", 0),
        "planned_waypoints_len": m.get("planned_waypoints_len", 0),
        "replans": m.get("replans", 0),
        "replan_storm_ratio": round(m.get("replan_storm_ratio", 0.0), 4),
        "collision_count": m.get("collision_count", 0),
        "nfz_violations": m.get("nfz_violations", 0),
        "infeasible": tr == "infeasible",
        "feasible_after_guardrail": m.get("feasible_after_guardrail", True),
        "mission_type": config.mission_type.value,
        "domain": config.domain.value,
        "difficulty": config.difficulty.value,
        "track": config.paper_track,
        "mission_score": round(m.get("mission_score", 0.0), 6),
        "tasks_completed": m.get("tasks_completed", 0),
        "tasks_total": m.get("tasks_total", 0),
        "task_events_json": json.dumps(task_events),
        "computation_time_ms": round(elapsed_ms, 2),
    }


def _run_block(args: tuple) -> list[dict]:
    """Worker: one (scenario, planner, rho) block over n_seeds. Module-level for spawn."""
    scenario_id, planner_id, rho, n_seeds = args
    import time as _t
    import traceback
    from dataclasses import replace

    from flare.benchmark.runner import run_episode
    from flare.scenarios.loader import load_scenario

    base = load_scenario(scenario_id)
    # rho is None → baseline (historical per-family coeff); else unified ρ.
    cfg = base if rho is None else replace(base, risk_rho=float(rho))
    rho_label = "none" if rho is None else float(rho)

    out = []
    for seed in range(n_seeds):
        try:
            t0 = _t.perf_counter()
            result = run_episode(scenario_id, planner_id, seed, config_override=cfg)
            elapsed = (_t.perf_counter() - t0) * 1000.0
            out.append({"status": "ok",
                        "row": _extract_row(scenario_id, planner_id, rho_label,
                                            seed, result, elapsed, cfg)})
        except Exception:
            out.append({"status": "error", "scenario_id": scenario_id,
                        "planner_id": planner_id, "rho": rho_label, "seed": seed,
                        "traceback": traceback.format_exc()})
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    from flare.scenarios.registry import list_scenarios

    p = argparse.ArgumentParser(description="FLARE unified-ρ factorial sweep.")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--scenarios", type=str, default=None,
                   help="comma-separated scenario ids (default: all 3)")
    p.add_argument("--rhos", type=str, default=None,
                   help="comma-separated ρ values (default: 0,1,2,5,10)")
    p.add_argument("--include-astar", action="store_true",
                   help="also sweep A* over ρ>0 (supplementary; bypass caveat)")
    p.add_argument("--sweep-planners", type=str, default=None,
                   help="comma-separated planner ids to sweep "
                        "(default: the 4 original risk-consuming planners)")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the risk_rho=None baseline pass")
    p.add_argument("--output", type=str,
                   default=os.path.join(OUTPUT_DIR, "rho_sweep.csv"))
    args = p.parse_args()

    scenarios = ([s.strip() for s in args.scenarios.split(",")]
                 if args.scenarios else list_scenarios())
    rhos = ([float(x) for x in args.rhos.split(",")]
            if args.rhos else list(RHO_VALUES))
    sweep_planners = ([s.strip() for s in args.sweep_planners.split(",")]
                      if args.sweep_planners else list(SWEEP_PLANNERS))
    if args.include_astar and "astar" not in sweep_planners:
        sweep_planners = ["astar"] + sweep_planners
    n_seeds = args.seeds

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Build work units.
    work = []
    if not args.no_baseline:
        for sid in scenarios:
            for pid in BASELINE_PLANNERS:
                work.append((sid, pid, None, n_seeds))   # None → historical diagonal
    for sid in scenarios:
        for pid in sweep_planners:
            for rho in rhos:
                work.append((sid, pid, rho, n_seeds))

    total = len(work) * n_seeds
    print(f"FLARE unified-ρ sweep")
    print(f"  Scenarios:      {len(scenarios)}")
    print(f"  Baseline:       {'off' if args.no_baseline else BASELINE_PLANNERS}")
    print(f"  Sweep planners: {sweep_planners}")
    print(f"  ρ values:       {rhos}")
    print(f"  Seeds:          {n_seeds}")
    print(f"  Work units:     {len(work)}  → {total} episodes")
    print(f"  Output:         {args.output}\n")

    attempted = completed = 0
    errors: list[dict] = []
    wall = time.perf_counter()

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_run_block, wu): wu for wu in work}
            for fut in as_completed(futs):
                wu = futs[fut]
                try:
                    block = fut.result()
                except Exception as e:
                    errors.append({"unit": str(wu), "traceback": str(e)})
                    attempted += n_seeds
                    continue
                for item in block:
                    attempted += 1
                    if item["status"] == "ok":
                        w.writerow(item["row"]); f.flush(); completed += 1
                    else:
                        errors.append(item)
                el = time.perf_counter() - wall
                eta = (el / attempted * (total - attempted)) if attempted else 0
                sid, pid, rho, _ = wu
                print(f"\r[{attempted}/{total}] ({100*attempted/total:.0f}%) "
                      f"| {el/60:.1f}m elapsed | ETA {eta/60:.0f}m "
                      f"| last {pid}/ρ={('none' if rho is None else rho)}/"
                      f"{sid.replace('osm_','')}", end="", flush=True)
    print()

    if errors:
        with open(ERROR_LOG, "w") as ef:
            for err in errors:
                ef.write(json.dumps({k: v for k, v in err.items()
                                     if k != "traceback"}) + "\n")
                ef.write(err.get("traceback", "") + "\n")

    print(f"\nCompleted: {completed}/{total}  Errors: {len(errors)}")
    print(f"Results:   {args.output}")

    import datetime, platform
    profile = {
        "total_wall_clock_seconds": round(time.perf_counter() - wall, 2),
        "total_episodes": total, "completed": completed, "errors": len(errors),
        "scenarios": scenarios, "sweep_planners": sweep_planners,
        "baseline_planners": [] if args.no_baseline else BASELINE_PLANNERS,
        "rhos": rhos, "seeds": n_seeds, "include_astar": args.include_astar,
        "python_version": sys.version, "platform": platform.platform(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, "runtime_profile.json"), "w") as pf:
        json.dump(profile, pf, indent=2)


if __name__ == "__main__":
    main()
