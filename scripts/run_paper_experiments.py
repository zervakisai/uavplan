#!/usr/bin/env python3
"""Paper experiment runner — 6 planners x 9 scenarios x 30 seeds = 1,620 episodes.

Saves per-episode results to outputs/paper_results/all_episodes.csv.
Episodes that crash are logged and skipped (never stops the whole run).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

from uavbench.benchmark.runner import run_episode
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.registry import list_scenarios

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/paper_results"
ERROR_LOG = os.path.join(OUTPUT_DIR, "errors.log")

COLUMNS = [
    "scenario_id",
    "planner_id",
    "seed",
    "success",
    "path_length",
    "executed_steps",
    "replans",
    "service_time_steps",
    "termination_reason",
    "objective_completed",
    "computation_time_ms",
    "guardrail_activated",
    "guardrail_depth",
    "infeasible",
    "mission_type",
    "domain",
    "difficulty",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


def _progress_line(
    done: int,
    total: int,
    elapsed: float,
    last_label: str,
    last_sec: float,
) -> str:
    """Build a single-line progress string with bar, ETA, and last episode."""
    pct = done / total if total else 1
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    parts = [f"[{bar}] {done}/{total} ({100 * pct:.0f}%)"]
    parts.append(f"Elapsed: {_fmt_duration(elapsed)}")

    if done > 0:
        avg = elapsed / done
        remaining = avg * (total - done)
        parts.append(f"ETA: {_fmt_duration(remaining)}")

    if last_label:
        parts.append(f"Last: {last_label} ({last_sec:.1f}s)")

    return " | ".join(parts)


def _extract_row(
    scenario_id: str,
    planner_id: str,
    seed: int,
    result: "EpisodeResult",
    elapsed_ms: float,
    config: "ScenarioConfig",
) -> dict:
    """Extract a CSV row dict from an EpisodeResult."""
    m = result.metrics

    # Extract mission briefing from events
    briefing = next(
        (e for e in result.events if e.get("type") == "mission_briefing"),
        {},
    )

    # Detect guardrail activation from events
    guardrail_events = [
        e for e in result.events if "guardrail" in str(e.get("type", ""))
    ]
    guardrail_depth = 0
    for ge in guardrail_events:
        d = ge.get("depth", 0)
        if d > guardrail_depth:
            guardrail_depth = d

    tr = m.get("termination_reason", "unknown")

    return {
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "seed": seed,
        "success": m.get("success", False),
        "path_length": m.get("path_length", 0),
        "executed_steps": m.get("executed_steps_len", 0),
        "replans": m.get("replans", 0),
        "service_time_steps": briefing.get("service_time_steps", 0),
        "termination_reason": tr,
        "objective_completed": m.get("objective_completed", False),
        "computation_time_ms": round(elapsed_ms, 2),
        "guardrail_activated": len(guardrail_events) > 0,
        "guardrail_depth": guardrail_depth,
        "infeasible": tr == "infeasible",
        "mission_type": config.mission_type.value,
        "domain": config.domain.value,
        "difficulty": config.difficulty.value,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run UAVBench v2 paper experiments.")
    p.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario IDs (default: all 9)",
    )
    p.add_argument(
        "--planners", type=str, default=None,
        help="Comma-separated planner IDs (default: all 6)",
    )
    p.add_argument(
        "--seeds", type=int, default=30,
        help="Number of seeds per (scenario, planner) pair (default: 30)",
    )
    p.add_argument(
        "--output", type=str, default=os.path.join(OUTPUT_DIR, "all_episodes.csv"),
        help="Output CSV path",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    scenarios = (
        [s.strip() for s in args.scenarios.split(",")]
        if args.scenarios else list_scenarios()
    )
    planners = (
        [p.strip() for p in args.planners.split(",")]
        if args.planners else sorted(PLANNERS.keys())
    )
    n_seeds = args.seeds
    output_file = args.output
    total = len(scenarios) * len(planners) * n_seeds

    print(f"UAVBench v2 Paper Experiments")
    print(f"  Scenarios:  {len(scenarios)}")
    print(f"  Planners:   {len(planners)} — {', '.join(planners)}")
    print(f"  Seeds:      {n_seeds}")
    print(f"  Total:      {total} episodes")
    print(f"  Output:     {output_file}")
    print()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Pre-load configs for metadata
    configs = {sid: load_scenario(sid) for sid in scenarios}

    errors: list[dict] = []
    completed = 0
    attempted = 0
    last_label = ""
    last_sec = 0.0
    wall_start = time.perf_counter()

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for scenario_id in scenarios:
            config = configs[scenario_id]
            for planner_id in planners:
                for seed in range(n_seeds):
                    try:
                        t0 = time.perf_counter()
                        result = run_episode(scenario_id, planner_id, seed)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        last_sec = elapsed_ms / 1000.0

                        row = _extract_row(
                            scenario_id, planner_id, seed,
                            result, elapsed_ms, config,
                        )
                        writer.writerow(row)
                        f.flush()
                        completed += 1

                    except Exception:
                        tb = traceback.format_exc()
                        last_sec = time.perf_counter() - t0
                        errors.append({
                            "scenario_id": scenario_id,
                            "planner_id": planner_id,
                            "seed": seed,
                            "traceback": tb,
                        })

                    attempted += 1
                    scn_short = scenario_id.replace("gov_", "")
                    last_label = f"{planner_id}/{scn_short}/seed={seed}"
                    wall_elapsed = time.perf_counter() - wall_start
                    line = _progress_line(
                        attempted, total, wall_elapsed, last_label, last_sec,
                    )
                    print(f"\r{line}", end="", flush=True)

    print()  # newline after progress bar

    # Error log
    if errors:
        with open(ERROR_LOG, "w") as ef:
            for err in errors:
                ef.write(
                    f"--- {err['scenario_id']} / {err['planner_id']} / seed={err['seed']} ---\n"
                )
                ef.write(err["traceback"])
                ef.write("\n")

    print()
    print(f"Completed: {completed}/{total}")
    print(f"Errors:    {len(errors)}")
    if errors:
        print(f"Error log: {ERROR_LOG}")
    print(f"Results:   {output_file}")


if __name__ == "__main__":
    main()
