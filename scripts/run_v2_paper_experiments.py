#!/usr/bin/env python3
"""Paper experiment runner — 6 planners x 9 scenarios x 30 seeds = 1,620 episodes.

Saves per-episode results to outputs/v2/paper_results/all_episodes.csv.
Episodes that crash are logged and skipped (never stops the whole run).
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from uavbench2.benchmark.runner import run_episode
from uavbench2.planners import PLANNERS
from uavbench2.scenarios.loader import load_scenario
from uavbench2.scenarios.registry import list_scenarios

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEEDS = 30
OUTPUT_DIR = "outputs/v2/paper_results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_episodes.csv")
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


class _FallbackBar:
    """Minimal progress bar when tqdm is not available."""

    def __init__(self, total: int, desc: str = "") -> None:
        self._total = total
        self._done = 0
        self._desc = desc
        self._last_pct = -1

    def update(self, n: int = 1) -> None:
        self._done += n
        pct = int(100 * self._done / self._total)
        if pct != self._last_pct and pct % 5 == 0:
            self._last_pct = pct
            print(f"\r{self._desc}: {self._done}/{self._total} ({pct}%)", end="", flush=True)

    def close(self) -> None:
        print()


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


def main() -> None:
    scenarios = list_scenarios()
    planners = sorted(PLANNERS.keys())
    total = len(scenarios) * len(planners) * N_SEEDS

    print(f"UAVBench v2 Paper Experiments")
    print(f"  Scenarios:  {len(scenarios)}")
    print(f"  Planners:   {len(planners)} — {', '.join(planners)}")
    print(f"  Seeds:      {N_SEEDS}")
    print(f"  Total:      {total} episodes")
    print(f"  Output:     {OUTPUT_FILE}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-load configs for metadata
    configs = {sid: load_scenario(sid) for sid in scenarios}

    errors: list[dict] = []
    completed = 0

    pbar = tqdm(total=total, desc="Episodes") if HAS_TQDM else _FallbackBar(total, "Episodes")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for scenario_id in scenarios:
            config = configs[scenario_id]
            for planner_id in planners:
                for seed in range(N_SEEDS):
                    try:
                        t0 = time.perf_counter()
                        result = run_episode(scenario_id, planner_id, seed)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0

                        row = _extract_row(
                            scenario_id, planner_id, seed,
                            result, elapsed_ms, config,
                        )
                        writer.writerow(row)
                        f.flush()
                        completed += 1

                    except Exception:
                        tb = traceback.format_exc()
                        errors.append({
                            "scenario_id": scenario_id,
                            "planner_id": planner_id,
                            "seed": seed,
                            "traceback": tb,
                        })

                    pbar.update(1)

    pbar.close()

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
    print(f"Results:   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
