#!/usr/bin/env python3
"""Extract critical moment timestamps from UAVBench episodes.

For each planner on the Penteli pharma_delivery scenario (seed=42):
- Records per-step telemetry: position, distance to goal, fire area,
  replan events, blocking mask changes, corridor status.
- Identifies critical events: first corridor closure, first replan,
  path divergence between planners, success/failure step.

Outputs JSON to outputs/critical_moments.json for use in annotated figures.

Usage:
    python scripts/extract_critical_moments.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.labels import PLANNER_ORDER


SCENARIO_ID = "osm_penteli_pharma_delivery_medium"
SEED = 42
OUTPUT_PATH = "outputs/critical_moments.json"


def extract_planner_telemetry(scenario_id: str, planner_id: str, seed: int) -> dict:
    """Run one episode and collect per-step telemetry via frame_callback."""
    config = load_scenario(scenario_id)

    telemetry: list[dict] = []
    prev_mask_hash: int | None = None

    def frame_callback(heightmap, state, dyn_state, cfg):
        nonlocal prev_mask_hash

        step = state.get("step_idx", 0)
        agent_xy = state.get("agent_xy", (0, 0))
        goal_xy = state.get("goal_xy", (heightmap.shape[1] - 1, heightmap.shape[0] - 1))

        # Distance to goal (Manhattan)
        dist = abs(agent_xy[0] - goal_xy[0]) + abs(agent_xy[1] - goal_xy[1])

        # Fire area (number of burning cells)
        fire_mask = dyn_state.get("fire_mask")
        fire_area = int(np.sum(fire_mask)) if fire_mask is not None else 0

        # Blocking mask change detection
        blocking = dyn_state.get("blocking_mask")
        mask_hash = hash(blocking.tobytes()) if blocking is not None else 0
        mask_changed = (prev_mask_hash is not None and mask_hash != prev_mask_hash)
        prev_mask_hash = mask_hash

        # Replan info from state
        replan_triggered = state.get("replan_triggered", False)
        replan_reason = state.get("replan_reason", "")

        # Plan length
        plan_len = state.get("plan_len", 0)

        telemetry.append({
            "step": step,
            "agent_xy": list(agent_xy),
            "goal_dist": dist,
            "fire_area": fire_area,
            "mask_changed": mask_changed,
            "replan_triggered": replan_triggered,
            "replan_reason": replan_reason,
            "plan_len": plan_len,
        })

    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)
    metrics = result.metrics

    return {
        "planner_id": planner_id,
        "telemetry": telemetry,
        "metrics": {
            "success": metrics.get("success", False),
            "termination_reason": str(metrics.get("termination_reason", "?")),
            "executed_steps": metrics.get("executed_steps_len", 0),
            "path_length": metrics.get("path_length", 0),
            "replans": metrics.get("replans", 0),
        },
    }


def identify_critical_events(all_data: list[dict]) -> dict:
    """Analyze telemetry across all planners to find critical moments."""
    events = {}

    for pdata in all_data:
        pid = pdata["planner_id"]
        tel = pdata["telemetry"]
        if not tel:
            continue

        # First mask change (corridor closure)
        first_mask_change = None
        for t in tel:
            if t["mask_changed"]:
                first_mask_change = t["step"]
                break

        # First replan
        first_replan = None
        for t in tel:
            if t["replan_triggered"]:
                first_replan = t["step"]
                break

        # Peak fire area and its step
        peak_fire = max(tel, key=lambda t: t["fire_area"])

        # Success/termination step
        term_step = tel[-1]["step"] if tel else 0

        # Distance progression: find when distance starts increasing (stuck)
        min_dist = float("inf")
        stuck_step = None
        for t in tel:
            if t["goal_dist"] < min_dist:
                min_dist = t["goal_dist"]
            elif t["goal_dist"] > min_dist + 5 and stuck_step is None:
                stuck_step = t["step"]

        events[pid] = {
            "first_mask_change": first_mask_change,
            "first_replan": first_replan,
            "peak_fire_step": peak_fire["step"],
            "peak_fire_area": peak_fire["fire_area"],
            "termination_step": term_step,
            "success": pdata["metrics"]["success"],
            "stuck_step": stuck_step,
        }

    # Cross-planner: path divergence step
    # Compare adaptive planners against A* trajectory
    astar_data = next((d for d in all_data if d["planner_id"] == "astar"), None)
    if astar_data and astar_data["telemetry"]:
        astar_positions = {t["step"]: tuple(t["agent_xy"]) for t in astar_data["telemetry"]}
        for pdata in all_data:
            if pdata["planner_id"] == "astar":
                continue
            pid = pdata["planner_id"]
            diverge_step = None
            for t in pdata["telemetry"]:
                s = t["step"]
                if s in astar_positions and tuple(t["agent_xy"]) != astar_positions[s]:
                    diverge_step = s
                    break
            events[pid]["diverge_from_astar"] = diverge_step

    return events


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Extracting critical moments: {SCENARIO_ID}, seed={SEED}")
    print(f"Planners: {PLANNER_ORDER}\n")

    all_data = []
    for pid in PLANNER_ORDER:
        print(f"  Running {pid}...", end=" ", flush=True)
        t0 = time.perf_counter()
        data = extract_planner_telemetry(SCENARIO_ID, pid, SEED)
        elapsed = time.perf_counter() - t0
        n_steps = len(data["telemetry"])
        print(f"done ({n_steps} steps, {elapsed:.1f}s)")
        all_data.append(data)

    print("\nIdentifying critical events...")
    events = identify_critical_events(all_data)

    # Build output
    output = {
        "scenario_id": SCENARIO_ID,
        "seed": SEED,
        "planners": {d["planner_id"]: d["metrics"] for d in all_data},
        "critical_events": events,
        "telemetry": {d["planner_id"]: d["telemetry"] for d in all_data},
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"\nCritical events summary:")
    for pid, ev in events.items():
        print(f"  {pid}:")
        for k, v in ev.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
