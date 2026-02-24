#!/usr/bin/env python3
"""Mission Realism Quantitative Audit — UAVBench.

For each of the 6 dynamic scenarios (medium+hard × 3 missions), steps the
environment with a deterministic agent (always action=0 = stay/up) and records:
  - fire_coverage_curve (fraction of map burning per step)
  - nfz_coverage_curve (fraction of map under dynamic restriction zones)
  - traffic_closure_coverage_curve
  - forced_block_active fraction of total steps
  - forced_block_cleared_by_guardrail occurrences
  - reachable_rate (BFS reachability from agent to goal per step)
  - corridor_closure events (reachability toggles false)

Also runs 2 planners (astar, periodic_replan) × 1 seed for a short episode
to measure time-to-first-collision and interdiction timing.

Usage:
    python scripts/mission_realism_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.loader import load_scenario

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "src" / "uavbench" / "scenarios" / "configs"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "mission_realism_quantitative.json"

DYNAMIC_SCENARIOS = [
    "gov_civil_protection_medium",
    "gov_civil_protection_hard",
    "gov_maritime_domain_medium",
    "gov_maritime_domain_hard",
    "gov_critical_infrastructure_medium",
    "gov_critical_infrastructure_hard",
]

EASY_SCENARIOS = [
    "gov_civil_protection_easy",
    "gov_maritime_domain_easy",
    "gov_critical_infrastructure_easy",
]

SAMPLE_STEPS = 300  # enough to see t1, t2 fire, and interdictions
SEED = 42


def bfs_reachable(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """Quick BFS on ~500×500 grid. Returns True if goal is reachable."""
    H, W = blocked.shape
    sx, sy = start
    gx, gy = goal
    if blocked[sy, sx] or blocked[gy, gx]:
        return False
    visited = np.zeros((H, W), dtype=bool)
    visited[sy, sx] = True
    queue = [(sx, sy)]
    head = 0
    while head < len(queue):
        x, y = queue[head]
        head += 1
        if x == gx and y == gy:
            return True
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx] and not blocked[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))
    return False


def audit_scenario(scenario_id: str) -> dict:
    """Run a passive audit of one dynamic scenario."""
    print(f"\n{'='*60}")
    print(f"Auditing: {scenario_id}")
    print(f"{'='*60}")

    cfg = load_scenario(CONFIGS_DIR / f"{scenario_id}.yaml")
    env = UrbanEnv(cfg)
    obs, info = env.reset(seed=SEED)

    map_size = cfg.map_size
    total_cells = map_size * map_size
    max_steps = 4 * map_size
    start_xy = env.agent_xy
    goal_xy = env.goal_xy

    # Record parameter snapshot
    params = {
        "scenario_id": scenario_id,
        "difficulty": cfg.difficulty.value,
        "mission_type": cfg.mission_type.value,
        "map_size": map_size,
        "building_density": cfg.building_density,
        "tile_name": getattr(cfg, "osm_tile_id", None) or getattr(cfg, "tile_name", "?"),
        "enable_fire": cfg.enable_fire,
        "fire_blocks_movement": cfg.fire_blocks_movement,
        "fire_ignition_points": cfg.fire_ignition_points,
        "wind_speed": cfg.wind_speed,
        "enable_traffic": cfg.enable_traffic,
        "traffic_blocks_movement": cfg.traffic_blocks_movement,
        "num_emergency_vehicles": cfg.num_emergency_vehicles,
        "enable_dynamic_nfz": cfg.enable_dynamic_nfz,
        "restrictions_max_coverage": getattr(cfg, "restrictions_max_coverage", None),
        "force_replan_count": cfg.force_replan_count,
        "event_t1": cfg.event_t1,
        "event_t2": cfg.event_t2,
        "start_xy": list(start_xy),
        "goal_xy": list(goal_xy),
        "max_episode_steps": max_steps,
    }

    # Time series
    fire_coverage = []
    nfz_coverage = []
    traffic_closure_coverage = []
    forced_block_active_series = []
    forced_block_area_series = []
    forced_block_cleared_series = []
    guardrail_depth_series = []
    reachable_series = []

    # Counters
    corridor_closures = 0
    prev_reachable = True

    # Step with action=0 (move up / stay) to observe dynamics passively
    t0 = time.time()
    steps_to_run = min(SAMPLE_STEPS, max_steps)

    for step_i in range(steps_to_run):
        obs, reward, terminated, truncated, info = env.step(0)  # action=0

        dyn = env.get_dynamic_state()

        # Fire coverage
        fire_mask = dyn.get("fire_mask")
        if fire_mask is not None:
            fc = float(np.sum(fire_mask)) / total_cells
        else:
            fc = 0.0
        fire_coverage.append(fc)

        # Dynamic NFZ coverage
        nfz_mask = dyn.get("dynamic_nfz_mask")
        if nfz_mask is not None:
            nc = float(np.sum(nfz_mask)) / total_cells
        else:
            nc = 0.0
        nfz_coverage.append(nc)

        # Traffic closure
        tc_mask = dyn.get("traffic_closure_mask")
        if tc_mask is not None:
            tc = float(np.sum(tc_mask)) / total_cells
        else:
            tc = 0.0
        traffic_closure_coverage.append(tc)

        # Forced block
        fb_active = bool(info.get("forced_block_active", False))
        fb_area = int(info.get("forced_block_area", 0))
        fb_cleared = bool(info.get("forced_block_cleared_by_guardrail", False))
        gd = int(info.get("guardrail_depth", 0))
        forced_block_active_series.append(fb_active)
        forced_block_area_series.append(fb_area)
        forced_block_cleared_series.append(fb_cleared)
        guardrail_depth_series.append(gd)

        # Reachability check (every 10 steps to save time on BFS)
        if step_i % 10 == 0:
            # Build full blocking mask
            blocking = np.zeros((map_size, map_size), dtype=bool)
            # buildings
            hmap = env._heightmap if hasattr(env, '_heightmap') else None
            if hmap is not None:
                blocking |= (hmap > 0)
            # Add dynamic blocks
            if fire_mask is not None and cfg.fire_blocks_movement:
                blocking |= fire_mask
            if tc_mask is not None and cfg.traffic_blocks_movement:
                blocking |= tc_mask
            fb_mask = dyn.get("forced_block_mask")
            if fb_mask is not None:
                blocking |= fb_mask
            if nfz_mask is not None:
                blocking |= nfz_mask

            reachable = bfs_reachable(blocking, start_xy, goal_xy)
            reachable_series.append({"step": step_i, "reachable": reachable})
            if prev_reachable and not reachable:
                corridor_closures += 1
            prev_reachable = reachable

        if terminated or truncated:
            break

    elapsed = time.time() - t0
    actual_steps = len(fire_coverage)

    # Compute summary statistics
    fire_peak = max(fire_coverage) if fire_coverage else 0.0
    fire_at_t1 = fire_coverage[cfg.event_t1 - 1] if cfg.event_t1 and cfg.event_t1 <= actual_steps else None
    fire_at_t2 = fire_coverage[cfg.event_t2 - 1] if cfg.event_t2 and cfg.event_t2 <= actual_steps else None

    nfz_peak = max(nfz_coverage) if nfz_coverage else 0.0
    traffic_peak = max(traffic_closure_coverage) if traffic_closure_coverage else 0.0

    fb_active_frac = sum(1 for x in forced_block_active_series if x) / max(actual_steps, 1)
    fb_cleared_count = sum(1 for x in forced_block_cleared_series if x)
    fb_max_area = max(forced_block_area_series) if forced_block_area_series else 0

    guardrail_activations = sum(1 for x in guardrail_depth_series if x > 0)
    guardrail_max_depth = max(guardrail_depth_series) if guardrail_depth_series else 0

    reachable_checks = len(reachable_series)
    reachable_true = sum(1 for x in reachable_series if x["reachable"])
    reachable_rate = reachable_true / max(reachable_checks, 1)

    # Subsample curves for JSON (every 5 steps)
    def subsample(series, every=5):
        return [round(series[i], 6) for i in range(0, len(series), every)]

    result = {
        "params": params,
        "audit_steps": actual_steps,
        "audit_seed": SEED,
        "audit_elapsed_s": round(elapsed, 2),
        "terminated_early": terminated if 'terminated' in dir() else False,
        "fire": {
            "peak_coverage": round(fire_peak, 6),
            "coverage_at_t1": round(fire_at_t1, 6) if fire_at_t1 is not None else None,
            "coverage_at_t2": round(fire_at_t2, 6) if fire_at_t2 is not None else None,
            "coverage_at_step_100": round(fire_coverage[99], 6) if actual_steps >= 100 else None,
            "coverage_at_step_200": round(fire_coverage[199], 6) if actual_steps >= 200 else None,
            "curve_subsampled": subsample(fire_coverage),
        },
        "nfz": {
            "peak_coverage": round(nfz_peak, 6),
            "coverage_at_step_100": round(nfz_coverage[99], 6) if actual_steps >= 100 else None,
            "coverage_at_step_200": round(nfz_coverage[199], 6) if actual_steps >= 200 else None,
            "curve_subsampled": subsample(nfz_coverage),
        },
        "traffic_closure": {
            "peak_coverage": round(traffic_peak, 6),
            "coverage_at_step_100": round(traffic_closure_coverage[99], 6) if actual_steps >= 100 else None,
            "curve_subsampled": subsample(traffic_closure_coverage),
        },
        "forced_block": {
            "active_fraction": round(fb_active_frac, 4),
            "max_area_cells": fb_max_area,
            "max_area_fraction": round(fb_max_area / total_cells, 6),
            "cleared_by_guardrail_count": fb_cleared_count,
        },
        "guardrail": {
            "activation_count": guardrail_activations,
            "activation_fraction": round(guardrail_activations / max(actual_steps, 1), 4),
            "max_depth": guardrail_max_depth,
        },
        "reachability": {
            "checks": reachable_checks,
            "reachable_rate": round(reachable_rate, 4),
            "corridor_closures": corridor_closures,
            "timeline": reachable_series,
        },
    }

    # Print summary
    print(f"  Steps run: {actual_steps} in {elapsed:.1f}s")
    print(f"  Fire peak: {fire_peak:.4f} ({fire_peak*100:.2f}%)")
    print(f"  NFZ peak:  {nfz_peak:.4f} ({nfz_peak*100:.2f}%)")
    print(f"  Traffic closure peak: {traffic_peak:.6f} ({traffic_peak*100:.4f}%)")
    print(f"  Forced block active: {fb_active_frac:.2%} of steps, max area={fb_max_area} cells")
    print(f"  Guardrail: {guardrail_activations} activations, max depth={guardrail_max_depth}")
    print(f"  Reachability: {reachable_rate:.2%} ({reachable_true}/{reachable_checks} checks)")
    print(f"  Corridor closures: {corridor_closures}")
    if fb_cleared_count > 0:
        print(f"  ** Forced block CLEARED by guardrail {fb_cleared_count} times **")

    return result


def audit_easy(scenario_id: str) -> dict:
    """Quick sanity check for easy scenarios: just verify solvability."""
    print(f"\n--- Easy scenario: {scenario_id} ---")
    cfg = load_scenario(CONFIGS_DIR / f"{scenario_id}.yaml")
    env = UrbanEnv(cfg)
    obs, info = env.reset(seed=SEED)

    start_xy = env.agent_xy
    goal_xy = env.goal_xy
    map_size = cfg.map_size

    # BFS check
    hmap = env._heightmap if hasattr(env, '_heightmap') else np.zeros((map_size, map_size))
    blocking = (hmap > 0)
    reachable = bfs_reachable(blocking, start_xy, goal_xy)
    manhattan = abs(goal_xy[0] - start_xy[0]) + abs(goal_xy[1] - start_xy[1])

    result = {
        "scenario_id": scenario_id,
        "difficulty": "easy",
        "mission_type": cfg.mission_type.value,
        "map_size": map_size,
        "building_density": cfg.building_density,
        "start_xy": list(start_xy),
        "goal_xy": list(goal_xy),
        "manhattan_distance": manhattan,
        "bfs_reachable": reachable,
        "enable_fire": cfg.enable_fire,
        "enable_traffic": cfg.enable_traffic,
        "enable_dynamic_nfz": cfg.enable_dynamic_nfz,
    }
    print(f"  Reachable: {reachable}, Manhattan: {manhattan}, Density: {cfg.building_density}")
    return result


def main():
    from datetime import datetime, timezone

    print("UAVBench Mission Realism Quantitative Audit")
    print("=" * 60)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Quantitative mission realism audit — fire/NFZ/traffic/reachability time series",
        "seed": SEED,
        "sample_steps": SAMPLE_STEPS,
        "easy_scenarios": {},
        "dynamic_scenarios": {},
    }

    # Easy scenarios (quick BFS check)
    for sid in EASY_SCENARIOS:
        try:
            results["easy_scenarios"][sid] = audit_easy(sid)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["easy_scenarios"][sid] = {"error": str(e)}

    # Dynamic scenarios (full time-series audit)
    for sid in DYNAMIC_SCENARIOS:
        try:
            results["dynamic_scenarios"][sid] = audit_scenario(sid)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results["dynamic_scenarios"][sid] = {"error": str(e)}

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
