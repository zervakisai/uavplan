#!/usr/bin/env python3
"""Profile baseline env.step() to identify bottlenecks."""
import json
import os
import sys
import time

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "src")

import numpy as np
from uavbench.scenarios.loader import load_scenario
from uavbench.cli.benchmark import scenario_path, _apply_protocol_variant, PLANNERS
from uavbench.envs.urban import UrbanEnv

SCENARIO = "gov_civil_protection_hard"
PLANNER = "periodic_replan"
SEED = 42
HORIZON = 200  # enough to capture representative step costs

cfg = load_scenario(scenario_path(SCENARIO))
cfg = _apply_protocol_variant(cfg, "default")
env = UrbanEnv(cfg)
env.reset(seed=SEED)

heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()
planner = PLANNERS[PLANNER](heightmap, no_fly)
plan_result = planner.plan(start_xy, goal_xy)

# Helper to expand path to actions
def waypoint_action(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dy < 0: return 0
    if dy > 0: return 1
    if dx < 0: return 2
    if dx > 0: return 3
    return 0

path = list(plan_result.path)
if not path:
    print("Plan failed!")
    sys.exit(1)

# Monkey-patch env to time each component
timings = {
    "step_total": [],
    "fire_step": [],
    "traffic_step": [],
    "nfz_step": [],
    "moving_target_step": [],
    "intruder_step": [],
    "population_step": [],
    "adversarial_step": [],
    "interaction_update": [],
    "risk_map_compute": [],
    "guardrail": [],
    "build_runtime_mask": [],
    "get_dynamic_state": [],
    "observation": [],
    "collision_checks": [],
    "bfs_reachable": [],
}

# Save original methods
_orig_fire_step = env._fire_model.step if env._fire_model else None
_orig_traffic_step = env._traffic_model.step if env._traffic_model else None
_orig_nfz_step = env._dynamic_nfz.step if env._dynamic_nfz else None
_orig_pop_step = env._population_model.step
_orig_interaction_update = env._interaction_engine.update
_orig_is_reachable = env._is_reachable
_orig_build_runtime = env._build_runtime_blocking_mask
_orig_get_dynamic = env.get_dynamic_state
_orig_build_obs = env._build_observation

_bfs_count = [0]

if env._fire_model:
    orig = env._fire_model.step
    def timed_fire(*a, **kw):
        t = time.perf_counter()
        r = orig(*a, **kw)
        timings["fire_step"].append(time.perf_counter() - t)
        return r
    env._fire_model.step = timed_fire

if env._traffic_model:
    orig_t = env._traffic_model.step
    def timed_traffic(*a, **kw):
        t = time.perf_counter()
        r = orig_t(*a, **kw)
        timings["traffic_step"].append(time.perf_counter() - t)
        return r
    env._traffic_model.step = timed_traffic

if env._dynamic_nfz:
    orig_n = env._dynamic_nfz.step
    def timed_nfz(*a, **kw):
        t = time.perf_counter()
        r = orig_n(*a, **kw)
        timings["nfz_step"].append(time.perf_counter() - t)
        return r
    env._dynamic_nfz.step = timed_nfz

orig_p = env._population_model.step
def timed_pop(*a, **kw):
    t = time.perf_counter()
    r = orig_p(*a, **kw)
    timings["population_step"].append(time.perf_counter() - t)
    return r
env._population_model.step = timed_pop

orig_ie = env._interaction_engine.update
def timed_ie(*a, **kw):
    t = time.perf_counter()
    r = orig_ie(*a, **kw)
    timings["interaction_update"].append(time.perf_counter() - t)
    return r
env._interaction_engine.update = timed_ie

orig_reach = env._is_reachable
def timed_reach(*a, **kw):
    t = time.perf_counter()
    r = orig_reach(*a, **kw)
    timings["bfs_reachable"].append(time.perf_counter() - t)
    _bfs_count[0] += 1
    return r
env._is_reachable = timed_reach

orig_rm = env._build_runtime_blocking_mask
def timed_rm(*a, **kw):
    t = time.perf_counter()
    r = orig_rm(*a, **kw)
    timings["build_runtime_mask"].append(time.perf_counter() - t)
    return r
env._build_runtime_blocking_mask = timed_rm

orig_gds = env.get_dynamic_state
def timed_gds(*a, **kw):
    t = time.perf_counter()
    r = orig_gds(*a, **kw)
    timings["get_dynamic_state"].append(time.perf_counter() - t)
    return r
env.get_dynamic_state = timed_gds

# Run episode
path_idx = 0
total_steps = 0
step_times = []

print(f"Profiling {SCENARIO}/{PLANNER} seed={SEED} horizon={HORIZON}")
print(f"Path length: {len(path)}")

is_adaptive = hasattr(planner, "should_replan")

for step in range(min(HORIZON, len(path) - 1)):
    if path_idx >= len(path) - 1:
        break

    action = waypoint_action(path[path_idx], path[path_idx + 1])

    t0 = time.perf_counter()
    obs, reward, terminated, truncated, info = env.step(action)
    step_time = time.perf_counter() - t0
    step_times.append(step_time)

    current_pos = tuple(int(v) for v in env._agent_pos)
    current_xy = (current_pos[0], current_pos[1])
    if current_xy == path[path_idx + 1]:
        path_idx += 1

    # Call get_dynamic_state as the harness does
    t_gds = time.perf_counter()
    dyn = env.get_dynamic_state()
    timings["get_dynamic_state"].append(time.perf_counter() - t_gds)

    # Adaptive replan check
    if is_adaptive and step % 6 == 0:
        fire_mask = dyn.get("fire_mask")
        traffic_pos = dyn.get("traffic_positions")
        should, reason = planner.should_replan(
            current_pos, fire_mask, traffic_pos,
            smoke_mask=dyn.get("smoke_mask"),
        )
        if should:
            new_path = planner.replan(
                current_pos, goal_xy, fire_mask, traffic_pos,
                reason=reason,
                smoke_mask=dyn.get("smoke_mask"),
            )
            if new_path and len(new_path) > 1:
                path = new_path
                path_idx = 0

    total_steps += 1
    if terminated or truncated:
        break

print(f"\nCompleted {total_steps} steps")
print(f"Total wall time: {sum(step_times):.3f}s")
print(f"Avg step time: {np.mean(step_times)*1000:.2f}ms")
print(f"BFS reachability calls: {_bfs_count[0]}")
print()

# Summarize
profile = {}
for key, vals in timings.items():
    if vals:
        profile[key] = {
            "count": len(vals),
            "total_ms": round(sum(vals) * 1000, 3),
            "mean_ms": round(np.mean(vals) * 1000, 3),
            "max_ms": round(max(vals) * 1000, 3),
            "pct_of_total": round(100 * sum(vals) / max(sum(step_times), 1e-9), 1),
        }
        print(f"  {key:30s}: {profile[key]['total_ms']:8.1f}ms total  "
              f"({profile[key]['mean_ms']:.2f}ms avg × {profile[key]['count']} calls)  "
              f"{profile[key]['pct_of_total']:5.1f}%")

profile["_summary"] = {
    "scenario": SCENARIO,
    "planner": PLANNER,
    "seed": SEED,
    "horizon": HORIZON,
    "total_steps": total_steps,
    "total_wall_ms": round(sum(step_times) * 1000, 3),
    "avg_step_ms": round(np.mean(step_times) * 1000, 3),
    "bfs_calls": _bfs_count[0],
    "step_times_ms": [round(t * 1000, 3) for t in step_times],
}

out_path = "outputs/performance_profile_before.json"
os.makedirs("outputs", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(profile, f, indent=2)
print(f"\nProfile saved to {out_path}")
