"""Render a single episode to GIF for visual inspection.

Uses the main Renderer class for consistency with paper snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from uavbench.benchmark.runner import _path_to_action
from uavbench.envs.urban import UrbanEnvV2
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer


def render_episode(
    scenario_id: str,
    planner_id: str,
    seed: int,
    out_path: str,
    max_frames: int = 600,
    frame_skip: int = 3,
) -> dict:
    config = load_scenario(scenario_id)
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)

    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    planner_cls = PLANNERS[planner_id]
    planner = planner_cls(heightmap, no_fly, config)
    planner.set_seed(seed)

    plan_result = planner.plan(start_xy, goal_xy)
    path = plan_result.path if plan_result.success else []
    path_idx = 0
    trajectory = [start_xy]

    renderer = Renderer(config, mode="ops_full")

    frames = []
    step_idx = 0
    replan_count = 0
    terminated = False
    truncated = False

    print(f"Running {scenario_id} / {planner_id} / seed={seed} ...")

    while not terminated and not truncated and step_idx < max_frames:
        step_idx += 1

        action = _path_to_action(env.agent_xy, path, path_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.agent_xy)

        if path_idx < len(path) - 1:
            next_wp = path[path_idx + 1]
            if env.agent_xy == next_wp:
                path_idx += 1

        dyn_state = env.get_dynamic_state()
        planner.update(dyn_state)

        should, reason = planner.should_replan(
            env.agent_xy, path, dyn_state, step_idx
        )
        if should:
            plan_result_new = planner.plan(env.agent_xy, goal_xy)
            if plan_result_new.success:
                path = plan_result_new.path
                path_idx = 0
                replan_count += 1

        # Render every N frames + first/last
        if step_idx % frame_skip == 0 or step_idx <= 3 or terminated or truncated:
            remaining_path = path[path_idx:] if path else []
            state = {
                "agent_xy": env.agent_xy,
                "goal_xy": goal_xy,
                "trajectory": list(trajectory),
                "plan_path": remaining_path,
                "plan_len": len(remaining_path),
                "plan_age_steps": 0,
                "plan_reason": "",
                "scenario_id": scenario_id,
                "planner_name": planner_id,
                "mission_domain": info.get("mission_domain", ""),
                "objective_label": info.get("objective_label", ""),
                "distance_to_task": info.get("distance_to_task", 0),
                "task_progress": info.get("task_progress", ""),
                "deliverable_name": info.get("deliverable_name", ""),
                "step_idx": step_idx,
                "replans": replan_count,
            }
            frame, _meta = renderer.render_frame(heightmap, state, dyn_state)
            frames.append(frame)

        if step_idx % 100 == 0:
            print(f"  step {step_idx}... (replans={replan_count})")

    # Save GIF
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=0.12, loop=0)

    term_reason = "unknown"
    if hasattr(info.get("termination_reason", ""), "value"):
        term_reason = info["termination_reason"].value
    elif "termination_reason" in info:
        term_reason = str(info["termination_reason"])

    success = info.get("objective_completed", False)

    print(f"\nResult:")
    print(f"  Scenario:  {scenario_id}")
    print(f"  Planner:   {planner_id}")
    print(f"  Seed:      {seed}")
    print(f"  Steps:     {step_idx}")
    print(f"  Replans:   {replan_count}")
    print(f"  Outcome:   {term_reason}")
    print(f"  Objective:  {'completed' if success else 'not completed'}")
    print(f"  Frames:    {len(frames)}")
    print(f"  GIF:       {out_path}")
    return {"steps": step_idx, "replans": replan_count, "outcome": term_reason}


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "osm_penteli_fire_delivery_medium"
    planner = sys.argv[2] if len(sys.argv) > 2 else "aggressive_replan"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    out = f"outputs/episode_{scenario}_{planner}_s{seed}.gif"

    render_episode(scenario, planner, seed, out)
