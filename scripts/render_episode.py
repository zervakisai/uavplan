"""Render a single v2 episode to GIF for visual inspection.

Fast version: vectorized basemap, skip frames, lower cell resolution.
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


# Fast vectorized rendering (no per-pixel loops)
CELL_PX = 2  # 2 pixels per cell → 1000x1000 image for 500x500 map

COLOR_GROUND = np.array([230, 230, 220], dtype=np.uint8)
COLOR_BUILDING = np.array([80, 80, 80], dtype=np.uint8)
COLOR_FIRE = np.array([255, 80, 20], dtype=np.uint8)
COLOR_SMOKE = np.array([180, 180, 180], dtype=np.uint8)
COLOR_AGENT = np.array([0, 102, 255], dtype=np.uint8)
COLOR_GOAL = np.array([255, 215, 0], dtype=np.uint8)
COLOR_PATH = np.array([79, 195, 247], dtype=np.uint8)
COLOR_TRAIL = np.array([0, 180, 255], dtype=np.uint8)
COLOR_FORCED = np.array([200, 0, 200], dtype=np.uint8)
COLOR_NFZ = np.array([255, 100, 100], dtype=np.uint8)


def _fast_render(
    heightmap: np.ndarray,
    fire_mask: np.ndarray | None,
    smoke_mask: np.ndarray | None,
    forced_block_mask: np.ndarray | None,
    agent_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    trajectory: list[tuple[int, int]],
    path: list[tuple[int, int]],
    step_idx: int,
    planner_id: str,
    scenario_id: str,
    success: bool | None = None,
) -> np.ndarray:
    """Render frame using vectorized numpy (fast)."""
    H, W = heightmap.shape
    c = CELL_PX

    # Base: all ground color
    frame = np.full((H * c, W * c, 3), COLOR_GROUND, dtype=np.uint8)

    # Buildings (vectorized)
    building_mask = heightmap > 0
    building_expanded = np.repeat(np.repeat(building_mask, c, axis=0), c, axis=1)
    frame[building_expanded] = COLOR_BUILDING

    # Fire overlay (vectorized)
    if fire_mask is not None and fire_mask.any():
        fire_expanded = np.repeat(np.repeat(fire_mask, c, axis=0), c, axis=1)
        frame[fire_expanded] = (
            frame[fire_expanded].astype(np.float32) * 0.3
            + COLOR_FIRE.astype(np.float32) * 0.7
        ).astype(np.uint8)

    # Smoke overlay (vectorized, semi-transparent)
    if smoke_mask is not None:
        smoke_bool = smoke_mask >= 0.3
        if smoke_bool.any():
            smoke_expanded = np.repeat(np.repeat(smoke_bool, c, axis=0), c, axis=1)
            frame[smoke_expanded] = (
                frame[smoke_expanded].astype(np.float32) * 0.6
                + COLOR_SMOKE.astype(np.float32) * 0.4
            ).astype(np.uint8)

    # Forced blocks
    if forced_block_mask is not None and forced_block_mask.any():
        fb_expanded = np.repeat(np.repeat(forced_block_mask, c, axis=0), c, axis=1)
        frame[fb_expanded] = COLOR_FORCED

    # Trajectory trail (last 60 steps)
    trail = trajectory[-60:] if len(trajectory) > 60 else trajectory
    for tx, ty in trail:
        y0, y1 = ty * c, (ty + 1) * c
        x0, x1 = tx * c, (tx + 1) * c
        if 0 <= y0 < H * c and 0 <= x0 < W * c:
            frame[y0:y1, x0:x1] = COLOR_TRAIL

    # Planned path
    for px, py in path[:80]:
        y0, y1 = py * c, (py + 1) * c
        x0, x1 = px * c, (px + 1) * c
        if 0 <= y0 < H * c and 0 <= x0 < W * c:
            frame[y0:y1, x0:x1] = COLOR_PATH

    # Goal marker (3x3 cells)
    gx, gy = goal_xy
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            py, px = (gy + dy) * c, (gx + dx) * c
            if 0 <= py < H * c - c and 0 <= px < W * c - c:
                frame[py:py + c, px:px + c] = COLOR_GOAL

    # Agent marker (3x3 cells, bright blue)
    ax, ay = agent_xy
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            py, px = (ay + dy) * c, (ax + dx) * c
            if 0 <= py < H * c - c and 0 <= px < W * c - c:
                frame[py:py + c, px:px + c] = COLOR_AGENT

    return frame


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
            frame = _fast_render(
                heightmap=heightmap,
                fire_mask=dyn_state.get("fire_mask"),
                smoke_mask=dyn_state.get("smoke_mask"),
                forced_block_mask=dyn_state.get("forced_block_mask"),
                agent_xy=env.agent_xy,
                goal_xy=goal_xy,
                trajectory=trajectory,
                path=remaining_path,
                step_idx=step_idx,
                planner_id=planner_id,
                scenario_id=scenario_id,
            )
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
    scenario = sys.argv[1] if len(sys.argv) > 1 else "gov_fire_delivery_medium"
    planner = sys.argv[2] if len(sys.argv) > 2 else "aggressive_replan"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    out = f"outputs/episode_{scenario}_{planner}_s{seed}.gif"

    render_episode(scenario, planner, seed, out)
