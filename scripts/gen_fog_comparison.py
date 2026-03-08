#!/usr/bin/env python3
"""Generate fog-of-war split-screen comparison GIF.

Shows side-by-side: what the planner sees (fog-filtered) vs ground truth.

Usage:
    python scripts/gen_fog_comparison.py [--scenario ...] [--seed 42] [--fps 8]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from uavbench.benchmark.runner import run_episode
from uavbench.blocking import compute_risk_cost_map
from uavbench.dynamics.fog_of_war import FogOfWar
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

OUTPUT_DIR = "outputs/fog_comparison"
DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fog split-screen GIF.")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--planner", type=str, default="aggressive_replan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--skip-frames", type=int, default=3)
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    config = load_scenario(args.scenario)
    renderer = Renderer(config, mode="ops_full")

    # Force fog for this visualization
    fog = FogOfWar((config.map_size, config.map_size), config.sensor_radius or 50)

    frames: list[np.ndarray] = []
    frame_count = 0

    def frame_callback(heightmap, state, dyn_state, cfg):
        nonlocal frame_count
        frame_count += 1
        if frame_count % args.skip_frames != 0:
            return

        agent_xy = state.get("agent_xy", (0, 0))
        step = state.get("step_idx", 0)

        # Fog-filtered view
        fog_state = fog.observe(agent_xy, dyn_state, step)
        # Ground truth
        ground_truth = dyn_state

        frame = renderer.render_fog_comparison(
            heightmap, state, fog_state, ground_truth,
        )
        frames.append(frame)

    print(f"=== Fog Split-Screen Comparison ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Planner:  {args.planner}")
    print(f"  Seed:     {args.seed}")

    t0 = time.perf_counter()
    result = run_episode(args.scenario, args.planner, args.seed,
                         frame_callback=frame_callback)
    elapsed = time.perf_counter() - t0

    m = result.metrics
    status = "OK" if m.get("success") else m.get("termination_reason", "?")
    print(f"  Result: [{status}] {m.get('executed_steps_len', 0)} steps ({elapsed:.1f}s)")

    gif_name = f"fog_comparison_{args.planner}_s{args.seed}.gif"
    gif_path = os.path.join(args.output, gif_name)
    if frames:
        duration_ms = 1000 // args.fps
        iio.imwrite(str(gif_path), frames, extension=".gif",
                     duration=duration_ms, loop=0)
        print(f"  GIF: {gif_path} ({len(frames)} frames)")
    else:
        print("  Warning: no frames captured")

    print("Done.")


if __name__ == "__main__":
    main()
