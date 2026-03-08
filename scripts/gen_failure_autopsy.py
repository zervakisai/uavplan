#!/usr/bin/env python3
"""Generate failure autopsy GIF — zoomed view of last N steps of failed episodes.

Runs an episode, and if it fails, extracts the last 20 frames,
crops + zooms 3x around the agent's final position, and annotates
each frame with step, distance-to-goal, and reject reasons.

Usage:
    python scripts/gen_failure_autopsy.py [--scenario ...] [--planner astar] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import time

import imageio.v3 as iio
import numpy as np

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer
from uavbench.visualization.hud import _render_text

OUTPUT_DIR = "outputs/failure_autopsy"
DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate failure autopsy GIF.")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--planner", type=str, default="astar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--last-n", type=int, default=20, help="Number of final frames to capture")
    p.add_argument("--zoom", type=float, default=3.0, help="Zoom factor")
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def _crop_and_zoom(
    frame: np.ndarray,
    center_xy: tuple[int, int],
    cell: int,
    zoom: float,
) -> np.ndarray:
    """Crop frame around center and zoom in."""
    h, w = frame.shape[:2]
    # Center in pixel coords
    cx_px = center_xy[0] * cell + cell // 2
    cy_px = center_xy[1] * cell + cell // 2
    # Crop radius (smaller = more zoom)
    crop_w = int(w / zoom / 2)
    crop_h = int(h / zoom / 2)
    # Clamp
    x0 = max(0, cx_px - crop_w)
    y0 = max(0, cy_px - crop_h)
    x1 = min(w, cx_px + crop_w)
    y1 = min(h, cy_px + crop_h)
    cropped = frame[y0:y1, x0:x1]
    # Upscale back to original size
    from PIL import Image
    pil = Image.fromarray(cropped)
    pil = pil.resize((w, h), Image.LANCZOS)
    return np.array(pil)


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    config = load_scenario(args.scenario)
    renderer = Renderer(config, mode="ops_full")
    cell = renderer._cell_px

    # Collect ALL frames (we'll only keep the last N)
    all_frames: list[tuple[np.ndarray, dict]] = []

    def frame_callback(heightmap, state, dyn_state, cfg):
        frame, _meta = renderer.render_frame(heightmap, state, dyn_state)
        agent_xy = state.get("agent_xy", (0, 0))
        goal_xy = state.get("goal_xy", (0, 0))
        step = state.get("step_idx", 0)
        dist = abs(agent_xy[0] - goal_xy[0]) + abs(agent_xy[1] - goal_xy[1])
        all_frames.append((frame, {
            "step": step,
            "agent_xy": agent_xy,
            "goal_xy": goal_xy,
            "dist": dist,
        }))

    print(f"=== Failure Autopsy GIF ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Planner:  {args.planner}")
    print(f"  Seed:     {args.seed}")

    t0 = time.perf_counter()
    result = run_episode(args.scenario, args.planner, args.seed,
                         frame_callback=frame_callback)
    elapsed = time.perf_counter() - t0

    m = result.metrics
    success = m.get("success", False)
    status = "OK" if success else m.get("termination_reason", "?")
    print(f"  Result: [{status}] {m.get('executed_steps_len', 0)} steps ({elapsed:.1f}s)")

    if success:
        print("  Episode succeeded — no autopsy needed.")
        return

    # Extract last N frames
    last_n = min(args.last_n, len(all_frames))
    if last_n == 0:
        print("  No frames captured.")
        return

    final_frames = all_frames[-last_n:]
    # Use last agent position as zoom center
    zoom_center = final_frames[-1][1]["agent_xy"]

    print(f"  Zooming {args.zoom}x on last {last_n} frames around {zoom_center}...")

    autopsy_frames: list[np.ndarray] = []
    for frame, meta in final_frames:
        # Crop and zoom
        zoomed = _crop_and_zoom(frame, zoom_center, cell, args.zoom)
        # Annotate
        h, w = zoomed.shape[:2]
        # Dark bar at bottom for text
        bar_h = 30
        bar = np.full((bar_h, w, 3), 20, dtype=np.uint8)
        annotated = np.vstack([zoomed, bar])
        _render_text(annotated, f"STEP {meta['step']}  DIST={meta['dist']}  {status}",
                     4, h + 4, (255, 100, 100), 2)
        autopsy_frames.append(annotated)

    gif_name = f"autopsy_{args.planner}_s{args.seed}.gif"
    gif_path = os.path.join(args.output, gif_name)
    duration_ms = 1000 // args.fps
    iio.imwrite(str(gif_path), autopsy_frames, extension=".gif",
                 duration=duration_ms, loop=0)
    print(f"  GIF: {gif_path} ({len(autopsy_frames)} frames)")
    print("Done.")


if __name__ == "__main__":
    main()
