#!/usr/bin/env python3
"""Generate failure autopsy GIF — zoomed view of last N steps of failed episodes.

Runs an episode, and if it fails, renders the last N steps in the
paper-figure style with the view zoomed around the agent's final
position. The title line annotates step, distance-to-goal, and the
termination reason.

Usage:
    python scripts/gen_failure_autopsy.py [--scenario ...] [--planner astar] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import time

import imageio.v3 as iio

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.paper_frame import (
    EpisodeCapture,
    PanelSpec,
    PaperFrameRenderer,
    growing_trajectory,
    make_capture_callback,
    pick_dyn_snapshot,
)

OUTPUT_DIR = "outputs/failure_autopsy"
DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate failure autopsy GIF.")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--planner", type=str, default="astar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--last-n", type=int, default=20,
                   help="Number of final steps to render")
    p.add_argument("--zoom-radius", type=int, default=25,
                   help="Half-width of the zoom window, in grid cells")
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"=== Failure Autopsy GIF (paper-figure style) ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Planner:  {args.planner}")
    print(f"  Seed:     {args.seed}")

    config = load_scenario(args.scenario)

    capture = EpisodeCapture()
    cb = make_capture_callback(capture, record_dyn=True)

    t0 = time.perf_counter()
    result = run_episode(args.scenario, args.planner, args.seed,
                         frame_callback=cb)
    elapsed = time.perf_counter() - t0

    m = result.metrics
    success = bool(m.get("success", False))
    term = m.get("termination_reason", "?")
    status = "OK" if success else term
    print(f"  Result: [{status}] {m.get('executed_steps_len', 0)} steps "
          f"({elapsed:.1f}s)")

    if success:
        print("  Episode succeeded — no autopsy needed.")
        return

    total_steps = len(capture.trajectory)
    if total_steps == 0:
        print("  No trajectory captured.")
        return

    last_n = min(args.last_n, total_steps)
    start_t = total_steps - last_n

    goal_xy = tuple(capture.state0.get("goal_xy", (0, 0)))
    zoom_center = capture.trajectory[-1]
    print(f"  Zooming ±{args.zoom_radius} cells on last {last_n} steps "
          f"around {zoom_center}...")

    pr = PaperFrameRenderer(config)
    frames = []
    for t in range(start_t, total_steps):
        ax_, ay_ = capture.trajectory[t]
        dist = abs(ax_ - goal_xy[0]) + abs(ay_ - goal_xy[1])
        finished = (t == total_steps - 1)
        panel = PanelSpec(
            planner_id=args.planner,
            trajectory=growing_trajectory(capture.trajectory, t),
            goal_xy=goal_xy,
            success=False if finished else None,
            termination_reason=term if finished else None,
            subtitle_override=f"step {t} · dist={dist} · {status}",
            zoom_center=zoom_center,
            zoom_radius=args.zoom_radius,
        )
        frames.append(
            pr.render_frame(
                capture.heightmap,
                capture.state0,
                pick_dyn_snapshot(capture.dyn_snapshots, t),
                panels=[panel],
                suptitle=f"Failure autopsy · {args.planner} · seed {args.seed}",
            )
        )

    gif_name = f"autopsy_{args.planner}_s{args.seed}.gif"
    gif_path = os.path.join(args.output, gif_name)
    duration_ms = 1000 // args.fps
    iio.imwrite(str(gif_path), frames, extension=".gif",
                duration=duration_ms, loop=0)
    print(f"  GIF: {gif_path} ({len(frames)} frames)")
    print("Done.")


if __name__ == "__main__":
    main()
