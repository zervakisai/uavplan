"""Render a single episode to GIF for visual inspection.

Uses run_episode() + frame_callback for consistency with the benchmark
runner. No divergent step loops.

Usage:
    python scripts/render_episode.py [scenario_id] [planner_id] [seed]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer


def render_episode_gif(
    scenario_id: str,
    planner_id: str,
    seed: int,
    out_path: str,
    fps: int = 8,
    frame_skip: int = 3,
    briefing_duration_s: float = 3.0,
) -> dict:
    """Run one episode and save animated GIF via frame_callback."""
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="ops_full")

    frames: list[np.ndarray] = []
    frame_count = 0
    briefing_done = False

    def frame_callback(
        heightmap: np.ndarray,
        state: dict,
        dyn_state: dict,
        cfg: object,
    ) -> None:
        nonlocal frame_count, briefing_done
        frame_count += 1

        # Briefing title card on first frame
        if not briefing_done:
            briefing_done = True
            n_briefing = max(1, int(briefing_duration_s * fps))
            card = renderer.render_briefing_card(heightmap, state)
            for _ in range(n_briefing):
                frames.append(card)

        if frame_count % frame_skip != 0:
            return
        frame, _meta = renderer.render_frame(heightmap, state, dyn_state)
        frames.append(frame)

    print(f"Running {scenario_id} / {planner_id} / seed={seed} ...")
    t0 = time.perf_counter()
    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)
    elapsed = time.perf_counter() - t0

    # Save GIF
    if frames:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        duration_ms = 1000 // fps
        iio.imwrite(str(out_path), frames, extension=".gif", duration=duration_ms, loop=0)

    m = result.metrics
    success = m.get("success", False)
    term = m.get("termination_reason", "?")

    print(f"\nResult:")
    print(f"  Scenario:  {scenario_id}")
    print(f"  Planner:   {planner_id}")
    print(f"  Seed:      {seed}")
    print(f"  Steps:     {m.get('executed_steps_len', 0)}")
    print(f"  Replans:   {m.get('replans', 0)}")
    print(f"  Outcome:   {term}")
    print(f"  Objective:  {'completed' if success else 'not completed'}")
    print(f"  Frames:    {len(frames)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  GIF:       {out_path}")
    return {"success": success, "steps": m.get("executed_steps_len", 0), "frames": len(frames)}


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "osm_penteli_pharma_delivery_medium"
    planner = sys.argv[2] if len(sys.argv) > 2 else "aggressive_replan"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    out = f"outputs/episode_{scenario}_{planner}_s{seed}.gif"

    render_episode_gif(scenario, planner, seed, out)
