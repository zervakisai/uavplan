"""Render a single episode to GIF in the paper-figure style.

Uses run_episode() + frame_callback for consistency with the benchmark
runner, then emits frames through `PaperFrameRenderer` so the output is
visually identical to the static paper figures (see
`gen_trajectory_comparison_5panel.py`).

Usage:
    python scripts/render_episode.py [scenario_id] [planner_id] [seed]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import imageio.v3 as iio

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.paper_frame import (
    Annotation,
    EpisodeCapture,
    PanelSpec,
    PaperFrameRenderer,
    growing_trajectory,
    make_capture_callback,
    pick_dyn_snapshot,
)


def render_episode_gif(
    scenario_id: str,
    planner_id: str,
    seed: int,
    out_path: str,
    fps: int = 8,
    frame_skip: int = 3,
    briefing_duration_s: float = 2.0,
) -> dict:
    """Run one episode and save a paper-figure-style animated GIF."""
    config = load_scenario(scenario_id)

    capture = EpisodeCapture()
    cb = make_capture_callback(capture, record_dyn=True)

    print(f"Running {scenario_id} / {planner_id} / seed={seed} ...")
    t0 = time.perf_counter()
    result = run_episode(scenario_id, planner_id, seed, frame_callback=cb)
    elapsed = time.perf_counter() - t0

    m = result.metrics
    success = bool(m.get("success", False))
    term = m.get("termination_reason", "?")

    # Render frames from captured data in the paper-figure style.
    pr = PaperFrameRenderer(config)
    goal_xy = tuple(capture.state0.get("goal_xy", (0, 0)))
    total_steps = len(capture.trajectory)

    frames = []
    # Briefing "title" frame — first frame with a figure annotation,
    # held for briefing_duration_s seconds.
    if total_steps > 0:
        title = f"{scenario_id}"
        subtitle = f"{planner_id} · seed {seed}"
        annotations = [
            Annotation(text=title, xy=(0.5, 0.96), fontsize=9, ha="center"),
            Annotation(text=subtitle, xy=(0.5, 0.90), fontsize=7, ha="center"),
        ]
        first_panel = PanelSpec(
            planner_id=planner_id,
            trajectory=growing_trajectory(capture.trajectory, 0),
            goal_xy=goal_xy,
            success=None,
        )
        briefing = pr.render_frame(
            capture.heightmap,
            capture.state0,
            pick_dyn_snapshot(capture.dyn_snapshots, 0),
            panels=[first_panel],
            suptitle=f"{scenario_id} · {planner_id} · seed {seed}",
            annotations=annotations,
        )
        n_brief = max(1, int(briefing_duration_s * fps))
        for _ in range(n_brief):
            frames.append(briefing)

    # Per-step frames
    for t in range(0, total_steps, max(1, frame_skip)):
        finished = (t >= total_steps - 1)
        panel = PanelSpec(
            planner_id=planner_id,
            trajectory=growing_trajectory(capture.trajectory, t),
            goal_xy=goal_xy,
            success=success if finished else None,
            termination_reason=(term if finished and not success else None),
        )
        frames.append(
            pr.render_frame(
                capture.heightmap,
                capture.state0,
                pick_dyn_snapshot(capture.dyn_snapshots, t),
                panels=[panel],
                suptitle=f"{scenario_id} · {planner_id} · step {t}",
            )
        )

    # Save GIF
    if frames:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        duration_ms = 1000 // fps
        iio.imwrite(str(out_path), frames, extension=".gif",
                    duration=duration_ms, loop=0)

    print(f"\nResult:")
    print(f"  Scenario:  {scenario_id}")
    print(f"  Planner:   {planner_id}")
    print(f"  Seed:      {seed}")
    print(f"  Steps:     {m.get('executed_steps_len', 0)}")
    print(f"  Replans:   {m.get('replans', 0)}")
    print(f"  Outcome:   {term}")
    print(f"  Objective: {'completed' if success else 'not completed'}")
    print(f"  Frames:    {len(frames)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  GIF:       {out_path}")
    return {"success": success,
            "steps": m.get("executed_steps_len", 0),
            "frames": len(frames)}


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "osm_penteli_pharma_delivery_medium"
    planner = sys.argv[2] if len(sys.argv) > 2 else "aggressive_replan"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    out = f"outputs/episode_{scenario}_{planner}_s{seed}.gif"

    render_episode_gif(scenario, planner, seed, out)
