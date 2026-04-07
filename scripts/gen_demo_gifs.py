"""Generate demo GIFs: dynamic planners on representative scenarios.

Produces animated GIF files in the paper-figure style (see
`gen_trajectory_comparison_5panel.py`). Each frame uses the cached
`paper_min` basemap with figure-style overlays, markers and matplotlib
typography so the GIFs match the static figures in the paper.

Usage:
    python scripts/gen_demo_gifs.py [--osm] [--fps 10]
"""

from __future__ import annotations

import argparse
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

OUT = Path("outputs/demo_gifs")

# OSM scenarios: one static + multiple adaptive planners
RUNS = [
    ("osm_penteli_pharma_delivery_medium", "astar", 42),
    ("osm_penteli_pharma_delivery_medium", "aggressive_replan", 42),
    ("osm_penteli_pharma_delivery_medium", "apf", 42),
    ("osm_piraeus_urban_rescue_medium", "periodic_replan", 42),
    ("osm_downtown_fire_surveillance_medium", "dstar_lite", 42),
]

OSM_RUNS = [
    ("osm_penteli_pharma_delivery_medium", "aggressive_replan", 42),
    ("osm_piraeus_urban_rescue_medium", "aggressive_replan", 42),
    ("osm_downtown_fire_surveillance_medium", "aggressive_replan", 42),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate demo GIFs.")
    p.add_argument("--osm", action="store_true", help="Use OSM-only adaptive runs")
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    p.add_argument("--skip-frames", type=int, default=3,
                   help="Render every Nth step (1=all, 3=every 3rd)")
    return p.parse_args()


def generate_gif(
    scenario_id: str,
    planner_id: str,
    seed: int,
    gif_path: Path,
    fps: int = 10,
    skip_frames: int = 3,
    briefing_duration_s: float = 2.0,
) -> dict:
    """Run episode and write a paper-figure-style animated GIF."""
    config = load_scenario(scenario_id)

    capture = EpisodeCapture()
    cb = make_capture_callback(capture, record_dyn=True)

    t0 = time.perf_counter()
    result = run_episode(scenario_id, planner_id, seed, frame_callback=cb)
    elapsed = time.perf_counter() - t0

    m = result.metrics
    success = bool(m.get("success", False))
    term = m.get("termination_reason", "?")

    pr = PaperFrameRenderer(config)
    goal_xy = tuple(capture.state0.get("goal_xy", (0, 0)))
    total_steps = len(capture.trajectory)

    frames = []
    if total_steps > 0:
        briefing_panel = PanelSpec(
            planner_id=planner_id,
            trajectory=growing_trajectory(capture.trajectory, 0),
            goal_xy=goal_xy,
            success=None,
        )
        briefing = pr.render_frame(
            capture.heightmap,
            capture.state0,
            pick_dyn_snapshot(capture.dyn_snapshots, 0),
            panels=[briefing_panel],
            suptitle=f"{scenario_id} · {planner_id} · seed {seed}",
            annotations=[
                Annotation(text="Mission briefing", xy=(0.5, 0.92),
                           fontsize=8, ha="center"),
            ],
        )
        n_brief = max(1, int(briefing_duration_s * fps))
        for _ in range(n_brief):
            frames.append(briefing)

    for t in range(0, total_steps, max(1, skip_frames)):
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

    if frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = 1000 // fps
        iio.imwrite(
            str(gif_path),
            frames,
            extension=".gif",
            duration=duration_ms,
            loop=0,
        )

    return {
        "success": success,
        "steps": m.get("executed_steps_len", 0),
        "replans": m.get("replans", 0),
        "termination": term,
        "frames_rendered": len(frames),
        "elapsed_s": elapsed,
        "gif_path": str(gif_path),
    }


def main() -> None:
    args = _parse_args()
    runs = OSM_RUNS if args.osm else RUNS
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== FLARE Demo GIF Generation (paper-figure style) ===")
    print(f"  Output: {OUT}")
    print(f"  FPS: {args.fps}")
    print(f"  Skip frames: {args.skip_frames}")
    print(f"  Episodes: {len(runs)}")
    print()

    for scenario, planner, seed in runs:
        gif_path = OUT / f"{scenario}_{planner}_s{seed}.gif"
        print(f"  {scenario} / {planner} / seed={seed} ...", end="", flush=True)
        info = generate_gif(
            scenario, planner, seed, gif_path,
            fps=args.fps, skip_frames=args.skip_frames,
        )
        status = "OK" if info["success"] else "FAIL"
        print(
            f" [{status}] {info['steps']}steps "
            f"{info['replans']}rep "
            f"{info['frames_rendered']}frames "
            f"({info['elapsed_s']:.0f}s) "
            f"-> {gif_path.name}"
        )

    print(f"\nAll done. GIFs in {OUT}/")


if __name__ == "__main__":
    main()
