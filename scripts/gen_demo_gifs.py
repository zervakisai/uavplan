"""Generate demo GIFs: dynamic planners on representative scenarios.

Produces animated GIF files showing the UAV navigating through dynamic
obstacles. Each frame shows the mission HUD, planned path, dynamic
obstacles (fire, NFZ, traffic), and agent trajectory.

Usage:
    python scripts/gen_demo_gifs.py [--easy] [--fps 10]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

OUT = Path("outputs/demo_gifs")

# Default: medium scenarios, one static + one adaptive + one reactive planner
RUNS = [
    ("gov_fire_delivery_medium", "astar", 42),
    ("gov_fire_delivery_medium", "aggressive_replan", 42),
    ("gov_fire_delivery_medium", "apf", 42),
    ("gov_flood_rescue_medium", "periodic_replan", 42),
    ("gov_fire_surveillance_medium", "dstar_lite", 42),
]

OSM_RUNS = [
    ("osm_penteli_fire_delivery_medium", "aggressive_replan", 42),
    ("osm_piraeus_flood_rescue_medium", "aggressive_replan", 42),
    ("osm_downtown_fire_surveillance_medium", "aggressive_replan", 42),
]

EASY_RUNS = [
    ("gov_fire_delivery_easy", "astar", 42),
    ("gov_fire_delivery_easy", "periodic_replan", 42),
    ("gov_fire_delivery_easy", "apf", 42),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate demo GIFs.")
    p.add_argument("--easy", action="store_true", help="Use easy scenarios (faster)")
    p.add_argument("--osm", action="store_true", help="Use OSM map scenarios (Greece)")
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Render every Nth frame (1=all, 5=every 5th)")
    return p.parse_args()


def generate_gif(
    scenario_id: str,
    planner_id: str,
    seed: int,
    gif_path: Path,
    fps: int = 10,
    skip_frames: int = 1,
    briefing_duration_s: float = 3.0,
) -> dict:
    """Run episode and write animated GIF with mission briefing title card."""
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="ops_full")

    frames: list[np.ndarray] = []
    frame_count = 0
    briefing_state: dict | None = None

    def frame_callback(
        heightmap: np.ndarray,
        state: dict,
        dyn_state: dict,
        cfg: object,
    ) -> None:
        nonlocal frame_count, briefing_state
        frame_count += 1

        # Capture briefing state from first frame
        if briefing_state is None:
            briefing_state = dict(state)
            # Add briefing title card frames (shown for briefing_duration_s)
            n_briefing_frames = max(1, int(briefing_duration_s * fps))
            card = renderer.render_briefing_card(heightmap, state)
            for _ in range(n_briefing_frames):
                frames.append(card)

        if frame_count % skip_frames != 0:
            return
        frame, _meta = renderer.render_frame(heightmap, state, dyn_state)
        frames.append(frame)

    t0 = time.perf_counter()
    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)
    elapsed = time.perf_counter() - t0

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

    m = result.metrics
    return {
        "success": m.get("success", False),
        "steps": m.get("executed_steps_len", 0),
        "replans": m.get("replans", 0),
        "termination": m.get("termination_reason", "?"),
        "frames_rendered": len(frames),
        "elapsed_s": elapsed,
        "gif_path": str(gif_path),
    }


def main() -> None:
    args = _parse_args()
    if args.osm:
        runs = OSM_RUNS
    elif args.easy:
        runs = EASY_RUNS
    else:
        runs = RUNS
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== UAVBench Demo GIF Generation ===")
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
