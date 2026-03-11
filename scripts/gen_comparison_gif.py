#!/usr/bin/env python3
"""Generate 5-planner comparison GIF grid.

Runs the same (scenario, seed) with all 5 planners simultaneously,
composites frames into a 3×2 grid layout:

    ┌──────────┬──────────┬──────────┐
    │  A*      │ Periodic │Aggressive│
    │ (static) │ (risk)   │ (time)   │
    ├──────────┼──────────┼──────────┤
    │ Incr.A*  │   APF    │ Legend + │
    │ (memory) │(reactive)│ Timeline │
    └──────────┴──────────┴──────────┘

The 6th cell shows a live fire-area plot, distance-to-goal per planner
(5 colored lines), step counter + wind arrow.

Usage:
    python scripts/gen_comparison_gif.py [--scenario ...] [--seed 42] [--fps 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/comparison_gifs"

PLANNER_ORDER = [
    "astar", "periodic_replan", "aggressive_replan",
    "incremental_astar", "apf",
]
PLANNER_LABELS = {
    "astar": "A* (static)",
    "periodic_replan": "Periodic (α=5.0)",
    "aggressive_replan": "Aggressive (β=0.5)",
    "incremental_astar": "Incr. A* (γ=2.0)",
    "apf": "APF (δ=3.0)",
}
PLANNER_COLORS = {
    "astar": "#e74c3c",
    "periodic_replan": "#3498db",
    "aggressive_replan": "#2ecc71",
    "incremental_astar": "#f39c12",
    "apf": "#9b59b6",
}

DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"


# ---------------------------------------------------------------------------
# Per-planner episode runner with frame capture
# ---------------------------------------------------------------------------


def _run_planner_frames(
    scenario_id: str,
    planner_id: str,
    seed: int,
    skip_frames: int = 2,
) -> dict:
    """Run one episode and collect per-step data for the grid."""
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="ops_full")

    frames: list[np.ndarray] = []
    distances: list[float] = []
    fire_areas: list[int] = []
    step_indices: list[int] = []
    frame_count = 0

    def frame_callback(heightmap, state, dyn_state, cfg):
        nonlocal frame_count
        frame_count += 1

        step = state.get("step_idx", 0)
        agent = state.get("agent_xy", (0, 0))
        goal = state.get("goal_xy", (0, 0))
        dist = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])

        fire_mask = dyn_state.get("fire_mask")
        fire_area = int(fire_mask.sum()) if fire_mask is not None else 0

        step_indices.append(step)
        distances.append(dist)
        fire_areas.append(fire_area)

        if frame_count % skip_frames == 0:
            frame, _meta = renderer.render_frame(heightmap, state, dyn_state)
            frames.append(frame)

    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)

    return {
        "frames": frames,
        "distances": distances,
        "fire_areas": fire_areas,
        "step_indices": step_indices,
        "metrics": result.metrics,
    }


# ---------------------------------------------------------------------------
# Legend / timeline cell rendering
# ---------------------------------------------------------------------------


def _render_legend_cell(
    step_idx: int,
    total_steps: int,
    planner_data: dict[str, dict],
    cell_size: tuple[int, int],
) -> np.ndarray:
    """Render the 6th cell: legend + mini timeline plots."""
    h, w = cell_size
    dpi = 100
    fig, axes = plt.subplots(2, 1, figsize=(w / dpi, h / dpi), dpi=dpi)

    # Top: fire area over time
    ax_fire = axes[0]
    # Use first planner's fire data (same for all — shared env)
    first_planner = next(iter(planner_data.values()))
    fire_areas = first_planner["fire_areas"]
    steps = first_planner["step_indices"]
    if steps:
        end = min(step_idx + 1, len(fire_areas))
        ax_fire.fill_between(steps[:end], fire_areas[:end], alpha=0.3, color="#e74c3c")
        ax_fire.plot(steps[:end], fire_areas[:end], color="#e74c3c", linewidth=1)
    ax_fire.set_ylabel("Fire cells", fontsize=7)
    ax_fire.set_title(f"Step {step_idx}/{total_steps}", fontsize=8, fontweight="bold")
    ax_fire.tick_params(labelsize=6)

    # Bottom: distance to goal per planner
    ax_dist = axes[1]
    for pid, data in planner_data.items():
        dists = data["distances"]
        s = data["step_indices"]
        if s:
            end = min(step_idx + 1, len(dists))
            ax_dist.plot(
                s[:end], dists[:end],
                color=PLANNER_COLORS.get(pid, "gray"),
                linewidth=1, label=PLANNER_LABELS.get(pid, pid)[:8],
            )
    ax_dist.set_ylabel("Dist to goal", fontsize=7)
    ax_dist.set_xlabel("Step", fontsize=7)
    ax_dist.tick_params(labelsize=6)
    ax_dist.legend(fontsize=5, loc="upper right", ncol=2)

    fig.tight_layout(pad=0.5)

    # Render to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = buf[:, :, :3].copy()  # RGBA → RGB
    plt.close(fig)

    # Resize to target
    from PIL import Image
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((w, h), Image.LANCZOS)
    return np.array(pil_img)


# ---------------------------------------------------------------------------
# Grid compositor
# ---------------------------------------------------------------------------


def _composite_grid(
    planner_frames: dict[str, list[np.ndarray]],
    planner_data: dict[str, dict],
    total_steps: int,
) -> list[np.ndarray]:
    """Composite per-planner frames into 3×2 grid frames."""
    # Determine frame count (min across planners)
    n_frames = min(len(f) for f in planner_frames.values())
    if n_frames == 0:
        return []

    # Get single frame size
    sample = next(iter(planner_frames.values()))[0]
    fh, fw = sample.shape[:2]

    grid_frames = []
    for fi in range(n_frames):
        # 3×2 grid
        row1 = []
        row2 = []
        for i, pid in enumerate(PLANNER_ORDER):
            frame = planner_frames[pid][min(fi, len(planner_frames[pid]) - 1)]
            # Resize if needed
            if frame.shape[:2] != (fh, fw):
                from PIL import Image
                pil = Image.fromarray(frame).resize((fw, fh), Image.LANCZOS)
                frame = np.array(pil)
            if i < 3:
                row1.append(frame)
            else:
                row2.append(frame)

        # 6th cell: legend/timeline
        step_approx = int(fi / max(n_frames - 1, 1) * total_steps)
        legend = _render_legend_cell(step_approx, total_steps, planner_data, (fh, fw))
        row2.append(legend)

        grid = np.vstack([
            np.hstack(row1),
            np.hstack(row2),
        ])
        grid_frames.append(grid)

    return grid_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 5-planner comparison GIF.")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--skip-frames", type=int, default=3)
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=== 5-Planner Comparison GIF ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Seed:     {args.seed}")
    print(f"  FPS:      {args.fps}")
    print()

    # Run all 5 planners
    planner_frames: dict[str, list[np.ndarray]] = {}
    planner_data: dict[str, dict] = {}
    total_steps = 0

    for pid in PLANNER_ORDER:
        print(f"  Running {pid}...", end="", flush=True)
        t0 = time.perf_counter()
        data = _run_planner_frames(args.scenario, pid, args.seed, args.skip_frames)
        elapsed = time.perf_counter() - t0
        planner_frames[pid] = data["frames"]
        planner_data[pid] = data
        total_steps = max(total_steps, len(data["step_indices"]))
        m = data["metrics"]
        status = "OK" if m.get("success") else m.get("termination_reason", "?")
        print(f" [{status}] {len(data['frames'])} frames ({elapsed:.1f}s)")

    # Composite into grid
    print("\n  Compositing grid...", end="", flush=True)
    grid_frames = _composite_grid(planner_frames, planner_data, total_steps)
    print(f" {len(grid_frames)} frames")

    # Write GIF
    gif_name = f"comparison_{args.scenario}_s{args.seed}.gif"
    gif_path = os.path.join(args.output, gif_name)
    if grid_frames:
        duration_ms = 1000 // args.fps
        iio.imwrite(str(gif_path), grid_frames, extension=".gif",
                     duration=duration_ms, loop=0)
        print(f"\n  GIF: {gif_path} ({len(grid_frames)} frames)")
    else:
        print("\n  Warning: no frames to write")

    print("Done.")


if __name__ == "__main__":
    main()
