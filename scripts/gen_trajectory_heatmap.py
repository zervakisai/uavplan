#!/usr/bin/env python3
"""Generate trajectory heatmap overlay — multi-seed path frequency visualization.

Runs N seeds per planner, collects all trajectories, and overlays visit
frequency as a viridis heatmap on the base map. One panel per planner,
IEEE two-column width figure.

Usage:
    python scripts/gen_trajectory_heatmap.py [--scenario ...] [--seeds 30]
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

OUTPUT_DIR = "outputs/trajectory_heatmaps"
DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"

PLANNER_ORDER = [
    "astar", "periodic_replan", "aggressive_replan",
    "incremental_astar", "apf",
]
PLANNER_LABELS = {
    "astar": "A*",
    "periodic_replan": "Periodic",
    "aggressive_replan": "Aggressive",
    "incremental_astar": "Incr. A*",
    "apf": "APF",
}

IEEE_TWO_COL_WIDTH = 7.16


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate trajectory heatmap overlay.")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def _collect_trajectories(
    scenario_id: str,
    planner_id: str,
    seeds: list[int],
) -> tuple[list[list[tuple[int, int]]], int]:
    """Collect trajectories across seeds. Returns (trajectories, map_size)."""
    trajectories = []
    map_size = 100  # default
    for seed in seeds:
        result = run_episode(scenario_id, planner_id, seed)
        trajectories.append(result.trajectory)
        config = load_scenario(scenario_id)
        map_size = config.map_size
    return trajectories, map_size


def _build_heatmap(
    trajectories: list[list[tuple[int, int]]],
    map_size: int,
) -> np.ndarray:
    """Build visit frequency heatmap from trajectories."""
    heatmap = np.zeros((map_size, map_size), dtype=np.float64)
    for traj in trajectories:
        for x, y in traj:
            if 0 <= y < map_size and 0 <= x < map_size:
                heatmap[y, x] += 1
    # Normalize to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap


def _render_basemap(scenario_id: str) -> np.ndarray:
    """Render the static basemap for overlay."""
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="paper_min")
    # Need a dummy heightmap — get from env
    from uavbench.envs.urban import UrbanEnvV2
    env = UrbanEnvV2(config)
    env.reset(seed=0)
    heightmap, _, start_xy, goal_xy = env.export_planner_inputs()
    state = {
        "agent_xy": start_xy,
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "landuse_map": getattr(env, '_landuse_map', None),
        "roads_mask": getattr(env, '_roads', None),
    }
    frame, _ = renderer.render_frame(heightmap, state)
    return frame


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)
    seeds = list(range(args.seeds))

    print("=== Trajectory Heatmap Overlay ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Seeds:    {args.seeds}")
    print()

    config = load_scenario(args.scenario)
    map_size = config.map_size

    # Collect trajectories for all planners
    all_heatmaps: dict[str, np.ndarray] = {}
    for pid in PLANNER_ORDER:
        print(f"  Running {pid}...", end="", flush=True)
        t0 = time.perf_counter()
        trajectories, _ = _collect_trajectories(args.scenario, pid, seeds)
        elapsed = time.perf_counter() - t0
        heatmap = _build_heatmap(trajectories, map_size)
        all_heatmaps[pid] = heatmap
        n_success = sum(
            1 for traj in trajectories
            if len(traj) > 0
        )
        print(f" {len(trajectories)} episodes ({elapsed:.1f}s)")

    # Create 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(IEEE_TWO_COL_WIDTH, 2.0))

    for ax, pid in zip(axes, PLANNER_ORDER):
        heatmap = all_heatmaps[pid]
        # Show heatmap with viridis colormap
        im = ax.imshow(heatmap, cmap="viridis", interpolation="nearest",
                       vmin=0, vmax=1, aspect="equal")
        ax.set_title(PLANNER_LABELS[pid], fontsize=8, fontweight="bold")
        ax.axis("off")

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label("Visit frequency", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Trajectory Frequency Heatmap (30 seeds)",
                 fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.95, 0.93])

    # Save
    png_path = os.path.join(args.output, "trajectory_heatmap.png")
    pdf_path = os.path.join(args.output, "trajectory_heatmap.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    print("Done.")


if __name__ == "__main__":
    main()
