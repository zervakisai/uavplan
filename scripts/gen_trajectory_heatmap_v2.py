#!/usr/bin/env python3
"""Improved trajectory heatmap — 5-panel IEEE figure (Task 10).

Runs N seeds per planner, builds visit frequency heatmaps overlaid on
grey basemap with plasma colormap. Risk-averse planners → BROAD cloud,
risk-tolerant → THIN corridor.

Output: outputs/paper_figures/trajectory_heatmap_5panel.{png,pdf}

Usage:
    python scripts/gen_trajectory_heatmap_v2.py [--seeds 10] [--scenario ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.labels import (
    PLANNER_ORDER, PLANNER_SHORT,
)

ROOT = Path(__file__).resolve().parent.parent

# IEEE formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

SCENARIO = "osm_penteli_pharma_delivery_medium"


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory heatmap (improved).")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scenario", type=str, default=SCENARIO)
    args = parser.parse_args()

    cfg = load_scenario(args.scenario)
    H = W = cfg.map_size

    # Get basemap heightmap from one env reset
    from flare.envs.urban import UrbanEnvV2
    env = UrbanEnvV2(cfg)
    env.reset(seed=0)
    heightmap, _, start_xy, goal_xy = env.export_planner_inputs()

    # Building mask for grey overlay
    building_mask = heightmap > 0

    print(f"Scenario: {args.scenario}")
    print(f"Map size: {H}x{W}")
    print(f"Seeds: {args.seeds}")
    print()

    heatmaps: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}

    for pid in PLANNER_ORDER:
        visit_count = np.zeros((H, W), dtype=np.int32)
        successes = 0
        print(f"  {PLANNER_SHORT[pid]:12s}", end="", flush=True)
        t0 = time.perf_counter()
        for seed in range(args.seeds):
            r = run_episode(args.scenario, pid, seed)
            if r.metrics.get("success"):
                successes += 1
            for x, y in r.trajectory:
                if 0 <= y < H and 0 <= x < W:
                    visit_count[y, x] += 1
        elapsed = time.perf_counter() - t0
        max_v = max(visit_count.max(), 1)
        heatmaps[pid] = visit_count / max_v
        stats[pid] = {
            "successes": successes,
            "unique_cells": int((visit_count > 0).sum()),
            "max_visits": int(visit_count.max()),
        }
        print(f"  {successes}/{args.seeds} success, "
              f"{stats[pid]['unique_cells']} unique cells ({elapsed:.1f}s)")

    # Figure
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 3.0))

    for ax, pid in zip(axes, PLANNER_ORDER):
        hm = heatmaps[pid]

        # Grey basemap
        base_grey = np.full((H, W), 0.92)  # light grey ground
        base_grey[building_mask] = 0.3       # dark buildings

        ax.imshow(base_grey, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")

        # Heatmap overlay (masked where no visits)
        hm_masked = np.ma.masked_where(hm == 0, hm)
        im = ax.imshow(hm_masked, cmap="plasma", vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal", alpha=0.6)

        # Start/goal markers
        ax.plot(start_xy[0], start_xy[1], "^", color="lime", markersize=4,
                markeredgecolor="k", markeredgewidth=0.3, zorder=10)
        ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", markersize=5,
                markeredgecolor="k", markeredgewidth=0.3, zorder=10)

        # Title with stats
        s = stats[pid]
        ax.set_title(
            f"{PLANNER_SHORT[pid]}\n({s['unique_cells']} cells, "
            f"{s['successes']}/{args.seeds} SR)",
            fontsize=6.5, fontweight="bold",
        )
        ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=30, pad=0.02,
                        orientation="horizontal")
    cbar.set_label("Normalized visit frequency", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Trajectory density ({args.seeds} seeds): risk-averse explores widely, "
        "risk-tolerant commits",
        fontsize=8, fontweight="bold", y=1.02,
    )

    _save(fig, "trajectory_heatmap_5panel")


if __name__ == "__main__":
    main()
