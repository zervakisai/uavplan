#!/usr/bin/env python3
"""5-panel planner trajectory comparison (one panel per planner).

Runs all 5 planners on the same scenario/seed, captures map state at t=212
(corridor blockage moment), and shows each planner's trajectory on its own
panel with the shared basemap + fire overlay.

Output: outputs/paper_figures/planner_comparison_5panel.{png,pdf}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.renderer import Renderer
from flare.visualization.labels import (
    PLANNER_ORDER, PLANNER_SHORT, PLANNER_COLORS,
)
from flare.visualization.overlays import (
    draw_fire, draw_smoke, draw_debris,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_DIR = "outputs/paper_figures"
SCENARIO = "osm_penteli_pharma_delivery_medium"
SEED = 42
CAPTURE_T = 212  # corridor blockage moment

COEFF_LABELS = {
    "astar": "\u2014",                    # em dash
    "periodic_replan": "\u03b1=5.0",      # alpha
    "aggressive_replan": "\u03b2=0.5",    # beta
    "incremental_astar": "\u03b3=2.0",    # gamma
    "apf": "\u03b4=3.0",                  # delta
}

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


def _save(fig, name, out_dir=FIG_DIR):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def main() -> None:
    print("5-panel planner comparison")
    print(f"  Scenario: {SCENARIO}, seed={SEED}, capture t={CAPTURE_T}")

    # --- Step 1: Run all 5 planners ---
    trajectories: dict[str, list[tuple[int, int]]] = {}
    metrics_map: dict[str, dict] = {}

    for pid in PLANNER_ORDER:
        print(f"  Running {pid}...", end="", flush=True)
        r = run_episode(SCENARIO, pid, SEED)
        trajectories[pid] = r.trajectory
        metrics_map[pid] = r.metrics
        success = r.metrics.get("success", False)
        steps = r.metrics.get("executed_steps_len", 0)
        term = r.metrics.get("termination_reason", "")
        status = "OK" if success else term
        print(f" {status} ({steps} steps)")

    # --- Step 2: Capture map state at CAPTURE_T ---
    config = load_scenario(SCENARIO)
    renderer = Renderer(config, mode="paper_min")
    cell = renderer._cell_px

    captured: dict[str, Any] = {}

    def _cb(heightmap, state, dyn_state, cfg):
        step = state.get("step_idx", 0)
        if step == CAPTURE_T and "heightmap" not in captured:
            captured["heightmap"] = heightmap
            captured["state"] = state.copy()
            captured["dyn_state"] = {
                k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in dyn_state.items()
            }

    print(f"  Capturing map state at t={CAPTURE_T}...")
    run_episode(SCENARIO, "aggressive_replan", SEED, frame_callback=_cb)

    if "heightmap" not in captured:
        print("  ERROR: Could not capture map state.")
        sys.exit(1)

    # --- Step 3: Build background frame ---
    heightmap = captured["heightmap"]
    state = captured["state"]
    dyn_state = captured["dyn_state"]
    H, W = heightmap.shape

    bg_frame = renderer._render_basemap(
        heightmap, H, W, cell,
        state.get("landuse_map"),
        state.get("roads_mask"),
    )

    # Add dynamic overlays
    smoke_mask = dyn_state.get("smoke_mask")
    if smoke_mask is not None:
        draw_smoke(bg_frame, smoke_mask, cell, alpha_256=51)
    fire_mask = dyn_state.get("fire_mask")
    if fire_mask is not None:
        draw_fire(bg_frame, fire_mask, cell)
    debris_mask = dyn_state.get("debris_mask")
    if debris_mask is not None:
        draw_debris(bg_frame, debris_mask, cell)

    # --- Step 4: Create 5-panel figure ---
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.5))
    fig.subplots_adjust(wspace=0.05)

    for ax, pid in zip(axes, PLANNER_ORDER):
        # Shared background
        ax.imshow(
            bg_frame, origin="upper", interpolation="nearest",
            aspect="equal", extent=[0, W, H, 0],
        )

        # This planner's trajectory
        traj = trajectories[pid]
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(
                xs, ys, color=PLANNER_COLORS[pid], linewidth=2.0,
                path_effects=[
                    pe.Stroke(linewidth=3.5, foreground="white", alpha=0.6),
                    pe.Normal(),
                ],
            )

        # Start marker
        if traj:
            ax.plot(traj[0][0], traj[0][1], "^", color="lime", markersize=5,
                    markeredgecolor="k", markeredgewidth=0.3, zorder=10)

        # End marker: success (gold star) or failure (red X)
        success = metrics_map[pid].get("success", False)
        if traj:
            ex, ey = traj[-1]
            if success:
                ax.plot(ex, ey, "*", color="gold", markersize=8,
                        markeredgecolor="k", markeredgewidth=0.3, zorder=10)
            else:
                ax.plot(ex, ey, "X", color="red", markersize=7,
                        markeredgecolor="k", markeredgewidth=0.3, zorder=10)

        # Title: planner name + coefficient + outcome
        coeff = COEFF_LABELS[pid]
        steps = metrics_map[pid].get("executed_steps_len", 0)
        if success:
            subtitle = f"{steps} steps"
        else:
            subtitle = f"Failed t={steps}"
        ax.set_title(
            f"{PLANNER_SHORT[pid]} ({coeff})\n{subtitle}",
            fontsize=7, fontweight="bold",
        )
        ax.axis("off")

    _save(fig, "planner_comparison_5panel")

    # Print summary
    print("\nPlanner outcomes:")
    for pid in PLANNER_ORDER:
        m = metrics_map[pid]
        status = "SUCCESS" if m.get("success") else m.get("termination_reason", "?")
        print(f"  {PLANNER_SHORT[pid]:12s}: {status:12s} "
              f"steps={m.get('executed_steps_len', '?')}")


if __name__ == "__main__":
    main()
