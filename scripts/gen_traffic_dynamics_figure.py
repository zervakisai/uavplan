#!/usr/bin/env python3
"""Traffic dynamics 3-panel figure — the MISSING figure.

3-panel temporal strip on Piraeus (densest traffic):
Panel 1: "Emergency operations" (t~30) — vehicles on roads, small fire
Panel 2: "Fire closes roads" (t~80) — road closures, vehicles diverted
Panel 3: "Compound blockage" (t~150) — fire + closures + vehicles = corridor cut

Output: outputs/paper_figures/traffic_dynamics.{png,pdf}
"""

from __future__ import annotations

import sys
from dataclasses import replace
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
from flare.visualization.overlays import (
    draw_fire, draw_smoke, draw_traffic, draw_vehicle_icons, draw_nfz, draw_debris,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = "outputs/paper_figures"
SCENARIO = "osm_piraeus_urban_rescue_medium"
PLANNER = "aggressive_replan"
SEED = 42

# Capture steps for the 3 panels
CAPTURE_STEPS = [30, 80, 150]

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
    print("Traffic dynamics 3-panel figure")
    print(f"  Scenario: {SCENARIO}, planner={PLANNER}, seed={SEED}")

    config = load_scenario(SCENARIO)
    config_mod = replace(config, enable_collapse=True, collapse_delay=80, debris_prob=0.6)

    renderer = Renderer(config_mod, mode="paper_min")
    renderer._cell_px = 1  # minimal res — trucks max proportion
    cell = renderer._cell_px

    # Collect frames at target steps
    frames: dict[int, dict[str, Any]] = {}
    trajectory: list[tuple[int, int]] = []

    def cb(heightmap, state, dyn_state, cfg):
        step = state.get("step_idx", 0)
        # Always track trajectory
        agent_xy = state.get("agent_xy")
        if agent_xy:
            trajectory.append(agent_xy)

        if step in CAPTURE_STEPS and step not in frames:
            frames[step] = {
                "heightmap": heightmap.copy() if hasattr(heightmap, "copy") else heightmap,
                "state": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()},
                "dyn_state": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in dyn_state.items()},
                "traj_so_far": list(trajectory),
            }

    print("  Running episode to capture 3 snapshots...")
    result = run_episode(SCENARIO, PLANNER, SEED, frame_callback=cb, config_override=config_mod)

    success = result.metrics.get("success", False)
    total_steps = result.metrics.get("executed_steps_len", 0)
    print(f"  Success: {success}, Steps: {total_steps}")
    print(f"  Captured frames at steps: {sorted(frames.keys())}")

    # If some capture steps weren't reached, use what we have
    actual_steps = sorted(frames.keys())
    if len(actual_steps) < 3:
        print(f"  WARNING: Only captured {len(actual_steps)} frames")
        # Pad with the last available frame
        while len(actual_steps) < 3:
            actual_steps.append(actual_steps[-1])

    # Panel labels
    panel_labels = [
        "(a) Emergency operations",
        "(b) Fire closes roads",
        "(c) Compound blockage",
    ]
    panel_annotations = [
        None,
        "Fire \u2192 road closures\n(interaction engine)",
        "Fire + vehicles + closures\n= corridor cut",
    ]

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.8))
    fig.subplots_adjust(wspace=0.08)

    for i, (ax, step) in enumerate(zip(axes, actual_steps[:3])):
        if step not in frames:
            ax.set_facecolor("#f0f0f0")
            ax.set_title(panel_labels[i], fontsize=7, fontweight="normal", color="black")
            ax.axis("off")
            continue

        f = frames[step]
        heightmap = f["heightmap"]
        state = f["state"]
        dyn_state = f["dyn_state"]
        H, W = heightmap.shape

        # Render basemap
        bg = renderer._render_basemap(
            heightmap, H, W, cell,
            state.get("landuse_map"),
            state.get("roads_mask"),
        )

        # Overlays (order matters for z-layering)
        smoke_mask = dyn_state.get("smoke_mask")
        if smoke_mask is not None:
            draw_smoke(bg, smoke_mask, cell, alpha_256=40)

        fire_mask = dyn_state.get("fire_mask")
        if fire_mask is not None:
            draw_fire(bg, fire_mask, cell)

        traffic_closure_mask = dyn_state.get("traffic_closure_mask")
        if traffic_closure_mask is not None:
            draw_traffic(bg, traffic_closure_mask, cell)

        # Dynamic obstacles: black circles with red outline
        traffic_positions = dyn_state.get("traffic_positions")
        if traffic_positions is not None and len(traffic_positions) > 0:
            H_px, W_px = bg.shape[:2]
            r = max(5, cell * 3)
            r_out = r + 2
            for vy, vx in traffic_positions:
                cy, cx = int(vy * cell + cell // 2), int(vx * cell + cell // 2)
                for dy in range(-r_out, r_out + 1):
                    for dx in range(-r_out, r_out + 1):
                        d2 = dy * dy + dx * dx
                        py, px = cy + dy, cx + dx
                        if 0 <= py < H_px and 0 <= px < W_px:
                            if d2 <= r * r:
                                bg[py, px] = [0, 0, 0]
                            elif d2 <= r_out * r_out:
                                bg[py, px] = [200, 30, 30]

        nfz_mask = dyn_state.get("nfz_mask")
        if nfz_mask is not None:
            draw_nfz(bg, nfz_mask, cell)

        debris_mask = dyn_state.get("debris_mask")
        if debris_mask is not None:
            draw_debris(bg, debris_mask, cell)

        ax.imshow(bg, origin="upper", interpolation="bilinear",
                  aspect="equal", extent=[0, W, H, 0])

        # Drone trajectory — thin line with subtle shadow
        traj = f["traj_so_far"]
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, "-", color="#00A0C0", linewidth=1.0, alpha=0.9,
                    path_effects=[
                        pe.Stroke(linewidth=1.6, foreground="black", alpha=0.2),
                        pe.Normal(),
                    ])

        # Agent — drone as blue X with white outline
        agent_xy = state.get("agent_xy")
        if agent_xy:
            # White outline
            ax.plot(agent_xy[0], agent_xy[1], "x", color="white",
                    markersize=6, markeredgewidth=1.5, zorder=8)
            # Blue cross (top)
            ax.plot(agent_xy[0], agent_xy[1], "x", color="#4090D0",
                    markersize=5, markeredgewidth=1.0, zorder=9)

        # Count stats for label
        fire_count = int(fire_mask.sum()) if fire_mask is not None else 0
        closure_count = int(traffic_closure_mask.sum()) if traffic_closure_mask is not None else 0
        n_vehicles = len(traffic_positions) if traffic_positions is not None else 0

        # Stats label
        stats_text = f"t={step}  fire={fire_count}  closures={closure_count}"
        if n_vehicles > 0:
            stats_text += f"  obstacles={n_vehicles}"
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes, fontsize=5, color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none",
                      boxstyle="round,pad=0.2"),
            va="bottom", ha="left",
        )

        # Annotation
        if panel_annotations[i]:
            ax.text(
                0.98, 0.98, panel_annotations[i],
                transform=ax.transAxes, fontsize=5.5, color="black",
                fontweight="bold", va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.85, pad=2, edgecolor="black",
                          linewidth=0.5, boxstyle="round,pad=0.3"),
            )

        ax.set_title(panel_labels[i], fontsize=7, fontweight="normal", color="black")
        ax.axis("off")

    fig.suptitle(
        "Coupled urban dynamics: emergency vehicles, fire-induced road closures, "
        "and corridor blockage",
        fontsize=8, fontweight="normal", color="black",
    )

    _save(fig, "traffic_dynamics")


if __name__ == "__main__":
    main()
