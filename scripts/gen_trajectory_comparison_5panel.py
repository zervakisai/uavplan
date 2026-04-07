#!/usr/bin/env python3
"""5-planner trajectory comparison — THE comparison figure.

5 panels side by side. Each panel = same map + same fire + that planner's
trajectory. Uses Renderer for proper basemap + ALL overlays including
traffic closures and vehicle icons.

Output: outputs/paper_figures/trajectory_comparison_5panel.{png,pdf}
"""

from __future__ import annotations

import ast
import re
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
    draw_fire, draw_smoke, draw_debris, draw_traffic, draw_vehicle_icons, draw_nfz,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "flare"
FIG_DIR = "outputs/paper_figures"
SCENARIO = "osm_penteli_pharma_delivery_medium"
SEED = 42

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

# Coefficient labels extracted from source
COEFF_LABELS = {
    "astar": "blind",
    "periodic_replan": None,
    "aggressive_replan": None,
    "incremental_astar": None,
    "apf": None,
}


def _extract(filepath: Path, varname: str) -> float:
    source = filepath.read_text()
    match = re.search(rf"^{varname}\s*=\s*(.+)", source, re.MULTILINE)
    return float(ast.literal_eval(match.group(1).strip()))


def _save(fig, name, out_dir=FIG_DIR):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def main() -> None:
    # Extract coefficients
    alpha = _extract(SRC / "planners" / "periodic_replan.py", "_RISK_ALPHA")
    beta = _extract(SRC / "planners" / "aggressive_replan.py", "_RISK_BETA")
    gamma = _extract(SRC / "planners" / "incremental_astar.py", "_RISK_GAMMA")
    delta = _extract(SRC / "planners" / "apf.py", "_RISK_DELTA")

    COEFF_LABELS["periodic_replan"] = f"\u03b1={alpha}"
    COEFF_LABELS["aggressive_replan"] = f"\u03b2={beta}"
    COEFF_LABELS["incremental_astar"] = f"\u03b3={gamma}"
    COEFF_LABELS["apf"] = f"\u03b4={delta}"

    print("5-planner trajectory comparison (with traffic overlays)")
    print(f"  Scenario: {SCENARIO}, seed={SEED}")

    # --- Step 1: Run all 5 planners, collect trajectories ---
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

    # --- Step 2: Find a good capture time (when fire is visible) ---
    # Use ~300 steps or the longest successful episode
    max_steps = 0
    for pid in PLANNER_ORDER:
        s = metrics_map[pid].get("executed_steps_len", 0)
        if s > max_steps:
            max_steps = s
    capture_t = min(300, max(max_steps // 2, 100))

    config = load_scenario(SCENARIO)
    renderer = Renderer(config, mode="paper_min")
    renderer._cell_px = 1  # minimal res — trucks max proportion of image
    cell = renderer._cell_px

    captured: dict[str, Any] = {}

    def _cb(heightmap, state, dyn_state, cfg):
        step = state.get("step_idx", 0)
        if step == capture_t and "heightmap" not in captured:
            captured["heightmap"] = heightmap
            captured["state"] = state.copy()
            captured["dyn_state"] = {
                k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in dyn_state.items()
            }

    print(f"  Capturing map state at t={capture_t}...")
    # Use longest-running planner for capture
    longest_pid = max(PLANNER_ORDER, key=lambda p: metrics_map[p].get("executed_steps_len", 0))
    run_episode(SCENARIO, longest_pid, SEED, frame_callback=_cb)

    if "heightmap" not in captured:
        # Fallback: capture at whatever step is available
        captured.clear()
        last_data: dict[str, Any] = {}

        def _cb_last(heightmap, state, dyn_state, cfg):
            last_data["heightmap"] = heightmap
            last_data["state"] = state.copy()
            last_data["dyn_state"] = {
                k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in dyn_state.items()
            }

        run_episode(SCENARIO, longest_pid, SEED, frame_callback=_cb_last)
        captured = last_data
        print(f"  Fallback: using last available frame")

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

    # All overlays in numpy layer — including vehicle icons (GIF style)
    smoke_mask = dyn_state.get("smoke_mask")
    if smoke_mask is not None:
        draw_smoke(bg_frame, smoke_mask, cell, alpha_256=40)
    fire_mask = dyn_state.get("fire_mask")
    if fire_mask is not None:
        draw_fire(bg_frame, fire_mask, cell)
    traffic_closure_mask = dyn_state.get("traffic_closure_mask")
    if traffic_closure_mask is not None:
        draw_traffic(bg_frame, traffic_closure_mask, cell)
    # Dynamic obstacles: black circles with red outline
    traffic_positions = dyn_state.get("traffic_positions")
    if traffic_positions is not None and len(traffic_positions) > 0:
        H_px, W_px = bg_frame.shape[:2]
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
                            bg_frame[py, px] = [0, 0, 0]
                        elif d2 <= r_out * r_out:
                            bg_frame[py, px] = [200, 30, 30]
    nfz_mask = dyn_state.get("nfz_mask")
    if nfz_mask is not None:
        draw_nfz(bg_frame, nfz_mask, cell)
    debris_mask = dyn_state.get("debris_mask")
    if debris_mask is not None:
        draw_debris(bg_frame, debris_mask, cell)

    # --- Step 4: Create 3-panel figure (key contrast) ---
    PANEL_PLANNERS = ["astar", "aggressive_replan", "periodic_replan"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    fig.subplots_adjust(wspace=0.05)

    for ax, pid in zip(axes, PANEL_PLANNERS):
        ax.imshow(
            bg_frame, origin="upper", interpolation="nearest",
            aspect="equal", extent=[0, W, H, 0],
        )

        # Drone at start position (blue X with white outline)
        start_xy = trajectories[pid][0] if trajectories[pid] else None
        if start_xy:
            # White outline
            ax.plot(start_xy[0], start_xy[1], "x", color="white",
                    markersize=6, markeredgewidth=1.5, zorder=2)
            # Blue cross (top)
            ax.plot(start_xy[0], start_xy[1], "x", color="#4090D0",
                    markersize=5, markeredgewidth=1.0, zorder=3)

        # Trajectory
        traj = trajectories[pid]
        success = metrics_map[pid].get("success", False)
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ls = "-" if success else ":"
            ax.plot(xs, ys, ls, color=PLANNER_COLORS[pid], linewidth=1.2,
                    path_effects=[
                        pe.Stroke(linewidth=2.0, foreground="black", alpha=0.25),
                        pe.Normal(),
                    ], zorder=5)

        # Goal (yellow cross)
        goal = state.get("goal_xy")
        if goal:
            ax.plot(goal[0], goal[1], "P", color="#E6C619", markersize=7,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=10)

        # End marker
        if traj:
            ex, ey = traj[-1]
            if success:
                ax.plot(ex, ey, "o", color="#009E73", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=11)
            else:
                ax.plot(ex, ey, "X", color="#CC3311", markersize=6,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=11)

        # Title
        coeff = COEFF_LABELS[pid]
        steps = metrics_map[pid].get("executed_steps_len", 0)
        if success:
            subtitle = f"{steps} steps"
        else:
            term = metrics_map[pid].get("termination_reason", "failed")
            subtitle = f"{term}, t={steps}"
        ax.set_title(f"{PLANNER_SHORT[pid]} ({coeff})\n{subtitle}",
                     fontsize=5.5, fontweight="normal", color="black")
        ax.axis("off")

    fig.suptitle(f"Three planners, same wildfire (seed {SEED})",
                 fontsize=8, fontweight="normal", color="black")

    _save(fig, "trajectory_comparison_5panel")

    # Print summary
    print("\nPlanner outcomes:")
    for pid in PLANNER_ORDER:
        m = metrics_map[pid]
        status = "SUCCESS" if m.get("success") else m.get("termination_reason", "?")
        print(f"  {PLANNER_SHORT[pid]:12s}: {status:20s} "
              f"steps={m.get('executed_steps_len', '?')}")


if __name__ == "__main__":
    main()
