#!/usr/bin/env python3
"""Corridor blockage — HERO FIGURE (Task 6).

Shows fire cutting the primary corridor and how each planner responds.
Three panels: before blockage, at blockage, aftermath.

Output: outputs/paper_figures/corridor_blockage_dramatic.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.labels import (
    PLANNER_ORDER, PLANNER_COLORS, PLANNER_SHORT,
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
SEED = 42

# Capture times (before corridor is blocked, at blockage, aftermath)
T_EARLY = 100
T_BLOCK = 212
T_LATE = 292

# Distinct line styles per planner
PLANNER_STYLES = {
    "astar": {"linestyle": "-", "marker": "o", "markevery": 50, "markersize": 3},
    "periodic_replan": {"linestyle": (0, (8, 3))},  # long dashes
    "aggressive_replan": {"linestyle": "-"},
    "incremental_astar": {"linestyle": "-."},
    "apf": {"linestyle": ":"},
}


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def main() -> None:
    cfg = load_scenario(SCENARIO)
    capture_times = {T_EARLY, T_BLOCK, T_LATE}

    # Step 1: Run all 5 planners, collect trajectories + metrics
    trajectories = {}
    all_metrics = {}
    print("Running planners...")
    for pid in PLANNER_ORDER:
        print(f"  {pid}...", end="", flush=True)
        r = run_episode(SCENARIO, pid, SEED)
        trajectories[pid] = r.trajectory
        all_metrics[pid] = r.metrics
        status = "OK" if r.metrics.get("success") else r.metrics.get("termination_reason", "?")
        print(f" {status} ({len(r.trajectory)} steps)")

    # Step 2: Capture map state at 3 moments using one planner's frame_callback
    frames = {}

    def cb(heightmap, state, dyn_state, cfg_arg):
        step = state.get("step_idx", 0)
        if step in capture_times and step not in frames:
            frames[step] = {
                "fire": dyn_state.get(
                    "fire_mask", np.zeros_like(heightmap, dtype=bool)
                ).copy(),
                "smoke": dyn_state.get(
                    "smoke_mask", np.zeros_like(heightmap, dtype=np.float32)
                ).copy(),
                "debris": dyn_state.get(
                    "debris_mask", np.zeros_like(heightmap, dtype=bool)
                ).copy(),
                "heightmap": heightmap.copy(),
                "start_xy": state.get("start_xy"),
                "goal_xy": state.get("goal_xy"),
            }

    print("Capturing map frames...")
    run_episode(SCENARIO, "aggressive_replan", SEED, frame_callback=cb)

    if not frames:
        print("ERROR: No frames captured.")
        sys.exit(1)

    # Use available frames (some capture times may not exist)
    available_times = sorted(frames.keys())
    if len(available_times) < 3:
        print(f"WARNING: Only captured {len(available_times)} frames: {available_times}")

    panel_times = []
    panel_titles = []
    if T_EARLY in frames:
        panel_times.append(T_EARLY)
        panel_titles.append(f"t={T_EARLY}: Fire approaching")
    if T_BLOCK in frames:
        panel_times.append(T_BLOCK)
        panel_titles.append(f"t={T_BLOCK}: Corridor blocked")
    if T_LATE in frames:
        panel_times.append(T_LATE)
        panel_titles.append(f"t={T_LATE}: Aftermath")

    # Fallback: use whatever frames we have
    if len(panel_times) < 3:
        for t in available_times:
            if t not in panel_times:
                panel_times.append(t)
                panel_titles.append(f"t={t}")
            if len(panel_times) >= 3:
                break

    n_panels = min(len(panel_times), 3)

    # Step 3: Render
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 3.5))
    if n_panels == 1:
        axes = [axes]

    # Determine crop bounds from fire at blockage time
    crop_bounds = None
    if T_BLOCK in frames:
        fire_b = frames[T_BLOCK]["fire"]
        if fire_b.any():
            fy, fx = np.where(fire_b)
            cx_fire, cy_fire = int(fx.mean()), int(fy.mean())
            hm_shape = frames[T_BLOCK]["heightmap"].shape
            pad = 120
            crop_bounds = (
                max(0, cx_fire - pad),
                min(hm_shape[1], cx_fire + pad),
                max(0, cy_fire - pad),
                min(hm_shape[0], cy_fire + pad),
            )

    for idx, (ax, t, title) in enumerate(zip(axes, panel_times, panel_titles)):
        f = frames[t]
        hm = f["heightmap"]

        # Base image: light ground, dark buildings
        base = np.full((*hm.shape, 3), 240, dtype=np.uint8)
        base[hm > 0] = [80, 80, 80]

        # Fire overlay
        fire = f["fire"]
        if fire.any():
            base[fire] = [255, 60, 20]

        # Smoke overlay (semi-transparent blend)
        smoke = f["smoke"]
        smoke_mask = smoke >= 0.3
        if smoke_mask.any():
            smoke_color = np.array([160, 160, 160], dtype=np.float64)
            base[smoke_mask] = (
                base[smoke_mask].astype(np.float64) * 0.6 + smoke_color * 0.4
            ).astype(np.uint8)

        # Debris overlay
        debris = f["debris"]
        if debris.any():
            base[debris] = [139, 90, 43]  # brown

        ax.imshow(base, origin="upper")

        # Overlay trajectories up to this timestep
        for pid in PLANNER_ORDER:
            traj = trajectories[pid]
            traj_t = traj[: min(t, len(traj))]
            if len(traj_t) < 2:
                continue
            xs, ys = zip(*traj_t)
            success = all_metrics[pid].get("success", False)

            # Line style from PLANNER_STYLES
            style = PLANNER_STYLES.get(pid, {})
            ls = style.get("linestyle", "-")
            marker = style.get("marker", None)
            markevery = style.get("markevery", None)
            ms = style.get("markersize", 3)

            plot_kwargs = dict(
                color=PLANNER_COLORS[pid],
                linewidth=2.5,
                linestyle=ls,
                alpha=0.9 if success else 0.4,
                label=PLANNER_SHORT[pid] if idx == 0 else None,
                path_effects=[
                    pe.Stroke(linewidth=3.5, foreground="white", alpha=0.6),
                    pe.Normal(),
                ],
            )
            if marker:
                plot_kwargs["marker"] = marker
                plot_kwargs["markevery"] = markevery
                plot_kwargs["markersize"] = ms
                plot_kwargs["markeredgecolor"] = PLANNER_COLORS[pid]

            ax.plot(xs, ys, **plot_kwargs)

            # Endpoint markers
            if t >= len(traj):
                ex, ey = traj[-1]
                if success:
                    ax.plot(ex, ey, "*", color="gold", markersize=8,
                            markeredgecolor="k", markeredgewidth=0.3, zorder=10)
                else:
                    ax.plot(ex, ey, "X", color="red", markersize=8,
                            markeredgecolor="k", markeredgewidth=0.3, zorder=10)

        # Mark start and goal
        start = f.get("start_xy")
        goal = f.get("goal_xy")
        if start:
            ax.plot(start[0], start[1], "^", color="lime", markersize=6,
                    markeredgecolor="k", markeredgewidth=0.5, zorder=10)
        if goal:
            ax.plot(goal[0], goal[1], "*", color="gold", markersize=8,
                    markeredgecolor="k", markeredgewidth=0.5, zorder=10)

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

        # Apply crop
        if crop_bounds:
            x0, x1, y0, y1 = crop_bounds
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # inverted y for image coordinates

    # Red starburst annotation at blockage point on middle panel
    if n_panels >= 2:
        mid_ax = axes[1]
        if T_BLOCK in frames:
            fire_b = frames[T_BLOCK]["fire"]
            if fire_b.any():
                fy, fx = np.where(fire_b)
                cx, cy = int(fx.mean()), int(fy.mean())
                mid_ax.plot(cx, cy, "*", color="red", markersize=14,
                            markeredgecolor="yellow", markeredgewidth=0.8,
                            zorder=15)
                mid_ax.annotate(
                    "BLOCKED", (cx, cy),
                    textcoords="offset points", xytext=(15, -15),
                    fontsize=10, fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                    bbox=dict(facecolor="white", alpha=0.8, pad=2,
                              edgecolor="red", linewidth=0.5),
                )

    # Legend on first panel
    if n_panels > 0:
        axes[0].legend(fontsize=7, loc="lower left", framealpha=0.8)

    fig.suptitle(
        "Fire cuts the corridor: who adapts? who dies?",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()

    _save(fig, "corridor_blockage_dramatic")

    # Print summary
    print("\nPlanner outcomes:")
    for pid in PLANNER_ORDER:
        m = all_metrics[pid]
        print(f"  {PLANNER_SHORT[pid]:12s}: "
              f"{'SUCCESS' if m.get('success') else m.get('termination_reason', '?'):12s} "
              f"steps={m.get('executed_steps_len', '?')}")


if __name__ == "__main__":
    main()
