#!/usr/bin/env python3
"""Collapse cascade figure (FIXED).

Two-row figure:
Top = timeline with fire+debris counts + TWO agent distance lines (Aggressive + Incr. A*)
Bottom = 4 rendered map snapshots at key moments using Renderer

Output: outputs/paper_figures/collapse_cascade.{png,pdf}
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.renderer import Renderer
from flare.visualization.overlays import (
    draw_fire, draw_smoke, draw_debris, draw_traffic, draw_vehicle_icons,
)
from flare.visualization.labels import PLANNER_COLORS, PLANNER_SHORT

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
PLANNERS_TO_PLOT = ["aggressive_replan", "incremental_astar"]
SEED = 42

C_FIRE = "#D55E00"
C_DEBRIS = "#8B5A2B"

# Target snapshot steps: t=0, t≈80 (fire hits building), t≈collapse, t≈150+ (debris permanent)
SNAPSHOT_TARGETS = [0, 80, None, 150]  # None = auto-detect first collapse


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def _run_with_collection(planner: str, scenario: str, seed: int, cfg) -> dict[str, Any]:
    """Run episode and collect per-step data + snapshot frames."""
    steps_list: list[int] = []
    fire_counts: list[int] = []
    debris_counts: list[int] = []
    distances: list[float] = []
    collapse_events: list[int] = []
    snapshots: dict[int, dict[str, Any]] = {}
    prev_debris = 0

    def cb(heightmap, state, dyn_state, cfg_arg):
        nonlocal prev_debris

        step = state.get("step_idx", 0)
        agent = state.get("agent_xy", (0, 0))
        goal = state.get("goal_xy", (0, 0))
        dist = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])

        fire_mask = dyn_state.get("fire_mask")
        fc = int(fire_mask.sum()) if fire_mask is not None else 0

        debris_mask = dyn_state.get("debris_mask")
        dc = int(debris_mask.sum()) if debris_mask is not None else 0

        if dc > prev_debris:
            collapse_events.append(step)
        prev_debris = dc

        steps_list.append(step)
        fire_counts.append(fc)
        debris_counts.append(dc)
        distances.append(dist)

        # Store snapshots for rendering
        # Always keep step 0, step near 80, first collapse, and a late step
        should_snap = (
            step == 0
            or step == 80
            or (dc > 0 and not any(s in snapshots for s in collapse_events))
            or step == 150
            or step % 50 == 0  # periodic backups
        )
        if should_snap:
            snapshots[step] = {
                "heightmap": heightmap.copy(),
                "state": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()},
                "dyn_state": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in dyn_state.items()},
            }

    print(f"  Running {planner} with collapse enabled...")
    result = run_episode(scenario, planner, seed, frame_callback=cb, config_override=cfg)
    success = result.metrics.get("success", False)
    term = result.metrics.get("termination_reason", "")
    print(f"  Success: {success}, Steps: {len(steps_list)}, Term: {term}")
    print(f"  Collapse events: {len(collapse_events)}")

    return {
        "steps": steps_list,
        "fire_counts": fire_counts,
        "debris_counts": debris_counts,
        "distances": distances,
        "collapse_events": collapse_events,
        "snapshots": snapshots,
        "success": success,
        "planner": planner,
    }


def _render_snapshot(renderer, cell, snap_data):
    """Render a snapshot using Renderer basemap + overlays."""
    heightmap = snap_data["heightmap"]
    state = snap_data["state"]
    dyn_state = snap_data["dyn_state"]
    H, W = heightmap.shape

    bg = renderer._render_basemap(
        heightmap, H, W, cell,
        state.get("landuse_map"),
        state.get("roads_mask"),
    )

    smoke_mask = dyn_state.get("smoke_mask")
    if smoke_mask is not None:
        draw_smoke(bg, smoke_mask, cell, alpha_256=40)

    fire_mask = dyn_state.get("fire_mask")
    if fire_mask is not None:
        draw_fire(bg, fire_mask, cell)

    debris_mask = dyn_state.get("debris_mask")
    if debris_mask is not None:
        draw_debris(bg, debris_mask, cell)

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

    return bg


def main() -> None:
    cfg = replace(
        load_scenario(SCENARIO),
        enable_collapse=True,
        collapse_delay=80,
        debris_prob=0.6,
    )

    renderer = Renderer(cfg, mode="paper_min")
    renderer._cell_px = 1
    cell = renderer._cell_px

    # Run both planners
    all_data = {}
    for pid in PLANNERS_TO_PLOT:
        all_data[pid] = _run_with_collection(pid, SCENARIO, SEED, cfg)

    # Use the longer-running planner for hazard timeline
    primary = max(all_data.values(), key=lambda d: len(d["steps"]))

    steps_list = primary["steps"]
    fire_counts = primary["fire_counts"]
    debris_counts = primary["debris_counts"]
    collapse_events = primary["collapse_events"]
    snapshots = primary["snapshots"]

    # --- Select 4 snapshot steps ---
    snap_steps = [0]  # (b) Before fire

    # (c) Fire approaching: just before first collapse or at t=80
    if collapse_events:
        pre_collapse = max(0, collapse_events[0] - 5)
    else:
        pre_collapse = 80
    snap_steps.append(pre_collapse)

    # (d) First collapse
    if collapse_events:
        snap_steps.append(collapse_events[0])
    else:
        snap_steps.append(min(120, max(snapshots.keys())))

    # (e) Debris permanent: latest available step ≥ 150
    late_steps = [s for s in sorted(snapshots.keys()) if s >= 150]
    if late_steps:
        snap_steps.append(late_steps[-1])
    else:
        snap_steps.append(max(snapshots.keys()))

    # Find closest available snapshots
    def _closest_snap(target):
        available = sorted(snapshots.keys())
        if not available:
            return None
        return min(available, key=lambda s: abs(s - target))

    # --- Figure ---
    fig = plt.figure(figsize=(7.0, 4.5))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.2, 1], hspace=0.35, wspace=0.15)

    # === TOP ROW: Timeline ===
    ax_timeline = fig.add_subplot(gs[0, :])

    steps_arr = np.array(steps_list)
    fire_arr = np.array(fire_counts)
    debris_arr = np.array(debris_counts)

    # Left y-axis: fire + debris areas
    ax_timeline.fill_between(steps_arr, fire_arr, alpha=0.2, color=C_FIRE)
    ln1 = ax_timeline.plot(steps_arr, fire_arr, color=C_FIRE, lw=1.2, label="Fire cells")
    ax_timeline.fill_between(steps_arr, debris_arr, alpha=0.25, color=C_DEBRIS)
    ln2 = ax_timeline.plot(steps_arr, debris_arr, color=C_DEBRIS, lw=1.2, label="Debris cells")
    ax_timeline.set_ylabel("Cell count", fontsize=8)
    ax_timeline.set_xlabel("Step", fontsize=8)

    # Right y-axis: agent distances for BOTH planners
    ax_dist = ax_timeline.twinx()
    lns_dist = []
    for pid in PLANNERS_TO_PLOT:
        d = all_data[pid]
        if not d["distances"]:
            continue
        max_dist = max(d["distances"]) if max(d["distances"]) > 0 else 1
        norm_dist = [dd / max_dist for dd in d["distances"]]
        steps_p = np.array(d["steps"])
        ln = ax_dist.plot(steps_p, norm_dist, color=PLANNER_COLORS[pid],
                          lw=1.0, alpha=0.7,
                          label=f"{PLANNER_SHORT[pid]} dist.")
        lns_dist.extend(ln)

        # Mark failure
        if not d["success"]:
            ax_dist.plot(steps_p[-1], norm_dist[-1], "X", color="red",
                         markersize=5, zorder=10)

    ax_dist.set_ylabel("Agent dist. (norm.)", fontsize=7)
    ax_dist.tick_params(axis="y", labelsize=6)

    # Combined legend
    lns = ln1 + ln2 + lns_dist
    labs = [l.get_label() for l in lns]
    ax_timeline.legend(lns, labs, fontsize=5.5, loc="upper left")

    # Collapse event lines
    for i, cs in enumerate(collapse_events[:5]):
        ax_timeline.axvline(cs, color="grey", ls=":", lw=0.5, alpha=0.4)
        if i == 0:
            fire_at_cs = fire_arr[steps_list.index(cs)] if cs in steps_list else 0
            ax_timeline.annotate(
                "Collapse!", xy=(cs, fire_at_cs),
                xytext=(cs + 20, max(fire_arr) * 0.8 if len(fire_arr) > 0 else 10),
                fontsize=6, color="black", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )

    ax_timeline.set_title("(a) Hazard cascade timeline", fontsize=8,
                          fontweight="bold", loc="left")
    ax_timeline.spines["top"].set_visible(False)
    ax_timeline.grid(axis="x", alpha=0.12)

    # === BOTTOM ROW: 4 rendered map snapshots ===
    snap_labels = [
        "(b) t=0: Before fire",
        "(c) Fire approaching",
        "(d) First collapse",
        "(e) Debris permanent",
    ]

    for col in range(4):
        ax = fig.add_subplot(gs[1, col])
        target = snap_steps[col]
        actual = _closest_snap(target)

        if actual is not None and actual in snapshots:
            bg = _render_snapshot(renderer, cell, snapshots[actual])
            H_snap = snapshots[actual]["heightmap"].shape[0]
            W_snap = snapshots[actual]["heightmap"].shape[1]
            ax.imshow(bg, origin="upper", interpolation="nearest",
                      aspect="equal", extent=[0, W_snap, H_snap, 0])

            # Drone as blue X with white outline
            agent_xy = snapshots[actual]["state"].get("agent_xy")
            if agent_xy:
                # White outline
                ax.plot(agent_xy[0], agent_xy[1], "x", color="white",
                        markersize=6, markeredgewidth=1.5, zorder=8)
                # Blue cross (top)
                ax.plot(agent_xy[0], agent_xy[1], "x", color="#4090D0",
                        markersize=5, markeredgewidth=1.0, zorder=9)

            # Step label
            ax.text(0.02, 0.98, f"t={actual}", transform=ax.transAxes,
                    fontsize=5, color="white", va="top",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none",
                              boxstyle="round,pad=0.2"))
        else:
            ax.set_facecolor("#f0f0f0")

        ax.set_title(snap_labels[col], fontsize=7, fontweight="normal", color="black")
        ax.axis("off")

    fig.suptitle(
        r"Fire $\rightarrow$ 80 steps $\rightarrow$ collapse $\rightarrow$ permanent debris",
        fontsize=8, fontweight="normal", color="black",
    )

    _save(fig, "collapse_cascade")


if __name__ == "__main__":
    main()
