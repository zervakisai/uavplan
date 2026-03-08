#!/usr/bin/env python3
"""Generate IEEE-format timeline figure + critical moment snapshots.

Two outputs:

1. **Timeline figure**: Horizontal bar per planner showing episode phases
   (planning, moving, replanning, stuck, success/fail) with fire-area
   overlay and key event markers.

2. **Critical moment snapshots**: Map excerpts at pivotal moments
   (first replan, fire blockage, path divergence, goal approach)
   with annotated callouts.

Both outputs are 300 DPI PNG + vector PDF, IEEE column-width compatible.

Usage:
    python scripts/gen_timeline_figure.py [--scenario ...] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/paper_figures"

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
PLANNER_COLORS = {
    "astar": "#e74c3c",
    "periodic_replan": "#3498db",
    "aggressive_replan": "#2ecc71",
    "incremental_astar": "#f39c12",
    "apf": "#9b59b6",
}

DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"
DEFAULT_SEED = 42

# IEEE column width (inches)
IEEE_COL_WIDTH = 3.5
IEEE_TWO_COL_WIDTH = 7.16


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _collect_episode_data(
    scenario_id: str,
    planner_id: str,
    seed: int,
) -> dict[str, Any]:
    """Run episode and collect per-step telemetry for timeline."""
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="paper_min")

    steps: list[int] = []
    distances: list[float] = []
    fire_areas: list[int] = []
    replan_steps: list[int] = []
    blocked_steps: list[int] = []
    snapshots: dict[str, tuple[np.ndarray, dict]] = {}

    first_replan_captured = False
    last_frame = None
    last_heightmap = None

    def frame_callback(heightmap, state, dyn_state, cfg):
        nonlocal first_replan_captured, last_frame, last_heightmap

        step = state.get("step_idx", 0)
        agent = state.get("agent_xy", (0, 0))
        goal = state.get("goal_xy", (0, 0))
        dist = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])

        fire_mask = dyn_state.get("fire_mask")
        fire_area = int(fire_mask.sum()) if fire_mask is not None else 0

        steps.append(step)
        distances.append(dist)
        fire_areas.append(fire_area)

        # Detect replans
        reason = state.get("plan_reason", "")
        if reason and reason not in ("", "cooldown", "no_change", "path_clear", "calibration",
                                      "not_periodic", "naive_skip"):
            replan_steps.append(step)

        # Capture critical snapshots
        frame, _meta = renderer.render_frame(heightmap, state, dyn_state)
        last_frame = frame
        last_heightmap = heightmap

        # First frame
        if step == 1:
            snapshots["start"] = (frame.copy(), {"step": step, "dist": dist})

        # First replan
        if replan_steps and not first_replan_captured:
            snapshots["first_replan"] = (frame.copy(), {
                "step": step, "dist": dist, "reason": reason,
            })
            first_replan_captured = True

        # Mid-episode (when distance is at minimum before possible increase)
        if step > 0 and len(distances) > 2:
            if distances[-1] > distances[-2] and "turning_point" not in snapshots:
                snapshots["turning_point"] = (frame.copy(), {
                    "step": step, "dist": dist,
                })

    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)

    # Final frame
    if last_frame is not None:
        snapshots["end"] = (last_frame.copy(), {
            "step": steps[-1] if steps else 0,
            "success": result.metrics.get("success", False),
        })

    return {
        "steps": steps,
        "distances": distances,
        "fire_areas": fire_areas,
        "replan_steps": replan_steps,
        "snapshots": snapshots,
        "metrics": result.metrics,
    }


# ---------------------------------------------------------------------------
# Timeline figure
# ---------------------------------------------------------------------------


def generate_timeline_figure(
    all_data: dict[str, dict],
    output_dir: str,
) -> None:
    """Create horizontal timeline bar chart for all planners."""
    fig, axes = plt.subplots(
        len(PLANNER_ORDER) + 1, 1,
        figsize=(IEEE_TWO_COL_WIDTH, 0.8 * (len(PLANNER_ORDER) + 1.5)),
        gridspec_kw={"height_ratios": [1] * len(PLANNER_ORDER) + [0.8]},
        sharex=True,
    )

    max_steps = max(
        len(d["steps"]) for d in all_data.values()
    )

    for i, pid in enumerate(PLANNER_ORDER):
        ax = axes[i]
        data = all_data[pid]
        steps = data["steps"]
        dists = data["distances"]
        color = PLANNER_COLORS[pid]
        m = data["metrics"]
        success = m.get("success", False)

        # Distance to goal as filled area
        if steps:
            max_dist = max(dists) if dists else 1
            norm_dists = [d / max(max_dist, 1) for d in dists]
            ax.fill_between(steps, norm_dists, alpha=0.3, color=color)
            ax.plot(steps, norm_dists, color=color, linewidth=0.8)

        # Mark replans
        for rs in data["replan_steps"]:
            ax.axvline(x=rs, color=color, linewidth=0.3, alpha=0.5, linestyle=":")

        # Success/fail marker
        if steps:
            marker = "★" if success else "✗"
            marker_color = "green" if success else "red"
            ax.annotate(
                marker, xy=(steps[-1], 0), fontsize=8,
                color=marker_color, fontweight="bold",
                ha="center", va="bottom",
            )

        ax.set_ylabel(PLANNER_LABELS[pid], fontsize=7, rotation=0, labelpad=45, va="center")
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)

        # Stats annotation
        n_replans = len(data["replan_steps"])
        n_steps = m.get("executed_steps_len", 0)
        ax.annotate(
            f"{n_steps}s, {n_replans}r",
            xy=(0.98, 0.85), xycoords="axes fraction",
            fontsize=5, ha="right", va="top", color="gray",
        )

    # Bottom panel: fire area
    ax_fire = axes[-1]
    # Use first planner's fire data (shared environment)
    first_data = next(iter(all_data.values()))
    if first_data["fire_areas"]:
        ax_fire.fill_between(
            first_data["steps"], first_data["fire_areas"],
            alpha=0.3, color="#e74c3c",
        )
        ax_fire.plot(
            first_data["steps"], first_data["fire_areas"],
            color="#e74c3c", linewidth=0.8,
        )
    ax_fire.set_ylabel("Fire\ncells", fontsize=7, rotation=0, labelpad=35, va="center")
    ax_fire.set_xlabel("Step", fontsize=8)
    ax_fire.tick_params(labelsize=6)

    fig.suptitle("Episode Timeline — Planner Behavior Comparison",
                 fontsize=9, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    _save_fig(fig, "timeline_comparison", output_dir)


# ---------------------------------------------------------------------------
# Critical moment snapshots
# ---------------------------------------------------------------------------


def generate_critical_snapshots(
    all_data: dict[str, dict],
    output_dir: str,
) -> None:
    """Extract and annotate critical moment snapshots from episodes."""
    # Collect snapshot moments across planners
    moments = ["start", "first_replan", "turning_point", "end"]
    moment_labels = {
        "start": "t=0 (Initial)",
        "first_replan": "First Replan",
        "turning_point": "Path Divergence",
        "end": "Terminal State",
    }

    # Pick one representative planner for snapshots (aggressive_replan — most dynamic)
    rep_planner = "aggressive_replan"
    if rep_planner not in all_data:
        rep_planner = next(iter(all_data.keys()))

    data = all_data[rep_planner]
    snapshots = data["snapshots"]

    available = [m for m in moments if m in snapshots]
    if not available:
        print("  No snapshots captured — skipping critical moments figure.")
        return

    n_cols = min(len(available), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(IEEE_TWO_COL_WIDTH, 2.5))
    if n_cols == 1:
        axes = [axes]

    for ax, moment in zip(axes, available):
        frame, meta = snapshots[moment]
        ax.imshow(frame)
        ax.set_title(
            moment_labels.get(moment, moment),
            fontsize=8, fontweight="bold",
        )

        # Annotate with metadata
        info_lines = []
        if "step" in meta:
            info_lines.append(f"step={meta['step']}")
        if "dist" in meta:
            info_lines.append(f"d={meta['dist']:.0f}")
        if "reason" in meta:
            info_lines.append(meta["reason"])
        if "success" in meta:
            info_lines.append("SUCCESS" if meta["success"] else "FAIL")

        if info_lines:
            ax.annotate(
                "\n".join(info_lines),
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                va="bottom",
            )
        ax.axis("off")

    fig.suptitle(
        f"Critical Moments — {PLANNER_LABELS[rep_planner]}",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "critical_snapshots", output_dir)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, name: str, output_dir: str) -> None:
    """Save figure as 300dpi PNG + vector PDF."""
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {name}.png + .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate timeline + snapshot figures.")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    p.add_argument("--skip-snapshots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=== Timeline Figure + Critical Snapshots ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Seed:     {args.seed}")
    print()

    # Collect data for all planners
    all_data: dict[str, dict] = {}
    for pid in PLANNER_ORDER:
        print(f"  Running {pid}...", end="", flush=True)
        t0 = time.perf_counter()
        data = _collect_episode_data(args.scenario, pid, args.seed)
        elapsed = time.perf_counter() - t0
        all_data[pid] = data
        m = data["metrics"]
        status = "OK" if m.get("success") else m.get("termination_reason", "?")
        replans = len(data["replan_steps"])
        print(f" [{status}] {m.get('executed_steps_len', 0)}steps, {replans}replans ({elapsed:.1f}s)")

    # Generate timeline
    print("\n  Generating timeline figure...")
    generate_timeline_figure(all_data, args.output)

    # Generate critical snapshots
    if not args.skip_snapshots:
        print("  Generating critical snapshots...")
        generate_critical_snapshots(all_data, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
