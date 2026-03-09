#!/usr/bin/env python3
"""Generate paper-quality visualization snapshots.

For each of the 3 scenario families:
  - Representative episode snapshots at t=0, t=mid, t=end
For one scenario:
  - 5-panel figure showing all planner trajectories on the same map

Uses run_episode() + frame_callback for consistency with the benchmark runner.

Outputs 300dpi PNG + vector PDF to outputs/paper_figures/.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uavbench.benchmark.runner import run_episode
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer
from uavbench.visualization.labels import (
    PLANNER_ORDER, PLANNER_LABELS, PLANNER_COLORS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_DIR = "outputs/paper_figures"
SEED = 42

# One representative scenario per family
FAMILY_SCENARIOS = {
    "pharma_delivery": "osm_penteli_pharma_delivery_medium",
    "flood_rescue": "osm_piraeus_flood_rescue_medium",
    "fire_surveillance": "osm_downtown_fire_surveillance_medium",
}

# For the 5-panel planner comparison
COMPARISON_SCENARIO = "osm_penteli_pharma_delivery_medium"


# ---------------------------------------------------------------------------
# Frame capture via run_episode callback
# ---------------------------------------------------------------------------


def _run_with_snapshots(
    scenario_id: str,
    planner_id: str,
    seed: int,
    capture_steps: set[int] | None = None,
) -> dict[str, Any]:
    """Run episode via run_episode() and capture frames at specified steps.

    If capture_steps is None, captures t=0 and final frame only.
    """
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="paper_min")

    frames: dict[int, tuple[np.ndarray, dict]] = {}
    total_steps_box: list[int] = [0]

    # For final-frame-only capture: keep last frame data without
    # accumulating all frames in memory (7500x7500 frames = ~168MB each).
    last_frame_data: list[tuple | None] = [None]

    def frame_callback(
        heightmap: np.ndarray,
        state: dict,
        dyn_state: dict,
        cfg: object,
    ) -> None:
        step = state.get("step_idx", 0)
        total_steps_box[0] = step

        if capture_steps is not None and step in capture_steps:
            frame, meta = renderer.render_frame(heightmap, state, dyn_state)
            frames[step] = (frame, meta)
        elif capture_steps is None:
            # Keep reference to latest args for final-frame rendering
            last_frame_data[0] = (heightmap, state.copy(), dyn_state.copy())

    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)

    # If capture_steps is None, render the final frame now
    if capture_steps is None and last_frame_data[0] is not None:
        hm, st, ds = last_frame_data[0]
        frame, meta = renderer.render_frame(hm, st, ds)
        frames[total_steps_box[0]] = (frame, meta)

    return {
        "frames": frames,
        "metrics": result.metrics,
        "total_steps": total_steps_box[0],
    }


# ---------------------------------------------------------------------------
# Snapshot generators
# ---------------------------------------------------------------------------


def generate_fire_evolution() -> None:
    """Fig 1: 4-panel fire spread progression on Penteli scenario.

    Shows t=0, t=50, t=200, t=1000 to illustrate fire dynamics.
    """
    print("Generating fire evolution 4-panel...")

    scenario_id = "osm_penteli_pharma_delivery_medium"
    target_steps = {1, 50, 200, 1000}
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="paper_min")

    captured: dict[int, np.ndarray] = {}

    def cb(heightmap, state, dyn_state, cfg):
        step = state.get("step_idx", 0)
        if step in target_steps:
            frame, _ = renderer.render_frame(heightmap, state, dyn_state)
            captured[step] = frame

    run_episode(scenario_id, "astar", SEED, frame_callback=cb)

    # Build 4-panel figure
    panels = [(1, "(a) t=1"), (50, "(b) t=50"), (200, "(c) t=200"), (1000, "(d) t=1000")]
    available_steps = sorted(captured.keys())

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2))
    for ax, (step, label) in zip(axes, panels):
        if step in captured:
            frame = captured[step]
        else:
            # Use closest available
            closest = min(available_steps, key=lambda k: abs(k - step))
            frame = captured[closest]
            label += f" (actual={closest})"
        ax.imshow(frame)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    _save_fig(fig, "fire_evolution_4panel")


def generate_scenario_overview() -> None:
    """Fig 2: 3-panel basemap overview — one per scenario family.

    Shows clean maps (no fire, no HUD, no dynamics) with start/goal markers.
    Purpose: illustrate building density and geography differences.
    """
    from uavbench.visualization.overlays import draw_start, draw_goal

    print("Generating scenario overview (basemap-only)...")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    family_items = list(FAMILY_SCENARIOS.items())

    for ax, (family, scenario_id) in zip(axes, family_items):
        config = load_scenario(scenario_id)
        renderer = Renderer(config, mode="paper_min")

        # Run 1 step just to get heightmap, start, goal, landuse, roads
        captured = {}

        def cb(heightmap, state, dyn_state, cfg):
            if not captured:
                captured["heightmap"] = heightmap
                captured["state"] = state.copy()

        run_episode(scenario_id, "astar", SEED, frame_callback=cb)

        heightmap = captured["heightmap"]
        state = captured["state"]
        H, W = heightmap.shape
        cell = renderer._cell_px

        # Render basemap only (no dynamics)
        frame = renderer._render_basemap(
            heightmap, H, W, cell,
            state.get("landuse_map"),
            state.get("roads_mask"),
        )

        # Add start/goal markers
        start_xy = state.get("start_xy", state.get("agent_xy", (0, 0)))
        goal_xy = state.get("goal_xy", (H - 1, W - 1))
        draw_start(frame, start_xy, cell)
        draw_goal(frame, goal_xy, cell)

        ax.imshow(frame)
        label = family.replace("_", " ").title()
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    _save_fig(fig, "scenario_overview_3panel")


def generate_family_snapshots() -> None:
    """For each scenario family: snapshots at t=0, t=mid, t=end."""
    print("Generating family snapshots...")

    for family, scenario_id in FAMILY_SCENARIOS.items():
        print(f"\n  Family: {family} ({scenario_id})")

        # First pass: find total steps
        first = _run_with_snapshots(scenario_id, "astar", SEED, capture_steps=set())
        total = first["total_steps"]
        mid = max(1, total // 2)
        print(f"    Total steps: {total}, mid: {mid}")

        # Second pass: capture t=1 (earliest), t=mid, t=end
        capture = {1, mid, total}
        result = _run_with_snapshots(scenario_id, "astar", SEED, capture_steps=capture)

        for label, step in [("t0", 1), ("mid", mid), ("end", total)]:
            if step in result["frames"]:
                frame, meta = result["frames"][step]
            else:
                closest = min(result["frames"].keys(), key=lambda k: abs(k - step))
                frame, meta = result["frames"][closest]
                print(f"    Warning: step {step} not captured, using step {closest}")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(frame)
            ax.set_title(
                f"{family.replace('_', ' ').title()} — {label} (step {step})",
                fontsize=11, fontweight="bold",
            )
            ax.axis("off")
            fig.tight_layout()

            name = f"snapshot_{family}_{label}"
            _save_fig(fig, name)


def generate_planner_comparison() -> None:
    """5-panel figure: all planner trajectories on same map for one scenario."""
    print(f"\nGenerating 5-panel planner comparison ({COMPARISON_SCENARIO})...")

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    axes_flat = axes.flatten()

    for ax, pid in zip(axes_flat, PLANNER_ORDER):
        print(f"  Running {pid}...")

        # Capture only final frame
        result = _run_with_snapshots(
            COMPARISON_SCENARIO, pid, SEED, capture_steps=None,
        )

        # Get the last captured frame
        if result["frames"]:
            final_step = max(result["frames"].keys())
            frame, meta = result["frames"][final_step]
        else:
            # Fallback: run again capturing everything
            result = _run_with_snapshots(
                COMPARISON_SCENARIO, pid, SEED,
                capture_steps={result["total_steps"]},
            )
            final_step = max(result["frames"].keys())
            frame, meta = result["frames"][final_step]

        ax.imshow(frame)
        m = result["metrics"]
        tr = m.get("termination_reason", "?")
        steps = m.get("executed_steps_len", 0)
        success = m.get("success", False)
        status = "OK" if success else tr
        ax.set_title(
            f"{PLANNER_LABELS[pid]}  [{status}, {steps} steps]",
            fontsize=11, fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        f"Planner Comparison — {COMPARISON_SCENARIO.replace('osm_', '').replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "planner_comparison_5panel")


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both 300dpi PNG and vector PDF."""
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {name}.png + .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate UAVBench v2 paper snapshots.")
    p.add_argument(
        "--scenario", type=str, default=None,
        help="Single scenario ID for a quick snapshot (skips full family/comparison)",
    )
    p.add_argument(
        "--planner", type=str, default=None,
        help="Single planner ID (used with --scenario, default: astar)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (default: 42)",
    )
    p.add_argument(
        "--skip-families", action="store_true",
        help="Skip family snapshot generation",
    )
    p.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip 5-panel planner comparison generation",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output directory for figures (default: outputs/paper_figures)",
    )
    return p.parse_args()


def main() -> None:
    global FIG_DIR, SEED  # noqa: PLW0603
    args = _parse_args()

    if args.output:
        FIG_DIR = args.output
    if args.seed is not None:
        SEED = args.seed

    os.makedirs(FIG_DIR, exist_ok=True)
    print("UAVBench v2 Paper Snapshots")
    print(f"  Output: {FIG_DIR}/")
    print()

    if args.scenario:
        planner_id = args.planner or "astar"
        print(f"Quick snapshot: {args.scenario} / {planner_id} / seed={SEED}")

        first = _run_with_snapshots(args.scenario, planner_id, SEED, capture_steps=set())
        total = first["total_steps"]
        mid = max(1, total // 2)
        capture = {1, mid, total}
        result = _run_with_snapshots(args.scenario, planner_id, SEED, capture_steps=capture)

        for label, step in [("t0", 1), ("mid", mid), ("end", total)]:
            if step in result["frames"]:
                frame, meta = result["frames"][step]
            else:
                closest = min(result["frames"].keys(), key=lambda k: abs(k - step))
                frame, meta = result["frames"][closest]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(frame)
            ax.set_title(
                f"{args.scenario} / {planner_id} — {label} (step {step})",
                fontsize=11, fontweight="bold",
            )
            ax.axis("off")
            fig.tight_layout()
            name = f"snapshot_{args.scenario}_{planner_id}_{label}"
            _save_fig(fig, name)
        print(f"\nDone. {len(capture)} snapshots in {FIG_DIR}/")
        return

    # Fire evolution (4 panels: t=0, t=50, t=200, t=1000)
    generate_fire_evolution()

    # Scenario overview (basemap-only, no dynamics)
    generate_scenario_overview()

    if not args.skip_families:
        generate_family_snapshots()
    if not args.skip_comparison:
        generate_planner_comparison()

    print(f"\nDone. All snapshots in {FIG_DIR}/")


if __name__ == "__main__":
    main()
