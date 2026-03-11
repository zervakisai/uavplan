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
    """Fig 1: 4-panel fire spread at event-driven timestamps on Penteli.

    Event-driven timestamps (seed 42, A* planner):
    - t=1: corridor fires just ignited
    - t=210: fire narrows corridor—last passable moment
    - t=212: corridor blocked—A*'s first fire rejection
    - t=520: Incremental A* reaches goal; A* still blocked
    """
    print("Generating fire evolution 4-panel (event-driven)...")

    scenario_id = "osm_penteli_pharma_delivery_medium"
    target_steps = {1, 210, 212, 520}
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
    panels = [(1, "(a) t=1"), (210, "(b) t=210"), (212, "(c) t=212"), (520, "(d) t=520")]
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
    """Fig 2: 3-panel overview — one per scenario family.

    Shows basemap + dynamics (fire, smoke, traffic, debris) + POI icons +
    start/goal markers at t=OVERVIEW_T.  No trajectory, agent, or HUD.
    """
    from uavbench.visualization.overlays import (
        draw_start, draw_goal, draw_task_pois,
        draw_fire, draw_smoke, draw_debris, draw_traffic, draw_nfz,
    )

    OVERVIEW_T = 200  # timestep with visible dynamics but not overwhelming

    print(f"Generating scenario overview with dynamics (t={OVERVIEW_T})...")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    family_items = list(FAMILY_SCENARIOS.items())

    for ax, (family, scenario_id) in zip(axes, family_items):
        config = load_scenario(scenario_id)
        renderer = Renderer(config, mode="paper_min")

        # Capture state + dynamic_state at OVERVIEW_T
        captured: dict[str, Any] = {}

        def cb(heightmap, state, dyn_state, cfg, _store=captured):
            step = state.get("step_idx", 0)
            if step == OVERVIEW_T and "heightmap" not in _store:
                _store["heightmap"] = heightmap
                _store["state"] = state.copy()
                _store["dyn_state"] = {k: (v.copy() if hasattr(v, 'copy') else v)
                                       for k, v in dyn_state.items()}

        run_episode(scenario_id, "astar", SEED, frame_callback=cb)

        # Fallback: if episode ended before OVERVIEW_T, use last captured
        if "heightmap" not in captured:
            # Re-run capturing last frame
            last: dict[str, Any] = {}
            def cb_last(heightmap, state, dyn_state, cfg, _s=last):
                _s["heightmap"] = heightmap
                _s["state"] = state.copy()
                _s["dyn_state"] = {k: (v.copy() if hasattr(v, 'copy') else v)
                                   for k, v in dyn_state.items()}
            run_episode(scenario_id, "astar", SEED, frame_callback=cb_last)
            captured = last

        heightmap = captured["heightmap"]
        state = captured["state"]
        dyn_state = captured["dyn_state"]
        H, W = heightmap.shape
        cell = renderer._cell_px

        # Z=1: Basemap (buildings, landuse, roads)
        frame = renderer._render_basemap(
            heightmap, H, W, cell,
            state.get("landuse_map"),
            state.get("roads_mask"),
        )

        # Z=3-4: Dynamic overlays (smoke, fire, debris, traffic, NFZ)
        smoke_mask = dyn_state.get("smoke_mask")
        if smoke_mask is not None:
            draw_smoke(frame, smoke_mask, cell, alpha_256=51)
        fire_mask = dyn_state.get("fire_mask")
        if fire_mask is not None:
            draw_fire(frame, fire_mask, cell)
        debris_mask = dyn_state.get("debris_mask")
        if debris_mask is not None:
            draw_debris(frame, debris_mask, cell)
        nfz_mask = dyn_state.get("dynamic_nfz_mask")
        if nfz_mask is not None:
            draw_nfz(frame, nfz_mask, cell)
        traffic_mask = dyn_state.get("traffic_closure_mask")
        if traffic_mask is not None:
            draw_traffic(frame, traffic_mask, cell)
        traffic_occ = dyn_state.get("traffic_occupancy_mask")
        if traffic_occ is not None:
            draw_traffic(frame, traffic_occ, cell)

        # Z=9.6: Start/Goal markers
        start_xy = state.get("start_xy", state.get("agent_xy", (0, 0)))
        goal_xy = state.get("goal_xy", (H - 1, W - 1))
        draw_start(frame, start_xy, cell)
        draw_goal(frame, goal_xy, cell)

        # Z=9.7: Mission POI icons
        task_info = state.get("task_info_list", [])
        if task_info:
            draw_task_pois(frame, task_info, cell)

        ax.imshow(frame)
        FAMILY_LABELS = {
            "pharma_delivery": "Pharma Delivery",
            "flood_rescue": "Urban Rescue",
            "fire_surveillance": "Fire Surveillance",
        }
        label = FAMILY_LABELS.get(family, family.replace("_", " ").title())
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
    """Single-panel figure: all 5 trajectories overlaid on OSM basemap with
    dynamics rendered at t=212 (corridor blockage moment)."""
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from uavbench.visualization.overlays import (
        draw_fire, draw_smoke, draw_debris, draw_traffic, draw_nfz,
    )

    CAPTURE_T = 212  # corridor blockage moment

    print(f"\nGenerating planner comparison with dynamics ({COMPARISON_SCENARIO})...")

    config = load_scenario(COMPARISON_SCENARIO)
    renderer = Renderer(config, mode="paper_min")
    cell = renderer._cell_px

    # --- Run all planners, capture raw state at t=CAPTURE_T (no HUD) ---
    trajectories = {}
    metrics_map = {}
    captured_state = {}  # raw state/dyn_state for basemap rendering

    for pid in PLANNER_ORDER:
        print(f"  Running {pid}...")
        captured = {}

        def _cb(heightmap, state, dyn_state, cfg, _store=captured):
            step = state.get("step_idx", 0)
            if step == CAPTURE_T and "heightmap" not in _store:
                _store["heightmap"] = heightmap
                _store["state"] = state.copy()
                _store["dyn_state"] = {k: (v.copy() if hasattr(v, 'copy') else v)
                                       for k, v in dyn_state.items()}

        r = run_episode(COMPARISON_SCENARIO, pid, SEED, frame_callback=_cb)
        trajectories[pid] = r.trajectory
        metrics_map[pid] = r.metrics
        if not captured_state and "heightmap" in captured:
            captured_state = captured

    if not captured_state:
        print("  WARNING: no state captured at t=CAPTURE_T")
        return

    # --- Build background: basemap + dynamics only (no HUD, no agent) ---
    heightmap = captured_state["heightmap"]
    state = captured_state["state"]
    dyn_state = captured_state["dyn_state"]
    H, W = heightmap.shape

    bg_frame = renderer._render_basemap(
        heightmap, H, W, cell,
        state.get("landuse_map"),
        state.get("roads_mask"),
    )
    smoke_mask = dyn_state.get("smoke_mask")
    if smoke_mask is not None:
        draw_smoke(bg_frame, smoke_mask, cell, alpha_256=51)
    fire_mask = dyn_state.get("fire_mask")
    if fire_mask is not None:
        draw_fire(bg_frame, fire_mask, cell)
    debris_mask = dyn_state.get("debris_mask")
    if debris_mask is not None:
        draw_debris(bg_frame, debris_mask, cell)
    nfz_mask = dyn_state.get("dynamic_nfz_mask")
    if nfz_mask is not None:
        draw_nfz(bg_frame, nfz_mask, cell)
    traffic_mask = dyn_state.get("traffic_closure_mask")
    if traffic_mask is not None:
        draw_traffic(bg_frame, traffic_mask, cell)
    traffic_occ = dyn_state.get("traffic_occupancy_mask")
    if traffic_occ is not None:
        draw_traffic(bg_frame, traffic_occ, cell)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.16, 5.5), dpi=150)

    H, W = config.map_size, config.map_size
    # Show the rendered frame (pixel coords → grid extent)
    ax.imshow(
        bg_frame, origin="upper", interpolation="nearest", aspect="equal",
        extent=[0, W, H, 0],
    )

    # --- Overlay all 5 trajectories ---
    for pid in PLANNER_ORDER:
        traj = trajectories[pid]
        if len(traj) < 2:
            continue
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        m = metrics_map[pid]
        success = m.get("success", False)
        steps = m.get("executed_steps_len", 0)
        status = "goal" if success else m.get("termination_reason", "?")
        lbl = f"{PLANNER_LABELS[pid]} ({status}, {steps}s)"
        ax.plot(
            xs, ys, color=PLANNER_COLORS[pid],
            linewidth=1.3, alpha=0.85, zorder=3, label=lbl,
            path_effects=[
                pe.Stroke(linewidth=2.2, foreground="white", alpha=0.4),
                pe.Normal(),
            ],
        )

    # --- Start / Goal markers ---
    start_xy = trajectories[PLANNER_ORDER[0]][0]
    # Find a successful planner's last position as goal
    goal_xy = start_xy
    for pid in PLANNER_ORDER:
        if metrics_map[pid].get("success"):
            goal_xy = trajectories[pid][-1]
            break

    ax.plot(start_xy[0], start_xy[1], "o", color="#009E73", markersize=10,
            markeredgecolor="black", markeredgewidth=1.0, zorder=10)
    ax.plot(goal_xy[0], goal_xy[1], "*", color="#FFD700", markersize=14,
            markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    # --- Annotation: A* stuck position ---
    astar_stuck = trajectories["astar"][-1]
    ax.annotate(
        "Fire blocks\ncorridor", xy=(astar_stuck[0], astar_stuck[1]),
        xytext=(astar_stuck[0] + 40, astar_stuck[1] - 40),
        fontsize=6, color="#CC0000", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1),
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=8,
    )

    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9, edgecolor="grey")
    ax.set_title(
        f"Five planners, same episode (Penteli, seed {SEED}) \u2014 "
        f"world state at t={CAPTURE_T}",
        fontsize=8, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
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
