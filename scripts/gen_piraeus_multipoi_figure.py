#!/usr/bin/env python3
"""Generate 3-panel Piraeus multi-POI comparison figure with dynamic background.

Shows ranking inversion on OSM basemap with fire/smoke/debris/traffic visible.
Each panel renders the world state at a key timestep, then overlays the trajectory.

Usage:
    python scripts/gen_piraeus_multipoi_figure.py
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from uavbench.benchmark.runner import run_episode
from uavbench.envs.urban import UrbanEnvV2
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCENARIO_ID = "osm_piraeus_urban_rescue_medium"
SEED = 11
PLANNERS = ["astar", "incremental_astar", "aggressive_replan"]
PLANNER_LABELS = {
    "astar": "A*",
    "incremental_astar": "Incr. A*",
    "aggressive_replan": "Aggressive Replan",
}
PLANNER_COLORS = {
    "astar": "#E69F00",
    "incremental_astar": "#CC79A7",
    "aggressive_replan": "#D55E00",
}

# Timestep at which to capture dynamic state for background.
# Must be reachable by ALL planners (shortest episode is ~141 steps).
# t=100: fire developing, traffic active, all planners still running.
CAPTURE_T = 100

OUTPUT_DIR = "outputs/paper_figures"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = load_scenario(SCENARIO_ID)
    renderer = Renderer(config, mode="paper_min")
    cell = renderer._cell_px

    # ------------------------------------------------------------------
    # Step 1: Run episodes, capture rendered frame at CAPTURE_T + results
    # ------------------------------------------------------------------
    results = {}
    rendered_bgs = {}

    for p in PLANNERS:
        print(f"Running {p} on {SCENARIO_ID} seed={SEED}...")
        t0 = time.perf_counter()

        captured = {}

        def _cb(heightmap, state, dyn_state, cfg, _store=captured):
            step = state.get("step_idx", 0)
            # Always keep the latest frame args for fallback
            _store["_last"] = (heightmap, dict(state), dict(dyn_state))
            if step == CAPTURE_T:
                frame, _ = renderer.render_frame(heightmap, state, dyn_state)
                _store["frame"] = frame
                _store["heightmap"] = heightmap

        r = run_episode(SCENARIO_ID, p, seed=SEED, frame_callback=_cb)
        elapsed = time.perf_counter() - t0
        # If CAPTURE_T wasn't reached, render the last available frame
        if "frame" not in captured and "_last" in captured:
            hm, st, ds = captured["_last"]
            frame, _ = renderer.render_frame(hm, st, ds)
            captured["frame"] = frame
            captured["heightmap"] = hm
        results[p] = r
        rendered_bgs[p] = captured
        m = r.metrics
        status = "SUCCESS" if m.get("success") else m.get("termination_reason", "?")
        print(
            f"  [{status}] steps={m.get('executed_steps_len', 0)} "
            f"score={m.get('mission_score', 0):.3f} "
            f"tasks={m.get('tasks_completed', 0)}/{m.get('tasks_total', '?')} "
            f"({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Step 2: Get POI positions from env
    # ------------------------------------------------------------------
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=SEED)
    heightmap = env._heightmap
    H, W = heightmap.shape
    start_xy = info.get("start_xy") or env._agent_xy
    goal_xy = info.get("goal_xy") or env._goal_xy

    mission = env._mission
    all_poi_positions = list(mission.all_task_positions) if mission else []
    print(f"Start: {start_xy}, Goal: {goal_xy}")
    print(f"POI positions: {all_poi_positions}")

    # ------------------------------------------------------------------
    # Step 3: Compute zoom bounding box (in grid coords)
    # ------------------------------------------------------------------
    all_xs = [start_xy[0], goal_xy[0]]
    all_ys = [start_xy[1], goal_xy[1]]
    for poi_xy in all_poi_positions:
        all_xs.append(poi_xy[0])
        all_ys.append(poi_xy[1])
    for planner_id in PLANNERS:
        for pt in results[planner_id].trajectory:
            all_xs.append(pt[0])
            all_ys.append(pt[1])

    PAD = 30
    gx_min = max(0, min(all_xs) - PAD)
    gx_max = min(W, max(all_xs) + PAD)
    gy_min = max(0, min(all_ys) - PAD)
    gy_max = min(H, max(all_ys) + PAD)

    # Expand viewport westward to show Piraeus coastline (water at X<350)
    gx_min = max(0, min(gx_min, 180))

    # Make square-ish
    span_x = gx_max - gx_min
    span_y = gy_max - gy_min
    if span_x > span_y:
        diff = span_x - span_y
        gy_min = max(0, gy_min - diff // 2)
        gy_max = min(H, gy_max + diff // 2)
    elif span_y > span_x:
        diff = span_y - span_x
        gx_min = max(0, gx_min - diff // 2)
        gx_max = min(W, gx_max + diff // 2)

    # Convert to pixel coords for cropping the rendered frame
    px_min = gx_min * cell
    px_max = gx_max * cell
    py_min = gy_min * cell
    py_max = gy_max * cell

    print(f"Zoom: grid [{gx_min}:{gx_max}, {gy_min}:{gy_max}], "
          f"pixels [{px_min}:{px_max}, {py_min}:{py_max}]")

    # ------------------------------------------------------------------
    # Step 4: Create 3-panel figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), dpi=150)

    for ax_idx, planner_id in enumerate(PLANNERS):
        ax = axes[ax_idx]
        r = results[planner_id]
        m = r.metrics
        trajectory = r.trajectory
        events = r.events

        # Extract completed POIs
        completed_pois = set()
        for ev in events:
            if ev.get("type") == "task_completed":
                xy = ev.get("xy")
                if xy is not None:
                    completed_pois.add(tuple(xy))

        # --- Background: rendered frame cropped to zoom region ---
        bg = rendered_bgs[planner_id]
        if "frame" in bg:
            frame = bg["frame"]
            # Crop to zoom region (pixel coords). Frame may include legend bar.
            fh, fw = frame.shape[:2]
            crop = frame[
                min(py_min, fh):min(py_max, fh),
                min(px_min, fw):min(px_max, fw),
            ]
        else:
            # Fallback: grey
            crop = np.full(
                (py_max - py_min, px_max - px_min, 3), 200, dtype=np.uint8
            )

        ax.imshow(
            crop, origin="upper", interpolation="nearest", aspect="equal",
            extent=[gx_min, gx_max, gy_max, gy_min],
        )

        # --- Trajectory ---
        color = PLANNER_COLORS[planner_id]
        if len(trajectory) > 1:
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]
            ax.plot(
                traj_x, traj_y,
                color=color, linewidth=1.5, alpha=0.9, zorder=3,
                path_effects=[
                    pe.Stroke(linewidth=2.5, foreground="white", alpha=0.5),
                    pe.Normal(),
                ],
            )

        # --- Start / Goal ---
        ax.plot(start_xy[0], start_xy[1], "o",
                color="#009E73", markersize=8, markeredgecolor="white",
                markeredgewidth=1.0, zorder=5)
        ax.plot(goal_xy[0], goal_xy[1], "s",
                color="#CC0000", markersize=8, markeredgecolor="white",
                markeredgewidth=1.0, zorder=5)

        # --- POI markers ---
        for i, poi_xy in enumerate(all_poi_positions):
            is_done = tuple(poi_xy) in completed_pois
            marker = "^" if is_done else "v"
            poi_color = "#009E73" if is_done else "#CC0000"
            ax.plot(poi_xy[0], poi_xy[1], marker,
                    color=poi_color, markersize=12, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=6)
            status_sym = "\u2713" if is_done else "\u2717"
            ax.annotate(
                f"P{i+1}{status_sym}", (poi_xy[0] + 2, poi_xy[1] - 3),
                fontsize=6, fontweight="bold", color=poi_color, zorder=7,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

        # --- Title ---
        success = m.get("success", False)
        score = m.get("mission_score", 0.0)
        tasks_done = m.get("tasks_completed", 0)
        tasks_total = m.get("tasks_total", "?")
        sr_text = "SR 100%" if success else "SR 0%"
        label = PLANNER_LABELS[planner_id]
        title = (f"{label}\n{sr_text}  |  score = {score:.2f}  "
                 f"|  tasks = {tasks_done}/{tasks_total}")
        ax.set_title(title, fontsize=6.5, fontweight="bold", pad=4)

        ax.set_xlim(gx_min, gx_max)
        ax.set_ylim(gy_max, gy_min)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#009E73",
               markersize=6, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#CC0000",
               markersize=6, label="Goal"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#009E73",
               markersize=7, label="POI (rescued)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#CC0000",
               markersize=7, label="POI (missed)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        fontsize=6, framealpha=0.9, borderpad=0.4,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(pad=0.4, w_pad=0.5)
    plt.subplots_adjust(bottom=0.08)

    pdf_path = os.path.join(OUTPUT_DIR, "piraeus_multipoi_comparison.pdf")
    png_path = os.path.join(OUTPUT_DIR, "piraeus_multipoi_comparison.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
