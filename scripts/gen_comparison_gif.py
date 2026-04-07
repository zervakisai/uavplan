#!/usr/bin/env python3
"""Generate the trajectory-comparison GIF in the *paper-figure* style.

This GIF mirrors `gen_trajectory_comparison_5panel.py` exactly — same basemap,
same overlays, same start/goal/trajectory marker style — but unrolls the
episode over time. Three side-by-side panels (A* / Aggressive / Periodic)
show each planner's trajectory growing on the same wildfire.

Output: outputs/comparison_gifs/comparison_<scenario>_s<seed>.gif

Usage:
    python scripts/gen_comparison_gif.py [--scenario ...] [--seed 42] [--fps 8]
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.renderer import Renderer
from flare.visualization.labels import (
    PLANNER_SHORT, PLANNER_COLORS,
)
from flare.visualization.overlays import (
    draw_fire, draw_smoke, draw_debris, draw_traffic, draw_nfz,
)

# ---------------------------------------------------------------------------
# Configuration — same defaults as the static figure
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "flare"
OUTPUT_DIR = "outputs/comparison_gifs"
DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"

# Three panels, identical ordering to the paper figure
PANEL_PLANNERS = ["astar", "aggressive_replan", "periodic_replan"]

# IEEE-style matplotlib formatting (matches the static figure)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 100,
})

COEFF_LABELS: dict[str, str | None] = {
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


# ---------------------------------------------------------------------------
# Per-planner episode runner — captures trajectories AND per-step dyn state
# ---------------------------------------------------------------------------


def _run_planner(
    scenario_id: str,
    planner_id: str,
    seed: int,
    capture_dyn: bool = False,
) -> dict[str, Any]:
    """Run one episode and collect trajectory + (optionally) per-step dyn_state.

    `capture_dyn` is only set for the first planner — fire CA is a pure
    function (FD-4), so dyn_state evolution is identical across planners
    for the same (scenario, seed). Reusing it saves a 5× memory overhead.
    """
    trajectory: list[tuple[int, int]] = []
    captured: dict[str, Any] = {"heightmap": None, "state0": None, "frames": []}

    def _cb(heightmap, state, dyn_state, cfg):
        ax, ay = state.get("agent_xy", (0, 0))
        trajectory.append((int(ax), int(ay)))
        if captured["heightmap"] is None:
            captured["heightmap"] = heightmap.copy()
            captured["state0"] = {
                "landuse_map": state.get("landuse_map"),
                "roads_mask": state.get("roads_mask"),
                "start_xy": tuple(state.get("start_xy", (ax, ay))),
                "goal_xy": tuple(state.get("goal_xy", (0, 0))),
            }
        if capture_dyn:
            snap = {
                "step_idx": int(state.get("step_idx", 0)),
                "fire_mask": _copy_arr(dyn_state.get("fire_mask")),
                "smoke_mask": _copy_arr(dyn_state.get("smoke_mask")),
                "debris_mask": _copy_arr(dyn_state.get("debris_mask")),
                "traffic_closure_mask": _copy_arr(dyn_state.get("traffic_closure_mask")),
                "traffic_positions": _copy_pos(dyn_state.get("traffic_positions")),
                "nfz_mask": _copy_arr(dyn_state.get("nfz_mask")),
            }
            captured["frames"].append(snap)

    result = run_episode(scenario_id, planner_id, seed, frame_callback=_cb)

    return {
        "trajectory": trajectory,
        "metrics": result.metrics,
        "heightmap": captured["heightmap"],
        "state0": captured["state0"],
        "dyn_frames": captured["frames"],
    }


def _copy_arr(a):
    if a is None:
        return None
    return np.asarray(a).copy()


def _copy_pos(p):
    if p is None:
        return None
    if hasattr(p, "copy"):
        return p.copy()
    return list(p)


# ---------------------------------------------------------------------------
# Frame compositor — replicates the paper-figure layout exactly
# ---------------------------------------------------------------------------


def _draw_dyn_overlays(bg: np.ndarray, dyn: dict[str, Any], cell: int) -> None:
    """Apply the SAME overlay sequence the static figure uses."""
    smoke_mask = dyn.get("smoke_mask")
    if smoke_mask is not None:
        draw_smoke(bg, smoke_mask, cell, alpha_256=40)
    fire_mask = dyn.get("fire_mask")
    if fire_mask is not None:
        draw_fire(bg, fire_mask, cell)
    traffic_closure_mask = dyn.get("traffic_closure_mask")
    if traffic_closure_mask is not None:
        draw_traffic(bg, traffic_closure_mask, cell)
    traffic_positions = dyn.get("traffic_positions")
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
    nfz_mask = dyn.get("nfz_mask")
    if nfz_mask is not None:
        draw_nfz(bg, nfz_mask, cell)
    debris_mask = dyn.get("debris_mask")
    if debris_mask is not None:
        draw_debris(bg, debris_mask, cell)


def _make_panel_frame(
    base_basemap: np.ndarray,
    H: int,
    W: int,
    cell: int,
    dyn_snapshot: dict[str, Any],
    state0: dict[str, Any],
    planner_traj_slices: dict[str, list[tuple[int, int]]],
    planner_metrics: dict[str, dict],
    seed: int,
    global_step: int,
) -> np.ndarray:
    """Render one composite frame matching the paper figure layout."""
    # Apply dyn overlays on a fresh copy of the cached basemap
    bg = base_basemap.copy()
    _draw_dyn_overlays(bg, dyn_snapshot, cell)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))
    fig.subplots_adjust(wspace=0.05, top=0.85, bottom=0.02, left=0.01, right=0.99)

    for ax, pid in zip(axes, PANEL_PLANNERS):
        ax.imshow(
            bg, origin="upper", interpolation="nearest",
            aspect="equal", extent=[0, W, H, 0],
        )
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        traj = planner_traj_slices[pid]
        success = planner_metrics[pid].get("success", False)

        # Start (cyan X with white halo) — matches static figure
        if traj:
            sx, sy = traj[0]
            ax.plot(sx, sy, "x", color="white", markersize=6,
                    markeredgewidth=1.5, zorder=2)
            ax.plot(sx, sy, "x", color="#4090D0", markersize=5,
                    markeredgewidth=1.0, zorder=3)

        # Trajectory grown to current step
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ls = "-" if success else ":"
            ax.plot(xs, ys, ls, color=PLANNER_COLORS[pid], linewidth=1.2,
                    path_effects=[
                        pe.Stroke(linewidth=2.0, foreground="black", alpha=0.25),
                        pe.Normal(),
                    ], zorder=5)

        # Goal (yellow plus)
        goal = state0.get("goal_xy")
        if goal:
            ax.plot(goal[0], goal[1], "P", color="#E6C619", markersize=7,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=10)

        # Current head marker — green circle if alive/successful, red X if failed
        traj_full_len = len(planner_metrics[pid].get("_full_traj", traj))
        finished = len(traj) >= traj_full_len
        if traj:
            ex, ey = traj[-1]
            if finished and success:
                ax.plot(ex, ey, "o", color="#009E73", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=11)
            elif finished and not success:
                ax.plot(ex, ey, "X", color="#CC3311", markersize=6,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=11)
            else:
                # Live UAV head — same style as the start marker in the
                # paper figure: cyan X with a white halo, drawn on top.
                ax.plot(ex, ey, "x", color="white", markersize=7,
                        markeredgewidth=2.0, zorder=11)
                ax.plot(ex, ey, "x", color="#4090D0", markersize=6,
                        markeredgewidth=1.3, zorder=12)

        coeff = COEFF_LABELS[pid]
        steps_now = max(0, len(traj) - 1)
        if finished:
            if success:
                subtitle = f"{steps_now} steps"
            else:
                term = planner_metrics[pid].get("termination_reason", "failed")
                subtitle = f"{term}, t={steps_now}"
        else:
            subtitle = f"t={steps_now}"
        ax.set_title(f"{PLANNER_SHORT[pid]} ({coeff})\n{subtitle}",
                     fontsize=5.5, fontweight="normal", color="black")
        ax.axis("off")

    fig.suptitle(f"Three planners, same wildfire (seed {seed})  ·  step {global_step}",
                 fontsize=8, fontweight="normal", color="black", y=0.98)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--skip-frames", type=int, default=4,
                   help="Render every Nth simulation step (smaller = smoother + bigger GIF)")
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Coefficient labels from source-of-truth files
    alpha = _extract(SRC / "planners" / "periodic_replan.py", "_RISK_ALPHA")
    beta = _extract(SRC / "planners" / "aggressive_replan.py", "_RISK_BETA")
    gamma = _extract(SRC / "planners" / "incremental_astar.py", "_RISK_GAMMA")
    delta = _extract(SRC / "planners" / "apf.py", "_RISK_DELTA")
    COEFF_LABELS["periodic_replan"] = f"\u03b1={alpha}"
    COEFF_LABELS["aggressive_replan"] = f"\u03b2={beta}"
    COEFF_LABELS["incremental_astar"] = f"\u03b3={gamma}"
    COEFF_LABELS["apf"] = f"\u03b4={delta}"

    print("=== Trajectory comparison GIF (paper-figure style) ===")
    print(f"  Scenario: {args.scenario}")
    print(f"  Seed:     {args.seed}")
    print(f"  Panels:   {PANEL_PLANNERS}")
    print()

    # Run all 3 panel planners. Capture dyn snapshots from the FIRST run only —
    # fire CA is deterministic & agent-independent (FD-4), so all planners see
    # the same dyn evolution and we save 3× memory.
    runs: dict[str, dict[str, Any]] = {}
    dyn_frames: list[dict[str, Any]] = []
    heightmap: np.ndarray | None = None
    state0: dict[str, Any] | None = None

    for i, pid in enumerate(PANEL_PLANNERS):
        print(f"  Running {pid}...", end="", flush=True)
        t0 = time.perf_counter()
        data = _run_planner(args.scenario, pid, args.seed, capture_dyn=(i == 0))
        elapsed = time.perf_counter() - t0
        runs[pid] = data
        if i == 0:
            dyn_frames = data["dyn_frames"]
            heightmap = data["heightmap"]
            state0 = data["state0"]
        m = data["metrics"]
        status = "OK" if m.get("success") else m.get("termination_reason", "?")
        print(f" [{status}] {len(data['trajectory'])} steps ({elapsed:.1f}s)")

    assert heightmap is not None and state0 is not None

    # Build the basemap ONCE with the renderer (paper_min, cell_px=1 — same as figure)
    config = load_scenario(args.scenario)
    renderer = Renderer(config, mode="paper_min")
    renderer._cell_px = 1
    cell = renderer._cell_px
    H, W = heightmap.shape
    base_basemap = renderer._render_basemap(
        heightmap, H, W, cell,
        state0.get("landuse_map"),
        state0.get("roads_mask"),
    )

    # Animation length = max trajectory length across panel planners
    max_len = max(len(runs[pid]["trajectory"]) for pid in PANEL_PLANNERS)
    n_dyn = len(dyn_frames)

    # Stash full trajectory length for finished/live marker decisions
    for pid in PANEL_PLANNERS:
        runs[pid]["metrics"]["_full_traj"] = runs[pid]["trajectory"]

    # Build frame indices (stride = skip_frames, plus a final frame)
    indices = list(range(0, max_len, max(1, args.skip_frames)))
    if indices[-1] != max_len - 1:
        indices.append(max_len - 1)

    print(f"\n  Compositing {len(indices)} frames "
          f"(max_len={max_len}, dyn_frames={n_dyn})...")

    frames: list[np.ndarray] = []
    for fi, t in enumerate(indices):
        # Sliced trajectories up to step t
        slices = {}
        for pid in PANEL_PLANNERS:
            tj = runs[pid]["trajectory"]
            slices[pid] = tj[: min(t + 1, len(tj))]

        # Pick dyn snapshot at the same logical step (clamp to last available)
        dyn_idx = min(t, n_dyn - 1) if n_dyn > 0 else 0
        dyn = dyn_frames[dyn_idx] if dyn_frames else {}

        img = _make_panel_frame(
            base_basemap, H, W, cell, dyn, state0,
            slices,
            {pid: runs[pid]["metrics"] for pid in PANEL_PLANNERS},
            args.seed, t,
        )
        frames.append(img)
        if (fi + 1) % 10 == 0 or fi == len(indices) - 1:
            print(f"    {fi + 1}/{len(indices)} frames")

    # Hold on the last frame for ~1s
    if frames:
        hold = max(1, args.fps)
        for _ in range(hold):
            frames.append(frames[-1])

    gif_name = f"comparison_{args.scenario}_s{args.seed}.gif"
    gif_path = os.path.join(args.output, gif_name)
    if frames:
        duration_ms = 1000 // args.fps
        iio.imwrite(str(gif_path), frames, extension=".gif",
                     duration=duration_ms, loop=0)
        print(f"\n  GIF: {gif_path} ({len(frames)} frames)")
    else:
        print("\n  Warning: no frames to write")

    print("Done.")


if __name__ == "__main__":
    main()
