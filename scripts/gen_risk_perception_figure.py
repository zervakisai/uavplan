#!/usr/bin/env python3
"""Risk perception 5-panel figure (Task 5).

Shows the same fire state at t=150 viewed through each planner's risk
weighting. Periodic (alpha=5.0) sees RED EVERYWHERE while Aggressive
(beta=0.5) sees MOSTLY GREEN.

Output: outputs/paper_figures/risk_perception_5panel.{png,pdf}
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare.benchmark.runner import run_episode
from flare.blocking import compute_risk_cost_map
from flare.scenarios.loader import load_scenario
from flare.visualization.labels import PLANNER_ORDER

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "flare"

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
CAPTURE_STEP = 150


def _extract(filepath: Path, varname: str) -> float:
    source = filepath.read_text()
    match = re.search(rf"^{varname}\s*=\s*(.+)", source, re.MULTILINE)
    return float(ast.literal_eval(match.group(1).strip()))


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def main() -> None:
    alpha = _extract(SRC / "planners" / "periodic_replan.py", "_RISK_ALPHA")
    beta = _extract(SRC / "planners" / "aggressive_replan.py", "_RISK_BETA")
    gamma = _extract(SRC / "planners" / "incremental_astar.py", "_RISK_GAMMA")
    delta = _extract(SRC / "planners" / "apf.py", "_RISK_DELTA")

    print(f"Coefficients: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")

    # Capture state at t=CAPTURE_STEP
    cfg = load_scenario(SCENARIO)
    captured = {}

    def cb(heightmap, state, dyn_state, cfg_arg):
        step = state.get("step_idx", 0)
        if step == CAPTURE_STEP and "done" not in captured:
            captured["heightmap"] = heightmap.copy()
            captured["dyn_state"] = {
                k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in dyn_state.items()
            }
            # Get no_fly from state if available, else approximate
            tp = dyn_state.get("traffic_positions")
            captured["traffic_positions"] = tp.copy() if tp is not None else None
            captured["agent_xy"] = state.get("agent_xy")
            captured["done"] = True

    print(f"Running episode to capture state at t={CAPTURE_STEP}...")
    run_episode(SCENARIO, "aggressive_replan", SEED, frame_callback=cb)

    if "done" not in captured:
        print("ERROR: Could not capture frame at target step.")
        sys.exit(1)

    heightmap = captured["heightmap"]
    dyn_state = captured["dyn_state"]

    # Compute base risk map
    # no_fly approximation: static NFZ not easily available here,
    # use empty mask (risk map focuses on dynamic layers)
    no_fly = np.zeros_like(heightmap, dtype=bool)
    risk = compute_risk_cost_map(heightmap, no_fly, cfg, dyn_state)

    print(f"Risk map: shape={risk.shape}, min={risk.min():.3f}, max={risk.max():.3f}")

    # Compute 5 weighted views
    views = {
        "A*\n(ignores risk)": np.ones_like(risk),
        f"Periodic\n(\u03b1={alpha})": 1.0 + alpha * risk,
        f"Aggressive\n(\u03b2={beta})": 1.0 + beta * risk,
        f"Incr. A*\n(\u03b3={gamma})": 1.0 + gamma * risk,
        f"APF\n(\u03b4={delta})": delta * risk,
    }

    # Shared vmin/vmax
    all_vals = np.concatenate([v.ravel() for v in views.values()])
    vmin = 0
    vmax = float(np.percentile(all_vals[all_vals > 0], 99))  # clip outliers

    # Building mask
    building_mask = heightmap > 0
    fire_mask = dyn_state.get("fire_mask", np.zeros_like(heightmap, dtype=bool))

    # Figure
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 3.0))

    for ax, (title, view) in zip(axes, views.items()):
        # Mask buildings
        display = view.copy().astype(np.float64)
        display[building_mask] = np.nan

        im = ax.imshow(
            display, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="equal",
        )

        # Grey out buildings
        building_overlay = np.zeros((*heightmap.shape, 4))
        building_overlay[building_mask] = [0.31, 0.31, 0.31, 1.0]
        ax.imshow(building_overlay, interpolation="nearest", aspect="equal")

        # Fire edges (bright red contour)
        if fire_mask.any():
            ax.contour(fire_mask.astype(float), levels=[0.5], colors=["red"],
                       linewidths=0.8)

        # Vehicle positions (small red squares)
        tp = captured.get("traffic_positions")
        if tp is not None and len(tp) > 0:
            ax.scatter(tp[:, 1], tp[:, 0], s=12, c="#D55E00", marker="s",
                       edgecolors="white", linewidths=0.3, zorder=8)

        # Drone position (blue x with white outline)
        agent = captured.get("agent_xy")
        if agent:
            # White outline
            ax.plot(agent[0], agent[1], "x", color="white",
                    markersize=6, markeredgewidth=1.5, zorder=8)
            # Blue cross (top)
            ax.plot(agent[0], agent[1], "x", color="#4090D0",
                    markersize=5, markeredgewidth=1.0, zorder=9)

        ax.set_title(title, fontsize=7, fontweight="normal", color="black")
        ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=30, pad=0.02,
                        orientation="horizontal")
    cbar.set_label("Weighted traversal cost", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Risk perception at t={CAPTURE_STEP}: same fire, five different worlds",
        fontsize=8, fontweight="normal", color="black", y=1.02,
    )

    _save(fig, "risk_perception_5panel")


if __name__ == "__main__":
    main()
