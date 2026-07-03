#!/usr/bin/env python3
"""Risk cost-surface figure as a pure risk-coefficient sweep.

Replaces the former per-planner "five different worlds" figure. The effective
cost surface w(x) = 1 + rho * R(x) depends ONLY on the risk coefficient rho
(the same formula for every graph-search planner), so this figure holds the
fire state fixed and sweeps rho in {0, 1, 2, 5, 10} — a within-formula
demonstration that rho alone reshapes the cost landscape. No planner comparison.

Output: outputs/paper_figures/risk_perception_rho_sweep.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare.benchmark.runner import run_episode
from flare.blocking import compute_risk_cost_map
from flare.scenarios.loader import load_scenario

ROOT = Path(__file__).resolve().parent.parent

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
RHOS = [0, 1, 2, 5, 10]
VMAX = 6.0  # fixed colour scale (matches the previous figure's 0..6 range)


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def main() -> None:
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
    no_fly = np.zeros_like(heightmap, dtype=bool)
    risk = compute_risk_cost_map(heightmap, no_fly, cfg, dyn_state)
    print(f"Risk map: shape={risk.shape}, min={risk.min():.3f}, max={risk.max():.3f}")

    # cost surface as a function of rho only: w(x) = 1 + rho * R(x)
    views = {(f"ρ = {r}" + ("\n(risk-blind)" if r == 0 else "")): (1.0 + r * risk)
             for r in RHOS}

    building_mask = heightmap > 0
    fire_mask = dyn_state.get("fire_mask", np.zeros_like(heightmap, dtype=bool))

    fig, axes = plt.subplots(1, len(RHOS), figsize=(7.0, 3.0))
    im = None
    for ax, (title, view) in zip(axes, views.items()):
        display = view.copy().astype(np.float64)
        display[building_mask] = np.nan
        im = ax.imshow(display, cmap="RdYlGn_r", vmin=0, vmax=VMAX,
                       interpolation="nearest", aspect="equal")
        building_overlay = np.zeros((*heightmap.shape, 4))
        building_overlay[building_mask] = [0.31, 0.31, 0.31, 1.0]
        ax.imshow(building_overlay, interpolation="nearest", aspect="equal")
        if fire_mask.any():
            ax.contour(fire_mask.astype(float), levels=[0.5], colors=["red"],
                       linewidths=0.8)
        tp = captured.get("traffic_positions")
        if tp is not None and len(tp) > 0:
            ax.scatter(tp[:, 1], tp[:, 0], s=12, c="#D55E00", marker="s",
                       edgecolors="white", linewidths=0.3, zorder=8)
        agent = captured.get("agent_xy")
        if agent:
            ax.plot(agent[0], agent[1], "x", color="white",
                    markersize=6, markeredgewidth=1.5, zorder=8)
            ax.plot(agent[0], agent[1], "x", color="#4090D0",
                    markersize=5, markeredgewidth=1.0, zorder=9)
        ax.set_title(title, fontsize=7, fontweight="normal", color="black")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=30, pad=0.02,
                        orientation="horizontal")
    cbar.set_label("Weighted traversal cost  w(x) = 1 + ρ·R(x)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        f"Same fire state (t = {CAPTURE_STEP}): the effective cost surface as the "
        f"risk coefficient ρ increases",
        fontsize=8, fontweight="normal", color="black", y=1.02,
    )
    _save(fig, "risk_perception_rho_sweep")


if __name__ == "__main__":
    main()
