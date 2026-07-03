#!/usr/bin/env python3
"""Within-planner ρ trajectory figure: ONE planner, ρ ∈ {0,1,2,5,10}.

Shows how the SAME planner's path deflects away from hazards as the risk
coefficient ρ grows (same scenario, same seed, same deterministic fire).
Replaces the old cross-planner trajectory figure, per the paper's theme
("how much ρ affects each planner").

Output: outputs/paper_figures/trajectory_rho_sweep.{png,pdf}
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.renderer import Renderer
from flare.visualization.overlays import (
    draw_fire, draw_smoke, draw_debris, draw_traffic, draw_nfz,
)

FIG_DIR = "outputs/paper_figures"
SCENARIO = "osm_penteli_pharma_delivery_medium"
SEED = 42
PLANNER = "aggressive_replan"        # featured planner (risk-tolerant → risk-averse with ρ)
RHOS = [0.0, 1.0, 2.0, 5.0, 10.0]
CAPTURE_T = 150
PCOLOR = "#c75b3a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def main() -> None:
    base = load_scenario(SCENARIO)
    print(f"Within-ρ trajectory: planner={PLANNER}, scenario={SCENARIO}, seed={SEED}")

    # Step 1: run the featured planner at each ρ, collect trajectories
    trajs: dict[float, list] = {}
    mets: dict[float, dict] = {}
    for rho in RHOS:
        cfg = replace(base, risk_rho=rho)
        r = run_episode(SCENARIO, PLANNER, SEED, config_override=cfg)
        trajs[rho] = r.trajectory
        mets[rho] = r.metrics
        print(f"  ρ={rho:<4}: {r.metrics.get('termination_reason'):16s} "
              f"({r.metrics.get('executed_steps_len',0)} steps)")

    # Step 2: capture the (ρ-independent) hazard basemap once, at t=CAPTURE_T
    renderer = Renderer(base, mode="paper_min")
    renderer._cell_px = 1
    cell = renderer._cell_px
    cap: dict[str, Any] = {}

    def _cb(heightmap, state, dyn_state, cfg):
        if state.get("step_idx", 0) == CAPTURE_T and "heightmap" not in cap:
            cap["heightmap"] = heightmap
            cap["state"] = state.copy()
            cap["dyn_state"] = {k: (v.copy() if hasattr(v, "copy") else v)
                                for k, v in dyn_state.items()}

    run_episode(SCENARIO, PLANNER, SEED,
                config_override=replace(base, risk_rho=2.0), frame_callback=_cb)
    if "heightmap" not in cap:  # fallback: last frame
        last: dict[str, Any] = {}
        def _cbl(hm, st, dyn, cfg):
            last.update(heightmap=hm, state=st.copy(),
                        dyn_state={k: (v.copy() if hasattr(v, "copy") else v)
                                   for k, v in dyn.items()})
        run_episode(SCENARIO, PLANNER, SEED,
                    config_override=replace(base, risk_rho=2.0), frame_callback=_cbl)
        cap = last

    heightmap = cap["heightmap"]; state = cap["state"]; dyn = cap["dyn_state"]
    H, W = heightmap.shape
    bg = renderer._render_basemap(heightmap, H, W, cell,
                                  state.get("landuse_map"), state.get("roads_mask"))
    if dyn.get("smoke_mask") is not None: draw_smoke(bg, dyn["smoke_mask"], cell, alpha_256=40)
    if dyn.get("fire_mask") is not None: draw_fire(bg, dyn["fire_mask"], cell)
    if dyn.get("traffic_closure_mask") is not None: draw_traffic(bg, dyn["traffic_closure_mask"], cell)
    if dyn.get("nfz_mask") is not None: draw_nfz(bg, dyn["nfz_mask"], cell)
    if dyn.get("debris_mask") is not None: draw_debris(bg, dyn["debris_mask"], cell)

    # Step 3: 5-panel figure — same map, one ρ per panel
    fig, axes = plt.subplots(1, len(RHOS), figsize=(11.0, 2.6))
    fig.subplots_adjust(wspace=0.04, top=0.80, bottom=0.02, left=0.01, right=0.99)
    goal = state.get("goal_xy")
    for ax, rho in zip(axes, RHOS):
        ax.imshow(bg, origin="upper", interpolation="nearest", aspect="equal",
                  extent=[0, W, H, 0])
        traj = trajs[rho]; m = mets[rho]; success = m.get("success", False)
        if traj:
            sx, sy = traj[0]
            ax.plot(sx, sy, "x", color="white", markersize=6, markeredgewidth=1.6, zorder=2)
            ax.plot(sx, sy, "x", color="#4090D0", markersize=5, markeredgewidth=1.0, zorder=3)
        if len(traj) >= 2:
            xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
            ax.plot(xs, ys, "-" if success else ":", color="#ffdd00", linewidth=2.4,
                    path_effects=[pe.Stroke(linewidth=4.0, foreground="#8a1a1a", alpha=0.9), pe.Normal()],
                    zorder=5)
        if goal:
            ax.plot(goal[0], goal[1], "P", color="#E6C619", markersize=7,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=10)
        if traj:
            ex, ey = traj[-1]
            if success:
                ax.plot(ex, ey, "o", color="#009E73", markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=11)
            else:
                ax.plot(ex, ey, "X", color="#CC3311", markersize=6.5,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=11)
        steps = m.get("executed_steps_len", 0)
        out = "reached goal" if success else m.get("termination_reason", "").replace("_", " ")
        ax.set_title(fr"$\rho={rho:g}$" + f"\n{steps} steps · {out}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same planner (Aggressive) under increasing ρ: at ρ=0 it ignores the risk "
                 "field and is caught by fire; ρ>0 detours to safety",
                 x=0.5, y=1.03, fontsize=9, fontweight="bold")
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/trajectory_rho_sweep.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/trajectory_rho_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {FIG_DIR}/trajectory_rho_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
