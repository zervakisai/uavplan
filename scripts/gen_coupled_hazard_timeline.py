#!/usr/bin/env python3
"""Coupled hazard timeline — fire drives everything.

2-panel stacked figure:
Panel A: Stacked area chart of hazard layers (fire, road closures, debris)
Panel B: Agent distance-to-goal for 2 planners (Aggressive + Periodic)

Shows how closures GROW with fire (they're coupled via InteractionEngine).

Output: outputs/paper_figures/coupled_hazard_timeline.{png,pdf}
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.labels import PLANNER_COLORS, PLANNER_SHORT

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = "outputs/paper_figures"
SCENARIO = "osm_piraeus_urban_rescue_medium"
SEED = 42

# Planners to compare
PLANNERS = ["aggressive_replan", "periodic_replan"]

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


def _save(fig, name, out_dir=FIG_DIR):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def _collect_data(scenario_id: str, planner_id: str, seed: int, cfg) -> dict[str, Any]:
    """Run episode and collect per-step hazard + distance data."""
    steps: list[int] = []
    fire_counts: list[int] = []
    closure_counts: list[int] = []
    debris_counts: list[int] = []
    nfz_counts: list[int] = []
    distances: list[float] = []
    replan_steps: list[int] = []

    def cb(heightmap, state, dyn_state, cfg_arg):
        step = state.get("step_idx", 0)
        agent = state.get("agent_xy", (0, 0))
        goal = state.get("goal_xy", (0, 0))
        dist = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])

        fire_mask = dyn_state.get("fire_mask")
        fc = int(fire_mask.sum()) if fire_mask is not None else 0

        closure_mask = dyn_state.get("traffic_closure_mask")
        cc = int(closure_mask.sum()) if closure_mask is not None else 0

        debris_mask = dyn_state.get("debris_mask")
        dc = int(debris_mask.sum()) if debris_mask is not None else 0

        nfz_mask = dyn_state.get("nfz_mask")
        nc = int(nfz_mask.sum()) if nfz_mask is not None else 0

        steps.append(step)
        fire_counts.append(fc)
        closure_counts.append(cc)
        debris_counts.append(dc)
        nfz_counts.append(nc)
        distances.append(dist)

    result = run_episode(scenario_id, planner_id, seed,
                         frame_callback=cb, config_override=cfg)

    return {
        "steps": steps,
        "fire_counts": fire_counts,
        "closure_counts": closure_counts,
        "debris_counts": debris_counts,
        "nfz_counts": nfz_counts,
        "distances": distances,
        "success": result.metrics.get("success", False),
        "termination_reason": result.metrics.get("termination_reason", ""),
    }


def main() -> None:
    print("Coupled hazard timeline (2-panel)")
    print(f"  Scenario: {SCENARIO}, seed={SEED}")

    config = load_scenario(SCENARIO)
    config_mod = replace(config, enable_collapse=True, collapse_delay=80, debris_prob=0.6)

    # Collect data for both planners
    planner_data = {}
    for pid in PLANNERS:
        print(f"  Running {pid}...", end="", flush=True)
        data = _collect_data(SCENARIO, pid, SEED, config_mod)
        planner_data[pid] = data
        print(f" success={data['success']}, steps={len(data['steps'])}")

    # Use the longer episode for the hazard panel
    longest_pid = max(PLANNERS, key=lambda p: len(planner_data[p]["steps"]))
    hd = planner_data[longest_pid]

    # --- Figure ---
    fig, (ax_hazard, ax_dist) = plt.subplots(2, 1, figsize=(3.5, 4.0),
                                              sharex=True,
                                              gridspec_kw={"hspace": 0.25})

    steps_arr = np.array(hd["steps"])
    fire_arr = np.array(hd["fire_counts"])
    closure_arr = np.array(hd["closure_counts"])
    debris_arr = np.array(hd["debris_counts"])

    # === Panel A: Stacked area chart ===
    ax_hazard.fill_between(steps_arr, 0, fire_arr, alpha=0.3, color="#D55E00",
                           label="Fire cells")
    ax_hazard.fill_between(steps_arr, fire_arr, fire_arr + closure_arr,
                           alpha=0.4, color="#E69F00", label="Road closures")
    if debris_arr.max() > 0:
        ax_hazard.fill_between(steps_arr, fire_arr + closure_arr,
                               fire_arr + closure_arr + debris_arr,
                               alpha=0.4, color="#8B5A2B", label="Debris")

    ax_hazard.plot(steps_arr, fire_arr, color="#D55E00", lw=0.8)
    ax_hazard.plot(steps_arr, fire_arr + closure_arr, color="#E69F00", lw=0.8)

    # Mark event_t1 (fire guarantee)
    event_t1 = getattr(config_mod, "event_t1", 40)
    ax_hazard.axvline(event_t1, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax_hazard.text(event_t1 + 3, ax_hazard.get_ylim()[1] * 0.85,
                   f"t\u2081={event_t1}\n(corridor fire)", fontsize=5.5,
                   color="grey", va="top")

    # Annotation showing coupling
    if closure_arr.max() > 0:
        # Find step where closures first appear
        closure_start_idx = np.argmax(closure_arr > 0)
        if closure_start_idx > 0:
            ax_hazard.annotate(
                "InteractionEngine:\nfire \u2192 road closures",
                xy=(steps_arr[closure_start_idx], closure_arr[closure_start_idx]),
                xytext=(steps_arr[closure_start_idx] + 40, fire_arr.max() * 0.6),
                fontsize=5.5, color="#E69F00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E69F00", lw=0.8),
                ha="left",
            )

    ax_hazard.set_ylabel("Hazard cells")
    ax_hazard.set_title("(a) Hazard layers grow together", fontsize=8,
                        fontweight="bold", loc="left")
    ax_hazard.legend(fontsize=6, loc="upper left")
    ax_hazard.grid(axis="x", alpha=0.12)
    ax_hazard.spines["top"].set_visible(False)
    ax_hazard.spines["right"].set_visible(False)

    # === Panel B: Agent distance for both planners ===
    for pid in PLANNERS:
        d = planner_data[pid]
        steps_p = np.array(d["steps"])
        dist_p = np.array(d["distances"])
        if len(dist_p) == 0:
            continue

        max_dist = dist_p.max() if dist_p.max() > 0 else 1
        norm_dist = dist_p / max_dist

        label = f"{PLANNER_SHORT[pid]}"
        ax_dist.plot(steps_p, norm_dist, color=PLANNER_COLORS[pid], lw=1.2,
                     label=label)

        # Mark failure point
        if not d["success"]:
            ax_dist.plot(steps_p[-1], norm_dist[-1], "X", color="red",
                         markersize=5, zorder=10)

    ax_dist.set_ylabel("Dist. to goal (norm.)")
    ax_dist.set_xlabel("Step")
    ax_dist.set_title("(b) Impact on the drone", fontsize=8,
                      fontweight="bold", loc="left")
    ax_dist.set_ylim(-0.02, 1.05)
    ax_dist.legend(fontsize=6, loc="upper right")
    ax_dist.grid(axis="x", alpha=0.12)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    fig.align_ylabels([ax_hazard, ax_dist])
    _save(fig, "coupled_hazard_timeline")


if __name__ == "__main__":
    main()
