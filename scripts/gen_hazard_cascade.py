#!/usr/bin/env python3
"""Generate hazard cascade timeline: fire → collapse → debris coupling.

Shows how the three dynamics layers interact over time in a single episode:
- Fire cell count (growing)
- Collapse events (triggered when buildings burn long enough)
- Debris cell count (permanent, accumulating)
- Agent distance-to-goal (showing impact on navigation)

Output: paper/figures/hazard_cascade.pdf (IEEE column-width)
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario

OUTPUT_DIR = "paper/figures"
DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"
DEFAULT_PLANNER = "aggressive_replan"
DEFAULT_SEED = 42

IEEE_COL_WIDTH = 3.5


def collect_hazard_data(
    scenario_id: str,
    planner_id: str,
    seed: int,
) -> dict[str, Any]:
    """Run episode and collect per-step hazard layer counts."""
    steps: list[int] = []
    fire_counts: list[int] = []
    debris_counts: list[int] = []
    smoke_counts: list[int] = []
    distances: list[float] = []
    collapse_steps: list[int] = []

    _prev_debris = 0

    def frame_callback(heightmap, state, dyn_state, cfg):
        nonlocal _prev_debris

        step = state.get("step_idx", 0)
        agent = state.get("agent_xy", (0, 0))
        goal = state.get("goal_xy", (0, 0))
        dist = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])

        fire_mask = dyn_state.get("fire_mask")
        fire_count = int(fire_mask.sum()) if fire_mask is not None else 0

        debris_mask = dyn_state.get("debris_mask")
        debris_count = int(debris_mask.sum()) if debris_mask is not None else 0

        # smoke_mask is a float array [0,1]; count cells >= 0.5
        smoke_mask = dyn_state.get("smoke_mask")
        smoke_count = int((smoke_mask >= 0.5).sum()) if smoke_mask is not None else 0

        # Detect collapse events (debris count jumps)
        if debris_count > _prev_debris:
            collapse_steps.append(step)
        _prev_debris = debris_count

        steps.append(step)
        fire_counts.append(fire_count)
        debris_counts.append(debris_count)
        smoke_counts.append(smoke_count)
        distances.append(dist)

    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)

    return {
        "steps": steps,
        "fire_counts": fire_counts,
        "debris_counts": debris_counts,
        "smoke_counts": smoke_counts,
        "distances": distances,
        "collapse_steps": collapse_steps,
        "metrics": result.metrics,
    }


def plot_hazard_cascade(data: dict[str, Any], output_path: str) -> None:
    """Create 3-panel hazard cascade figure."""
    steps = data["steps"]
    fire = np.array(data["fire_counts"])
    debris = np.array(data["debris_counts"])
    smoke = np.array(data["smoke_counts"])
    dist = data["distances"]
    collapse_steps = data["collapse_steps"]

    fig, axes = plt.subplots(3, 1, figsize=(IEEE_COL_WIDTH, 4.5),
                              sharex=True, gridspec_kw={"hspace": 0.35})

    # Color palette
    C_FIRE = "#D55E00"
    C_SMOKE = "#E69F00"
    C_DEBRIS = "#CC79A7"
    C_AGENT = "#0072B2"
    C_ANNOT = "#555555"

    # --- Panel A: Fire + Smoke ---
    ax0 = axes[0]
    ax0.fill_between(steps, fire, alpha=0.25, color=C_FIRE)
    ax0.plot(steps, fire, color=C_FIRE, linewidth=1.0, label="Fire")
    ax0.set_ylabel("Fire cells", fontsize=7.5, color=C_FIRE)
    ax0.tick_params(axis="y", labelsize=6.5, colors=C_FIRE)
    ax0.tick_params(axis="x", labelsize=6.5)

    # Smoke on twin axis only if non-zero
    if smoke.max() > 0:
        ax0t = ax0.twinx()
        ax0t.fill_between(steps, smoke, alpha=0.15, color=C_SMOKE)
        ax0t.plot(steps, smoke, color=C_SMOKE, linewidth=0.8, linestyle="--")
        ax0t.set_ylabel("Smoke cells", fontsize=7.5, color=C_SMOKE, labelpad=2)
        ax0t.tick_params(axis="y", labelsize=5.5, colors=C_SMOKE, pad=1)

    ax0.set_title("(a) Fire spread", fontsize=8,
                   fontweight="bold", loc="left", pad=2)

    # Mark first collapse with arrow annotation
    if collapse_steps:
        first_c = collapse_steps[0]
        fire_at_c = fire[steps.index(first_c)] if first_c in steps else 0
        ax0.annotate(
            f"1st collapse\n(t={first_c})",
            xy=(first_c, fire_at_c), xytext=(first_c + 40, fire_at_c * 0.55),
            fontsize=6, color=C_ANNOT,
            arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.8),
            ha="left", va="center",
        )
        ax0.axvline(first_c, color=C_ANNOT, linewidth=0.6, linestyle=":", alpha=0.5)
        # Draw the collapse line through all panels
        for ax in axes[1:]:
            ax.axvline(first_c, color=C_ANNOT, linewidth=0.6, linestyle=":", alpha=0.3)

    # --- Panel B: Debris accumulation ---
    ax1 = axes[1]
    ax1.fill_between(steps, debris, alpha=0.3, color=C_DEBRIS)
    ax1.plot(steps, debris, color=C_DEBRIS, linewidth=1.2)
    ax1.set_ylabel("Debris cells", fontsize=7.5, color=C_DEBRIS)
    ax1.tick_params(axis="y", labelsize=6.5, colors=C_DEBRIS)
    ax1.tick_params(axis="x", labelsize=6.5)
    ax1.set_title(r"(b) Structural collapse $\rightarrow$ permanent debris",
                   fontsize=8, fontweight="bold", loc="left", pad=2)

    # Annotate collapse count (no individual lines — too many)
    if collapse_steps:
        ax1.text(0.97, 0.82, f"{len(collapse_steps)} collapses",
                 transform=ax1.transAxes, fontsize=6.5, ha="right",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=C_DEBRIS,
                           alpha=0.15, edgecolor="none"))
        # Annotate the inflection point where debris ramp steepens
        if len(debris) > 100:
            # Find steepest growth: max derivative
            deriv = np.diff(debris)
            peak_idx = int(np.argmax(deriv)) + 1
            if peak_idx < len(steps):
                ax1.annotate(
                    "peak growth",
                    xy=(steps[peak_idx], debris[peak_idx]),
                    xytext=(steps[peak_idx] - 60, debris[peak_idx] * 0.6),
                    fontsize=6, color=C_ANNOT,
                    arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.8),
                    ha="center", va="center",
                )

    # --- Panel C: Agent distance to goal ---
    ax2 = axes[2]
    max_dist = max(dist) if dist else 1
    norm_dist = [d / max_dist for d in dist]
    ax2.plot(steps, norm_dist, color=C_AGENT, linewidth=1.0)
    ax2.fill_between(steps, norm_dist, alpha=0.12, color=C_AGENT)
    ax2.set_ylabel("Dist. to goal\n(normalized)", fontsize=7.5)
    ax2.set_xlabel("Step", fontsize=8)
    ax2.tick_params(axis="both", labelsize=6.5)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("(c) Navigation impact", fontsize=8,
                   fontweight="bold", loc="left", pad=2)

    # Success/fail badge
    success = data["metrics"].get("success", False)
    status = "SUCCESS" if success else "TIMEOUT"
    ax2.text(0.97, 0.82, status, transform=ax2.transAxes, fontsize=6.5, ha="right",
             fontweight="bold",
             color="#009E73" if success else C_FIRE,
             bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="#009E73" if success else C_FIRE,
                       alpha=0.12, edgecolor="none"))

    # Detect plateau in distance (agent detour) and annotate
    norm_arr = np.array(norm_dist)
    if len(norm_arr) > 50:
        # Find where distance stops decreasing temporarily (detour)
        window = 20
        for i in range(window, len(norm_arr) - window):
            # If distance increases for a stretch
            if (norm_arr[i] > norm_arr[i - window] + 0.05
                    and norm_arr[i] > norm_arr[i + window]):
                ax2.annotate(
                    "detour",
                    xy=(steps[i], norm_arr[i]),
                    xytext=(steps[i] + 30, min(norm_arr[i] + 0.15, 0.95)),
                    fontsize=6, color=C_ANNOT,
                    arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.8),
                    ha="left",
                )
                break

    # Clean up spines
    for ax in axes:
        ax.grid(axis="x", alpha=0.12, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.align_ylabels(axes)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    png_path = output_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hazard cascade figure.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--planner", default=DEFAULT_PLANNER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "hazard_cascade.pdf"))
    args = parser.parse_args()

    print(f"Running {args.planner} on {args.scenario} (seed {args.seed})...")
    data = collect_hazard_data(args.scenario, args.planner, args.seed)

    print(f"Fire peak: {max(data['fire_counts'])} cells")
    print(f"Debris final: {data['debris_counts'][-1]} cells")
    print(f"Smoke peak: {max(data['smoke_counts'])} cells")
    print(f"Collapse events: {len(data['collapse_steps'])}")
    print(f"Success: {data['metrics'].get('success', False)}")

    plot_hazard_cascade(data, args.output)


if __name__ == "__main__":
    main()
