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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_DIR = "outputs/paper_figures"
SEED = 42

# One representative scenario per family
FAMILY_SCENARIOS = {
    "fire_delivery": "osm_penteli_fire_delivery_medium",
    "flood_rescue": "osm_piraeus_flood_rescue_medium",
    "fire_surveillance": "osm_downtown_fire_surveillance_medium",
}

# For the 5-panel planner comparison
COMPARISON_SCENARIO = "osm_penteli_fire_delivery_medium"

PLANNER_ORDER = [
    "astar", "periodic_replan",
    "aggressive_replan", "dstar_lite", "apf",
]
PLANNER_LABELS = {
    "astar": "A*",
    "periodic_replan": "Periodic Replan",
    "aggressive_replan": "Aggressive Replan",
    "dstar_lite": "D* Lite",
    "apf": "APF",
}


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

    def frame_callback(
        heightmap: np.ndarray,
        state: dict,
        dyn_state: dict,
        cfg: object,
    ) -> None:
        step = state.get("step_idx", 0)
        total_steps_box[0] = step

        if capture_steps is None or step in capture_steps:
            frame, meta = renderer.render_frame(heightmap, state, dyn_state)
            frames[step] = (frame, meta)

    result = run_episode(scenario_id, planner_id, seed, frame_callback=frame_callback)

    # Always capture info about total steps
    return {
        "frames": frames,
        "metrics": result.metrics,
        "total_steps": total_steps_box[0],
    }


# ---------------------------------------------------------------------------
# Snapshot generators
# ---------------------------------------------------------------------------


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

        # Second pass: capture t=0, t=mid, t=end
        capture = {0, mid, total}
        result = _run_with_snapshots(scenario_id, "astar", SEED, capture_steps=capture)

        for label, step in [("t0", 0), ("mid", mid), ("end", total)]:
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
        capture = {0, mid, total}
        result = _run_with_snapshots(args.scenario, planner_id, SEED, capture_steps=capture)

        for label, step in [("t0", 0), ("mid", mid), ("end", total)]:
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

    if not args.skip_families:
        generate_family_snapshots()
    if not args.skip_comparison:
        generate_planner_comparison()

    print(f"\nDone. All snapshots in {FIG_DIR}/")


if __name__ == "__main__":
    main()
