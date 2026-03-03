#!/usr/bin/env python3
"""Generate paper-quality visualization snapshots.

For each of the 3 scenario families:
  - Representative episode snapshots at t=0, t=mid, t=end
For one hard scenario:
  - 6-panel figure showing all planner trajectories on the same map

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

from uavbench.envs.urban import UrbanEnvV2
from uavbench.planners import PLANNERS
from uavbench.planners.base import PlanResult
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.schema import ScenarioConfig
from uavbench.visualization.renderer import Renderer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_DIR = "outputs/paper_figures"
SEED = 42

# One representative scenario per family (hard = most visual dynamics)
FAMILY_SCENARIOS = {
    "fire_delivery": "gov_fire_delivery_hard",
    "flood_rescue": "gov_flood_rescue_hard",
    "fire_surveillance": "gov_fire_surveillance_hard",
}

# For the 6-panel planner comparison
COMPARISON_SCENARIO = "gov_fire_delivery_hard"

PLANNER_ORDER = [
    "astar", "theta_star", "periodic_replan",
    "aggressive_replan", "dstar_lite", "mppi_grid",
]
PLANNER_LABELS = {
    "astar": "A*",
    "theta_star": r"$\theta$*",
    "periodic_replan": "Periodic Replan",
    "aggressive_replan": "Aggressive Replan",
    "dstar_lite": "D* Lite",
    "mppi_grid": "MPPI",
}


# ---------------------------------------------------------------------------
# Episode runner with frame capture
# ---------------------------------------------------------------------------


def _run_episode_with_frames(
    config: ScenarioConfig,
    planner_id: str,
    seed: int,
    capture_steps: set[int] | None = None,
    capture_all: bool = False,
) -> dict[str, Any]:
    """Run a full episode and capture frames at specified timesteps.

    Returns:
        Dict with keys:
            frames: dict mapping step_idx -> (frame, meta)
            trajectory: full trajectory
            events: all events
            final_info: last info dict
            total_steps: number of steps taken
            plan_path: final plan path (for overlay)
    """
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)

    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()
    renderer = Renderer(config, mode="paper_min")

    planner_cls = PLANNERS[planner_id]
    planner = planner_cls(heightmap, no_fly, config)
    plan_result = planner.plan(start_xy, goal_xy)

    path = plan_result.path if plan_result.success else []
    path_idx = 0
    trajectory = [start_xy]
    frames: dict[int, tuple[np.ndarray, dict]] = {}

    # Capture t=0
    if capture_all or (capture_steps is not None and 0 in capture_steps):
        state_0 = _build_render_state(
            info, trajectory, path, planner_id, step_idx=0,
        )
        dyn_state_0 = env.get_dynamic_state()
        frame_0, meta_0 = renderer.render_frame(heightmap, state_0, dyn_state_0)
        frames[0] = (frame_0, meta_0)

    step_idx = 0
    terminated = False
    truncated = False
    final_info = info

    while not terminated and not truncated:
        step_idx += 1
        action = _path_to_action(env.agent_xy, path, path_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        trajectory.append(env.agent_xy)

        if path_idx < len(path) - 1:
            next_wp = path[path_idx + 1]
            if env.agent_xy == next_wp:
                path_idx += 1

        dyn_state = env.get_dynamic_state()
        planner.update(dyn_state)

        should, reason = planner.should_replan(
            env.agent_xy, path, dyn_state, step_idx,
        )
        if should:
            new_result = planner.plan(env.agent_xy, goal_xy)
            if new_result.success:
                path = new_result.path
                path_idx = 0

        # Capture frame if requested
        if capture_all or (capture_steps is not None and step_idx in capture_steps):
            state = _build_render_state(
                info, trajectory, path, planner_id, step_idx,
            )
            frame, meta = renderer.render_frame(heightmap, state, dyn_state)
            frames[step_idx] = (frame, meta)

    # Always capture final frame
    if step_idx not in frames:
        dyn_state = env.get_dynamic_state()
        state = _build_render_state(
            final_info, trajectory, path, planner_id, step_idx,
        )
        frame, meta = renderer.render_frame(heightmap, state, dyn_state)
        frames[step_idx] = (frame, meta)

    return {
        "frames": frames,
        "trajectory": trajectory,
        "events": env.events,
        "final_info": final_info,
        "total_steps": step_idx,
        "plan_path": path,
        "heightmap": heightmap,
    }


def _build_render_state(
    info: dict,
    trajectory: list[tuple[int, int]],
    plan_path: list[tuple[int, int]],
    planner_id: str,
    step_idx: int,
) -> dict[str, Any]:
    """Build the state dict expected by Renderer.render_frame."""
    return {
        "agent_xy": info.get("agent_xy", (0, 0)),
        "goal_xy": info.get("goal_xy", (0, 0)),
        "trajectory": list(trajectory),
        "plan_path": list(plan_path),
        "plan_len": len(plan_path),
        "plan_age_steps": 0,
        "plan_reason": "",
        "replan_every_steps": 6,
        "forced_block_active": info.get("forced_block_active", False),
        "forced_block_lifecycle": info.get("forced_block_lifecycle", "none"),
        "scenario_id": "",
        "mission_domain": info.get("mission_domain", ""),
        "planner_name": planner_id,
        "mode": "snapshot",
        "step_idx": step_idx,
        "replans": 0,
        "objective_label": info.get("objective_label", ""),
        "distance_to_task": info.get("distance_to_task", 0),
        "task_progress": info.get("task_progress", ""),
        "deliverable_name": info.get("deliverable_name", ""),
    }


def _path_to_action(
    agent_xy: tuple[int, int],
    path: list[tuple[int, int]],
    path_idx: int,
) -> int:
    """Convert path waypoint to action integer (mirrors runner logic)."""
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY = 0, 1, 2, 3, 4

    if not path:
        return ACTION_STAY

    target_idx = path_idx + 1 if path_idx + 1 < len(path) else len(path) - 1
    target = path[target_idx]
    ax, ay = agent_xy
    tx, ty = target

    if (ax, ay) == (tx, ty):
        if target_idx + 1 < len(path):
            tx, ty = path[target_idx + 1]
        else:
            return ACTION_STAY

    dx, dy = tx - ax, ty - ay
    if dx == 0 and dy == 0:
        return ACTION_STAY
    if abs(dx) >= abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    return ACTION_DOWN if dy > 0 else ACTION_UP


# ---------------------------------------------------------------------------
# Snapshot generators
# ---------------------------------------------------------------------------


def generate_family_snapshots() -> None:
    """For each scenario family: snapshots at t=0, t=mid, t=end."""
    print("Generating family snapshots...")

    for family, scenario_id in FAMILY_SCENARIOS.items():
        print(f"\n  Family: {family} ({scenario_id})")
        config = load_scenario(scenario_id)

        # First pass: run to find total steps
        result = _run_episode_with_frames(
            config, "astar", SEED, capture_steps={0},
        )
        total = result["total_steps"]
        mid = max(1, total // 2)

        print(f"    Total steps: {total}, mid: {mid}")

        # Second pass: capture t=0, t=mid, t=end
        capture = {0, mid, total}
        result = _run_episode_with_frames(
            config, "astar", SEED, capture_steps=capture,
        )

        for label, step in [("t0", 0), ("mid", mid), ("end", total)]:
            # Find closest captured frame
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
    """6-panel figure: all planner trajectories on same map for one scenario."""
    print(f"\nGenerating 6-panel planner comparison ({COMPARISON_SCENARIO})...")
    config = load_scenario(COMPARISON_SCENARIO)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for ax, pid in zip(axes_flat, PLANNER_ORDER):
        print(f"  Running {pid}...")
        result = _run_episode_with_frames(
            config, pid, SEED, capture_steps=set(),
        )

        # Get the final frame (always captured)
        final_step = result["total_steps"]
        if final_step in result["frames"]:
            frame, meta = result["frames"][final_step]
        else:
            frame, meta = list(result["frames"].values())[-1]

        ax.imshow(frame)
        tr = result["final_info"].get("termination_reason", "?")
        tr_str = tr.value if hasattr(tr, "value") else str(tr)
        steps = result["total_steps"]
        ax.set_title(
            f"{PLANNER_LABELS[pid]}  [{tr_str}, {steps} steps]",
            fontsize=11, fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        f"Planner Comparison — {COMPARISON_SCENARIO.replace('gov_', '').replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "planner_comparison_6panel")


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
        help="Skip 6-panel planner comparison generation",
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
        # Quick single-scenario mode
        planner_id = args.planner or "astar"
        print(f"Quick snapshot: {args.scenario} / {planner_id} / seed={SEED}")
        config = load_scenario(args.scenario)
        result = _run_episode_with_frames(
            config, planner_id, SEED, capture_steps={0},
        )
        total = result["total_steps"]
        mid = max(1, total // 2)
        capture = {0, mid, total}
        result = _run_episode_with_frames(
            config, planner_id, SEED, capture_steps=capture,
        )
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
