#!/usr/bin/env python3
"""Stakeholder Visualization V2 — 10-minute demo script.

Runs all three mission types through the V2 pipeline and produces:
  - Console summary with replan counts, violations, metrics
  - GeoJSON/CSV exports per mission
  - Optional PNG frames via StakeholderRenderer (if --render flag)

Usage:
    python scripts/demo_stakeholder_v2.py
    python scripts/demo_stakeholder_v2.py --render   # also generate frames
    python scripts/demo_stakeholder_v2.py --outdir results/demo_v2

Timing target: <10 minutes total for all 3 missions on a laptop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure project is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uavbench.missions.runner_v2 import plan_mission_v2, MissionResultV2
from uavbench.visualization.export import export_all


# ── Mission configurations ──────────────────────────────────────────

MISSIONS = [
    {
        "label": "M1 — Civil Protection (Wildfire SA)",
        "mission_id": "civil_protection",
        "difficulty": "easy",
        "seed": 42,
    },
    {
        "label": "M2 — Maritime Domain (Coastal Patrol)",
        "mission_id": "maritime_domain",
        "difficulty": "easy",
        "seed": 42,
    },
    {
        "label": "M3 — Critical Infrastructure (Inspection)",
        "mission_id": "critical_infrastructure",
        "difficulty": "easy",
        "seed": 42,
    },
]


def _make_grid(size: int = 64):
    """Create a simple clear grid for demo purposes."""
    H, W = size, size
    heightmap = np.zeros((H, W), dtype=np.float32)
    no_fly = np.zeros((H, W), dtype=bool)
    return heightmap, no_fly


def _print_header(title: str) -> None:
    width = 60
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def _print_result(result: MissionResultV2) -> None:
    """Pretty-print a mission result summary."""
    m = result.metrics
    replan_count = max(0, len(result.replan_log) - 1)

    print(f"  Mission:      {result.mission_id}")
    print(f"  Difficulty:   {result.difficulty}")
    print(f"  Planner:      {result.planner_id}")
    print(f"  Steps:        {result.step_count}")
    print(f"  Score:        {m.get('mission_score', 0):.3f}")
    print(f"  Tasks done:   {m.get('task_completion_rate', 0):.0%}")
    print(f"  Replans:      {replan_count}")
    print(f"  Violations:   {int(m.get('violation_count', 0))}")
    print(f"  Risk ∫:       {m.get('risk_integral', 0):.2f}")
    print()

    # Replan breakdown
    triggers: dict[str, int] = {}
    for r in result.replan_log[1:]:  # skip initial
        t = r.get("trigger", "?")
        triggers[t] = triggers.get(t, 0) + 1
    if triggers:
        parts = [f"{k}={v}" for k, v in sorted(triggers.items())]
        print(f"  Replan triggers:  {', '.join(parts)}")

    # Forced replan summary
    fs = result.forced_replan_summary
    print(f"  Forced injections: {fs.get('injected', 0)}/{fs.get('scheduled', 0)}")

    # Safety summary
    ss = result.safety_summary
    print(f"  Safety violations: {ss.get('total_violations', 0)}")
    by_type = ss.get("by_type", {})
    if by_type:
        parts = [f"{k}={v}" for k, v in sorted(by_type.items())]
        print(f"    by type: {', '.join(parts)}")

    # Bus summary
    bs = result.event_bus_summary
    if bs:
        parts = [f"{k}={v}" for k, v in sorted(bs.items())]
        print(f"  Bus events: {', '.join(parts)}")

    print(f"  Trajectory length: {len(result.trajectory)} cells")
    print()


def run_demo(outdir: Path, render: bool = False) -> None:
    """Run all three missions and export results."""
    outdir.mkdir(parents=True, exist_ok=True)
    heightmap, no_fly = _make_grid(64)
    results: list[MissionResultV2] = []

    _print_header("UAVBench — Stakeholder Visualization V2 Demo")
    print(f"  Output directory: {outdir}")
    print(f"  Grid size: 64×64 (demo)")
    print(f"  Render frames: {render}")
    print()

    for cfg in MISSIONS:
        _print_header(cfg["label"])
        t0 = time.perf_counter()

        result = plan_mission_v2(
            start=(5, 5),
            heightmap=heightmap,
            no_fly=no_fly,
            mission_id=cfg["mission_id"],
            difficulty=cfg["difficulty"],
            planner_id="astar",
            seed=cfg["seed"],
            replan_cadence=10,
            min_forced_replans=2,
            forced_obstacle_radius=3,
        )

        elapsed = time.perf_counter() - t0
        _print_result(result)
        print(f"  Wall time: {elapsed:.2f}s")

        # Export
        mission_dir = outdir / cfg["mission_id"]
        files = export_all(
            result, mission_dir,
            center_latlon=(37.97, 23.73),
            resolution_m=3.0,
            grid_size=64,
        )
        print(f"  Exported {len(files)} files to {mission_dir}/")
        for f in files:
            print(f"    • {f.name}")

        # Optional render
        if render:
            _render_frames(result, mission_dir, heightmap, no_fly)

        results.append(result)

    # ── Summary ──────────────────────────────────────────────────
    _print_header("Summary")
    all_ok = True
    for r in results:
        replan_count = max(0, len(r.replan_log) - 1)
        status = "✓" if replan_count >= 2 else "✗"
        if replan_count < 2:
            all_ok = False
        print(f"  {status}  {r.mission_id:30s}  replans={replan_count}  violations={int(r.metrics.get('violation_count', 0))}")

    print()
    if all_ok:
        print("  ✓ ALL MISSIONS: ≥2 replans guaranteed")
    else:
        print("  ✗ SOME MISSIONS FAILED ≥2 replan guarantee!")
    print()


def _render_frames(
    result: MissionResultV2,
    output_dir: Path,
    heightmap: np.ndarray,
    no_fly: np.ndarray,
) -> None:
    """Render a few sample frames using StakeholderRenderer."""
    try:
        from uavbench.visualization.stakeholder_renderer import (
            StakeholderRenderer, TileData,
        )
    except ImportError:
        print("  [skip render — StakeholderRenderer not available]")
        return

    H, W = heightmap.shape
    rng = np.random.default_rng(42)
    tile = TileData(
        tile_id=f"demo_{result.mission_id}",
        heightmap=heightmap,
        roads_mask=np.zeros((H, W), dtype=bool),
        landuse_map=np.zeros((H, W), dtype=np.int8),
        risk_map=np.zeros((H, W), dtype=np.float32),
        nfz_mask=no_fly,
        center_latlon=(37.97, 23.73),
        resolution_m=3.0,
        grid_size=H,
    )

    renderer = StakeholderRenderer(
        tile=tile,
        mission_type=result.mission_id,
        scenario_id=f"demo_{result.mission_id}",
        planner_name=result.planner_id,
        difficulty=result.difficulty,
        figsize=(9.6, 5.4),
        dpi=72,
    )

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Render 3 sample frames: start, middle, end
    traj = result.trajectory
    sample_indices = [0, len(traj) // 2, len(traj) - 1]
    path = traj  # use full trajectory as "path"

    for i, idx in enumerate(sample_indices):
        pos = traj[idx]
        frame = renderer.render_frame(
            agent_pos=pos,
            path=path[:idx + 1],
            step=idx,
            metrics={
                "score": result.metrics.get("mission_score", 0),
                "Replans": max(0, len(result.replan_log) - 1),
                "Violations": int(result.metrics.get("violation_count", 0)),
            },
        )
        out_path = frames_dir / f"frame_{i:03d}.png"
        frame.savefig(str(out_path), dpi=72, bbox_inches="tight")
        print(f"    Rendered: {out_path.name}")

    renderer.close()


def main():
    parser = argparse.ArgumentParser(
        description="UAVBench Stakeholder V2 Demo",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/demo_v2"),
        help="Output directory for exports (default: results/demo_v2)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render PNG frames (requires matplotlib)",
    )
    args = parser.parse_args()

    run_demo(args.outdir, render=args.render)


if __name__ == "__main__":
    main()
