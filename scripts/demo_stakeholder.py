#!/usr/bin/env python3
"""Stakeholder demo — run all 3 missions on real OSM tiles and export demo packs.

Produces cinema-quality 1080p visualisations for three Greek government agencies:
  - ΓΓΠΠ Civil Protection: wildfire monitoring on Penteli WUI
  - ΛΣ-ΕΛΑΚΤ Coast Guard: maritime SAR on Piraeus port
  - ΥΠΕΘΑ ISR-support: critical infrastructure on Athens downtown

Usage::

    # All missions, medium difficulty
    python scripts/demo_stakeholder.py

    # Single mission
    python scripts/demo_stakeholder.py --mission civil_protection

    # Custom output
    python scripts/demo_stakeholder.py --output demo_output --difficulty hard

    # Quick preview (fewer steps)
    python scripts/demo_stakeholder.py --preview
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavbench.visualization.stakeholder_renderer import (
    StakeholderRenderer,
    TileData,
    load_tile,
    MISSION_PROFILES,
)
from uavbench.visualization.overlays import create_overlay
from uavbench.visualization.demo_pack import export_demo_pack
from uavbench.visualization.icons import IconID


# ─────────────────────────────────────────────────────────────────────────────
# Demo scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

DEMO_SCENARIOS = {
    "civil_protection": {
        "tile_id": "penteli",
        "mission_type": "civil_protection",
        "planner": "astar",
        "max_steps": 200,
        "start": (50, 50),
        "poi_positions": [
            (120, 180), (200, 300), (350, 150), (400, 400), (280, 250),
        ],
        "fire_origins": [(100, 400), (150, 380)],
        "overlay_kwargs": {
            "fire_spread_rate": 0.3,
            "wind_direction_deg": 225.0,
            "wind_speed": 0.4,
        },
    },
    "maritime_domain": {
        "tile_id": "piraeus",
        "mission_type": "maritime_domain",
        "planner": "astar",
        "max_steps": 200,
        "start": (250, 250),
        "poi_positions": [
            (100, 100), (150, 400), (350, 350), (400, 100), (250, 300),
        ],
        "overlay_kwargs": {
            "patrol_radius": 150.0,
            "num_vessels": 4,
        },
    },
    "critical_infrastructure": {
        "tile_id": "downtown",
        "mission_type": "critical_infrastructure",
        "planner": "astar",
        "max_steps": 200,
        "start": (250, 250),
        "poi_positions": [
            (100, 200), (200, 400), (350, 300), (400, 150), (300, 100),
        ],
        "inspection_sites": [
            {"xy": (100, 200), "label": "Site A", "radius": 20},
            {"xy": (350, 300), "label": "Site B", "radius": 25},
            {"xy": (200, 400), "label": "Site C", "radius": 15},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Simple path planner (for demo purposes — greedy waypoint tour)
# ─────────────────────────────────────────────────────────────────────────────

def _plan_greedy_tour(
    start: tuple[int, int],
    pois: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour tour through POIs.  Returns full path."""
    remaining = list(pois)
    path = [start]
    current = start

    while remaining:
        # Find nearest
        dists = [abs(p[0] - current[0]) + abs(p[1] - current[1]) for p in remaining]
        idx = int(np.argmin(dists))
        target = remaining.pop(idx)

        # Interpolate straight line (for visualisation)
        steps = max(abs(target[0] - current[0]), abs(target[1] - current[1]), 1)
        for i in range(1, steps + 1):
            t = i / steps
            x = int(current[0] + t * (target[0] - current[0]))
            y = int(current[1] + t * (target[1] - current[1]))
            path.append((x, y))

        current = target

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(
    mission_key: str,
    difficulty: str = "medium",
    output_dir: Path | None = None,
    preview: bool = False,
) -> Path:
    """Run a single demo scenario and export the demo pack.

    Parameters
    ----------
    mission_key : str
        One of "civil_protection", "maritime_domain", "critical_infrastructure".
    difficulty : str
        "easy", "medium", or "hard".
    output_dir : Path, optional
        Override output directory.
    preview : bool
        If True, run fewer steps for quick preview.

    Returns
    -------
    Path
        Demo pack output directory.
    """
    scenario = DEMO_SCENARIOS[mission_key]
    tile_id = scenario["tile_id"]
    mission_type = scenario["mission_type"]
    max_steps = 50 if preview else scenario["max_steps"]

    print(f"\n{'='*60}")
    print(f"  Demo: {MISSION_PROFILES[mission_type].name}")
    print(f"  Tile: {tile_id}  |  Difficulty: {difficulty}")
    print(f"  Steps: {max_steps}")
    print(f"{'='*60}\n")

    # ── Load tile ──
    print(f"  Loading tile {tile_id}...")
    tile = load_tile(tile_id)
    H, W = tile.heightmap.shape
    print(f"  Tile loaded: {H}×{W} @ {tile.resolution_m} m/pixel")

    # ── Create renderer ──
    scenario_id = f"demo_{mission_type}_{difficulty}"
    renderer = StakeholderRenderer(
        tile=tile,
        mission_type=mission_type,
        scenario_id=scenario_id,
        planner_name=scenario["planner"],
        difficulty=difficulty,
    )

    # ── Create overlay ──
    overlay_kwargs = scenario.get("overlay_kwargs", {})
    if mission_type == "civil_protection":
        overlay_kwargs["fire_origins"] = scenario.get("fire_origins", [(H // 4, W // 4)])
    elif mission_type == "critical_infrastructure":
        overlay_kwargs["inspection_sites"] = scenario.get("inspection_sites", [])

    overlay = create_overlay(
        mission_type,
        grid_shape=(H, W),
        **overlay_kwargs,
    )

    # ── Plan path ──
    start = scenario["start"]
    poi_positions = scenario["poi_positions"]
    full_path = _plan_greedy_tour(start, poi_positions)

    # ── Build POI list for rendering ──
    mission_icons = {
        "civil_protection": IconID.FIRE,
        "maritime_domain": IconID.ANCHOR,
        "critical_infrastructure": IconID.BUILDING,
    }
    poi_icon = mission_icons.get(mission_type, IconID.WAYPOINT)

    # ── Run episode ──
    print("  Rendering frames...")
    t0 = time.time()

    for step in range(max_steps):
        # Current position along path
        path_idx = min(step * len(full_path) // max_steps, len(full_path) - 1)
        drone_pos = full_path[path_idx]

        # Compute overlay
        overlay_data = overlay.compute(step)

        # Build POIs with status
        pois = []
        for i, pos in enumerate(poi_positions):
            # Mark as completed if we've passed this POI
            poi_path_idx = 0
            for pi, pp in enumerate(full_path):
                if abs(pp[0] - pos[0]) <= 3 and abs(pp[1] - pos[1]) <= 3:
                    poi_path_idx = pi
                    break
            status = "completed" if path_idx >= poi_path_idx and poi_path_idx > 0 else "pending"
            if path_idx >= poi_path_idx - 10 and path_idx < poi_path_idx and poi_path_idx > 0:
                status = "active"
            pois.append({
                "xy": pos,
                "icon": poi_icon,
                "label": f"P{i+1}",
                "status": status,
            })

        # Merge with overlay POIs
        all_pois = pois + overlay_data.get("pois", [])

        # Heading from path tangent
        if path_idx > 0:
            dx = drone_pos[0] - full_path[path_idx - 1][0]
            dy = drone_pos[1] - full_path[path_idx - 1][1]
            heading = np.degrees(np.arctan2(-dy, dx)) if (dx != 0 or dy != 0) else 0.0
        else:
            heading = 0.0

        # Metrics
        completed = sum(1 for p in pois if p["status"] == "completed")
        metrics = {
            "tasks_completed": completed,
            "tasks_pending": len(poi_positions) - completed,
            "replans": 0,
            "risk_integral": step * 0.01,
            "energy_used": step / max_steps * 100,
            "mission_score": completed / max(len(poi_positions), 1),
        }

        # Log events at POI arrivals
        for p in pois:
            if p["status"] == "active":
                renderer.log_event(step, "task_complete", f"Reached {p['label']}")

        # Keyframe at events
        is_keyframe = step == 0 or step == max_steps - 1 or (step % 50 == 0)

        # Render
        renderer.render_frame(
            drone_pos=drone_pos,
            step=step,
            max_steps=max_steps,
            fire_mask=overlay_data.get("fire_mask"),
            smoke_mask=overlay_data.get("smoke_mask"),
            nfz_mask=overlay_data.get("nfz_mask"),
            pois=all_pois,
            entity_positions=overlay_data.get("entity_positions"),
            trajectory=full_path[:path_idx + 1],
            planned_path=full_path[path_idx:],
            heading_deg=heading,
            distress_position=overlay_data.get("distress_position"),
            metrics=metrics,
            is_keyframe=is_keyframe,
        )

        if step % 20 == 0:
            elapsed = time.time() - t0
            fps = (step + 1) / max(elapsed, 0.01)
            print(f"    Step {step}/{max_steps}  ({fps:.1f} FPS)")

    elapsed = time.time() - t0
    print(f"  Rendering complete: {max_steps} frames in {elapsed:.1f}s "
          f"({max_steps/max(elapsed, 0.01):.1f} FPS)")

    # ── Export demo pack ──
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "demo_packs" / f"{mission_key}_{difficulty}"

    print(f"  Exporting demo pack to {output_dir}...")
    pack_dir = export_demo_pack(
        renderer=renderer,
        output_dir=output_dir,
        metrics=metrics,
    )
    print(f"  ✓ Demo pack exported: {pack_dir}")

    renderer.close()
    return pack_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stakeholder demo packs for Greek government agencies",
    )
    parser.add_argument(
        "--mission",
        choices=list(DEMO_SCENARIOS.keys()),
        default=None,
        help="Run a specific mission (default: all)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Quick preview mode (50 steps)",
    )
    args = parser.parse_args()

    missions = [args.mission] if args.mission else list(DEMO_SCENARIOS.keys())

    print(f"\n{'='*60}")
    print(f"  UAVBench Stakeholder Demo Generator")
    print(f"  Missions: {', '.join(missions)}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"{'='*60}")

    for mission_key in missions:
        out = args.output / mission_key if args.output else None
        run_demo(
            mission_key,
            difficulty=args.difficulty,
            output_dir=out,
            preview=args.preview,
        )

    print(f"\n{'='*60}")
    print(f"  All demos complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
