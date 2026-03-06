#!/usr/bin/env python3
"""Export all v2 evidence artifacts.

Generates:
  - outputs/determinism_hashes.json  (DC-2 verification)
  - outputs/viz_manifest.csv         (VC rendering manifest)
  - outputs/viz_frame_checks.json    (VC per-frame validation)
  - outputs/repro_manifest.json      (reproducibility manifest)

Usage:
    python scripts/export_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uavbench.benchmark.runner import run_episode  # noqa: E402
from uavbench.scenarios.registry import list_scenarios  # noqa: E402
from uavbench.planners import PLANNERS  # noqa: E402
from uavbench.visualization.renderer import Renderer  # noqa: E402
from uavbench.scenarios.loader import load_scenario  # noqa: E402

OUT_DIR = ROOT / "outputs"


def _hash_obj(obj: object) -> str:
    """SHA-256 of JSON-serialized object."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def export_determinism_hashes() -> dict:
    """Run two identical episodes and verify bit-identical outputs (DC-2)."""
    scenario_id = "osm_penteli_fire_delivery_medium"
    planner_id = "astar"
    seed = 42

    result_a = run_episode(scenario_id, planner_id, seed)
    result_b = run_episode(scenario_id, planner_id, seed)

    def _hashes(r):
        return {
            "events": _hash_obj(r.events),
            "trajectory": _hash_obj(r.trajectory),
            "metrics": _hash_obj(r.metrics),
            "full": _hash_obj({
                "events": r.events,
                "trajectory": r.trajectory,
                "metrics": r.metrics,
            }),
        }

    ha = _hashes(result_a)
    hb = _hashes(result_b)

    doc = {
        "contract": "DC-2",
        "scenario": scenario_id,
        "planner": planner_id,
        "seed": seed,
        "run_a": ha,
        "run_b": hb,
        "all_match": ha == hb,
        "trajectory_length": len(result_a.trajectory),
        "event_count": len(result_a.events),
    }

    path = OUT_DIR / "determinism_hashes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

    status = "PASS" if doc["all_match"] else "FAIL"
    print(f"  DC-2 determinism: {status} (traj_len={doc['trajectory_length']})")
    return doc


def export_viz_artifacts() -> None:
    """Generate viz_manifest.csv and viz_frame_checks.json."""
    import numpy as np
    from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig

    config = ScenarioConfig(
        name="repro_viz_test",
        mission_type=MissionType.FIRE_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=20,
        building_density=0.1,
        max_episode_steps=50,
        terminate_on_collision=False,
    )
    renderer = Renderer(config, mode="ops_full")
    heightmap = np.zeros((20, 20), dtype=np.float32)

    manifest_rows = []
    frame_checks = []

    for step in range(5):
        plan_path = [(5 + i, 5 + i) for i in range(10)] if step > 0 else []
        state = {
            "plan_len": len(plan_path),
            "plan_path": plan_path,
            "plan_age_steps": 0,
            "plan_reason": "",
            "step_idx": step,
            "agent_xy": (5, 5),
            "goal_xy": (15, 15),
            "mission_domain": "fire_delivery",
            "objective_label": "Emergency Medical Supply Delivery",
            "distance_to_task": 20.0,
            "task_progress": "0/1",
            "deliverable_name": "medical_supplies",
            "trajectory": [(5, 5)],
            "scenario_id": "repro_viz_test",
            "planner_name": "astar",
            "replan_every_steps": 6,
            "replans": 0,
            "dynamic_block_hits": 0,
        }
        frame, meta = renderer.render_frame(heightmap, state)

        manifest_rows.append({
            "step": step,
            "mode": "ops_full",
            "width": frame.shape[1],
            "height": frame.shape[0],
            "path_rendered": meta["path_rendered"],
            "plan_badge": meta["plan_badge"],
            "block_badge": meta["block_badge"],
        })

        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        has_cyan = bool(((r < 150) & (g > 150) & (b > 200)).any()) if meta["path_rendered"] else False

        frame_checks.append({
            "step": step,
            "has_cyan_pixels": has_cyan,
            "plan_badge": meta["plan_badge"],
            "block_badge": meta["block_badge"],
        })

    # Write manifest
    csv_path = OUT_DIR / "viz_manifest.csv"
    with open(csv_path, "w") as f:
        f.write("step,mode,width,height,path_rendered,plan_badge,block_badge\n")
        for row in manifest_rows:
            f.write(
                f"{row['step']},{row['mode']},{row['width']},"
                f"{row['height']},{row['path_rendered']},"
                f"\"{row['plan_badge']}\",\"{row['block_badge']}\"\n"
            )

    # Write frame checks
    json_path = OUT_DIR / "viz_frame_checks.json"
    with open(json_path, "w") as f:
        json.dump(frame_checks, f, indent=2)

    print(f"  viz_manifest.csv: {len(manifest_rows)} rows")
    print(f"  viz_frame_checks.json: {len(frame_checks)} entries")


def export_repro_manifest() -> None:
    """Write repro_manifest.json — one command to regenerate everything."""
    scenarios = list_scenarios()
    planners = list(PLANNERS.keys())

    manifest = {
        "version": "v2",
        "regenerate_command": "python scripts/export_artifacts.py",
        "test_command": "pytest tests/ -q",
        "gif_command": "bash scripts/regenerate_paper_gifs.sh",
        "scenarios": scenarios,
        "planners": planners,
        "scenario_count": len(scenarios),
        "planner_count": len(planners),
        "artifacts": [
            "outputs/determinism_hashes.json",
            "outputs/viz_manifest.csv",
            "outputs/viz_frame_checks.json",
            "outputs/repro_manifest.json",
            "outputs/rebuild_audit.json",
        ],
        "contracts": [
            "DC-1", "DC-2", "FC-1", "FC-2", "EC-1", "EC-2",
            "GC-1", "GC-2", "EV-1", "VC-1", "VC-2", "VC-3",
            "MC-1", "MC-2", "MC-3", "MC-4", "PC-1", "PC-2",
            "MP-1", "RS-1",
        ],
    }

    path = OUT_DIR / "repro_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  repro_manifest.json: {manifest['scenario_count']} scenarios, "
          f"{manifest['planner_count']} planners, "
          f"{len(manifest['contracts'])} contracts")


def main() -> None:
    """Export all v2 evidence artifacts."""
    print("=== UAVBench v2 Artifact Export ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    print("\n1. Determinism hashes (DC-2)...")
    det = export_determinism_hashes()

    print("\n2. Visualization artifacts (VC-1/2/3)...")
    export_viz_artifacts()

    print("\n3. Reproducibility manifest...")
    export_repro_manifest()

    elapsed = time.perf_counter() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")

    if not det["all_match"]:
        print("WARNING: Determinism check FAILED!")
        sys.exit(1)

    print("All artifacts exported successfully.")


if __name__ == "__main__":
    main()
