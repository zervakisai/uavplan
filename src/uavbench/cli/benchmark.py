from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario


# ----------------- Scenario path resolution -----------------

def scenario_path(scenario_id: str) -> Path:
    # src/uavbench/cli/benchmark.py -> parents[1] = src/uavbench
    base = Path(__file__).resolve().parents[1]
    return base / "scenarios" / "configs" / f"{scenario_id}.yaml"


# ----------------- Planner run -----------------

def run_planner_once(
    scenario_id: str,
    planner_id: str,
    *,
    seed: int,
) -> dict[str, Any]:
    cfg = load_scenario(scenario_path(scenario_id))
    env = UrbanEnv(cfg)

    env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")

    planner = PLANNERS[planner_id](heightmap, no_fly)
    path = planner.plan(start_xy, goal_xy)  # [] if no path

    has_path = bool(path)
    violations = 0

    # Validate path if it exists
    if has_path:
        if path[0] != start_xy or path[-1] != goal_xy:
            violations += 1

        H, W = heightmap.shape
        for (x, y) in path:
            if not (0 <= x < W and 0 <= y < H):
                violations += 1
                break
            if bool(no_fly[y, x]):
                violations += 1
                break
            # A* baseline: buildings are obstacles
            if float(heightmap[y, x]) > 0.0:
                violations += 1
                break

    success = bool(has_path and violations == 0)

    return {
        "scenario": scenario_id,
        "planner": planner_id,
        "seed": int(seed),
        "success": success,
        "constraint_violations": int(violations),
        "path_length": int(len(path)) if success else 0,
        "path": path if success else None,
        "heightmap": heightmap,
        "no_fly": no_fly,
        "start": start_xy,
        "goal": goal_xy,
    }


# ----------------- Metrics aggregation -----------------

def aggregate(results: list[dict[str, Any]], metric_ids: list[str]) -> dict[str, float]:
    """Aggregate per-trial results into metrics."""
    out: dict[str, float] = {}

    successes = np.array([1.0 if r["success"] else 0.0 for r in results], dtype=float)

    if "success_rate" in metric_ids:
        out["success_rate"] = float(successes.mean()) if len(successes) else 0.0

    if "path_length" in metric_ids:
        lengths = np.array([float(r["path_length"]) for r in results if r["success"]], dtype=float)
        out["avg_path_length"] = float(lengths.mean()) if len(lengths) else float("nan")

    if "constraint_violations" in metric_ids:
        v = np.array([float(r["constraint_violations"]) for r in results], dtype=float)
        out["avg_constraint_violations"] = float(v.mean()) if len(v) else 0.0

    # Placeholders (not implemented yet)
    not_impl = [m for m in metric_ids if m in {"energy", "safety", "flight_time", "smoothness", "robustness"}]
    for m in not_impl:
        out[m] = float("nan")

    return out


# ----------------- CLI -----------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="uavbench",
        description="Run UAVBench benchmark experiments.",
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        default="urban_easy",
        help="Comma-separated list of scenario IDs to run (e.g. urban_easy,urban_medium).",
    )
    parser.add_argument(
        "--planners",
        type=str,
        default="astar",
        help="Comma-separated list of planner IDs to run (e.g. astar).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="path_length,success_rate,constraint_violations",
        help="Comma-separated list of metrics to compute "
             "(e.g. path_length,success_rate,constraint_violations).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per (scenario, planner) pair.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--play",
        type=str,
        default="",
        choices=["", "best", "worst"],
        help="Open a window to play the best or worst successful path (per scenario/planner).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Playback speed for --play (frames per second).",
    )
    parser.add_argument(
        "--save-videos",
        type=str,
        default="",
        choices=["", "best", "worst", "both"],
        help="Save best/worst/both successful paths as MP4 videos to 'videos/' directory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on any exception (V&V mode).",
    )

    args = parser.parse_args()

    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    planner_ids = [p.strip() for p in args.planners.split(",") if p.strip()]
    metric_ids = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print(
        "[UAVBench] "
        f"scenarios={scenario_ids}, "
        f"planners={planner_ids}, "
        f"metrics={metric_ids}, "
        f"trials={args.trials}, "
        f"seed_base={args.seed_base}"
    )

    # Run grid experiment
    for scenario_id in scenario_ids:
        sp = scenario_path(scenario_id)
        if not sp.exists():
            raise FileNotFoundError(f"Scenario not found: {sp}")

        for planner_id in planner_ids:
            per_trial: list[dict[str, Any]] = []

            for t in range(args.trials):
                seed = args.seed_base + (hash(scenario_id) & 0xFFFF) + (hash(planner_id) & 0x0FFF) + t
                try:
                    r = run_planner_once(scenario_id, planner_id, seed=seed)
                except Exception as e:
                    if args.fail_fast:
                        raise
                    r = {
                        "scenario": scenario_id,
                        "planner": planner_id,
                        "seed": int(seed),
                        "success": False,
                        "constraint_violations": 1,
                        "path_length": 0,
                        "path": None,
                        "heightmap": None,
                        "no_fly": None,
                        "start": None,
                        "goal": None,
                        "error": str(e),
                    }
                per_trial.append(r)

            # Print metrics
            metrics = aggregate(per_trial, metric_ids)

            print("\n------------------------------")
            print(f"Scenario: {scenario_id}")
            print(f"Planner : {planner_id}")
            print(f"Trials  : {args.trials}")
            for k, v in metrics.items():
                if isinstance(v, float) and np.isnan(v):
                    print(f"{k:>24}: n/a")
                else:
                    print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")
            print("------------------------------")

            # AFTER TRIALS: play best/worst if requested
            successful = [r for r in per_trial if r.get("success", False)]
            
            # Handle visualization (--play) and video saving (--save-videos)
            if (args.play or args.save_videos) and successful:
                # Lazy import: only load matplotlib if visualization is requested
                try:
                    from uavbench.viz.player import play_path_window, save_path_video
                except ModuleNotFoundError as e:
                    print(
                        f"\n[ERROR] Cannot play visualization: {e}\n"
                        "  To enable visualization, install the optional viz dependencies:\n"
                        "    pip install uavbench[viz]\n"
                        "  or:\n"
                        "    pip install matplotlib>=3.8.0\n"
                    )
                    return

                # Determine which paths to visualize
                to_visualize = {}
                if args.play:
                    if args.play == "best":
                        to_visualize["best"] = min(successful, key=lambda r: r["path_length"])
                    elif args.play == "worst":
                        to_visualize["worst"] = max(successful, key=lambda r: r["path_length"])
                
                if args.save_videos:
                    if args.save_videos in ("best", "both"):
                        to_visualize["best"] = min(successful, key=lambda r: r["path_length"])
                    if args.save_videos in ("worst", "both"):
                        to_visualize["worst"] = max(successful, key=lambda r: r["path_length"])

                # Visualize/save the selected paths
                for vis_type, chosen in to_visualize.items():
                    title = (
                        f"{scenario_id} – {planner_id} – {vis_type.upper()} "
                        f"(seed={chosen['seed']}, len={chosen['path_length']})"
                    )
                    
                    # Play in window if requested
                    if args.play and vis_type == (args.play if args.play != "" else None):
                        play_path_window(
                            chosen["heightmap"],
                            chosen["no_fly"],
                            chosen["start"],
                            chosen["goal"],
                            chosen["path"],
                            title=title,
                            fps=args.fps,
                        )
                    
                    # Save as video if requested
                    if args.save_videos:
                        video_dir = Path("videos")
                        video_name = f"{scenario_id}_{planner_id}_{vis_type}_seed{chosen['seed']}.mp4"
                        video_path = video_dir / video_name
                        
                        save_path_video(
                            chosen["heightmap"],
                            chosen["no_fly"],
                            chosen["start"],
                            chosen["goal"],
                            chosen["path"],
                            video_path,
                            title=title,
                            fps=args.fps,
                        )


if __name__ == "__main__":
    main()
