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
        "map_source": cfg.map_source,
        "osm_tile_id": cfg.osm_tile_id,
        "config": cfg,
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
    parser.add_argument(
        "--save-figures",
        type=str,
        default="",
        metavar="DIR",
        help="Save publication-quality trajectory figures to DIR.",
    )
    parser.add_argument(
        "--with-dynamics",
        action="store_true",
        help="Include fire/traffic dynamics overlays in visualizations (requires OSM tile).",
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

            # AFTER TRIALS: visualization (play / save-videos / save-figures)
            successful = [r for r in per_trial if r.get("success", False)]
            want_viz = args.play or args.save_videos or args.save_figures

            if want_viz and successful:
                try:
                    from uavbench.viz.player import play_path_window, save_path_video
                except ModuleNotFoundError as e:
                    print(
                        f"\n[ERROR] Cannot load visualization: {e}\n"
                        "  Install: pip install uavbench[viz]\n"
                    )
                    return

                # Determine which paths to visualize
                to_visualize: dict[str, dict] = {}
                if args.play in ("best",) or args.save_videos in ("best", "both") or args.save_figures:
                    to_visualize["best"] = min(successful, key=lambda r: r["path_length"])
                if args.play == "worst" or args.save_videos in ("worst", "both"):
                    to_visualize["worst"] = max(successful, key=lambda r: r["path_length"])

                for vis_type, chosen in to_visualize.items():
                    title = (
                        f"{scenario_id} - {planner_id} - {vis_type.upper()} "
                        f"(seed={chosen['seed']}, len={chosen['path_length']})"
                    )

                    # Prepare dynamics overlays if requested
                    dynamics_kwargs: dict[str, Any] = {}
                    if args.with_dynamics and chosen.get("map_source") == "osm":
                        try:
                            from uavbench.viz.dynamics_sim import simulate_dynamics_along_path
                            tile_path = Path("data/maps") / f"{chosen['osm_tile_id']}.npz"
                            cfg = chosen["config"]
                            sim = simulate_dynamics_along_path(
                                tile_path=tile_path,
                                path=chosen["path"],
                                enable_fire=cfg.enable_fire,
                                enable_traffic=cfg.enable_traffic,
                                fire_ignition_points=cfg.fire_ignition_points,
                                wind_direction=cfg.wind_direction,
                                wind_speed=cfg.wind_speed,
                                num_vehicles=cfg.num_emergency_vehicles,
                                seed=chosen["seed"],
                            )
                            dynamics_kwargs = {
                                "fire_states": sim["fire_states"],
                                "burned_states": sim["burned_states"],
                                "traffic_states": sim["traffic_states"],
                                "roads_mask": sim["roads_mask"],
                                "risk_map": sim["risk_map"],
                            }
                        except Exception as e:
                            print(f"[WARNING] Dynamics overlay failed: {e}")

                    # Play in window
                    if args.play and vis_type == args.play:
                        play_path_window(
                            chosen["heightmap"], chosen["no_fly"],
                            chosen["start"], chosen["goal"], chosen["path"],
                            title=title, fps=args.fps,
                            **dynamics_kwargs,
                        )

                    # Save video
                    if args.save_videos:
                        video_dir = Path("videos")
                        video_name = f"{scenario_id}_{planner_id}_{vis_type}_seed{chosen['seed']}.mp4"
                        save_path_video(
                            chosen["heightmap"], chosen["no_fly"],
                            chosen["start"], chosen["goal"], chosen["path"],
                            video_dir / video_name,
                            title=title, fps=args.fps,
                            **dynamics_kwargs,
                        )

                    # Save publication figure
                    if args.save_figures:
                        try:
                            from uavbench.viz.figures import plot_trajectory_with_dynamics
                            fig_dir = Path(args.save_figures)
                            fig_name = f"{scenario_id}_{planner_id}_{vis_type}_seed{chosen['seed']}.png"
                            fig_kwargs: dict[str, Any] = {}
                            if dynamics_kwargs.get("fire_states"):
                                fig_kwargs["fire_mask"] = dynamics_kwargs["fire_states"][-1]
                            if dynamics_kwargs.get("burned_states"):
                                fig_kwargs["burned_mask"] = dynamics_kwargs["burned_states"][-1]
                            if dynamics_kwargs.get("traffic_states"):
                                fig_kwargs["vehicle_positions"] = dynamics_kwargs["traffic_states"][-1]
                            if dynamics_kwargs.get("roads_mask") is not None:
                                fig_kwargs["roads_mask"] = dynamics_kwargs["roads_mask"]
                            if dynamics_kwargs.get("risk_map") is not None:
                                fig_kwargs["risk_map"] = dynamics_kwargs["risk_map"]
                            plot_trajectory_with_dynamics(
                                chosen["heightmap"], chosen["no_fly"],
                                chosen["start"], chosen["goal"], chosen["path"],
                                fig_dir / fig_name,
                                title=title, **fig_kwargs,
                            )
                        except Exception as e:
                            print(f"[WARNING] Figure save failed: {e}")


if __name__ == "__main__":
    main()
