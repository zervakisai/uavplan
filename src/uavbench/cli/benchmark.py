from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.metrics.operational import compute_all_metrics
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario


# ----------------- Scenario path resolution -----------------

def scenario_path(scenario_id: str) -> Path:
    # src/uavbench/cli/benchmark.py -> parents[1] = src/uavbench
    base = Path(__file__).resolve().parents[1]
    return base / "scenarios" / "configs" / f"{scenario_id}.yaml"


# ----------------- Path-to-action helper -----------------

def _waypoint_action(curr_xy: tuple[int, int], next_xy: tuple[int, int]) -> int:
    """Convert consecutive (x,y) waypoints to a Discrete(6) action.

    Actions: 0=up(y-1), 1=down(y+1), 2=left(x-1), 3=right(x+1).
    """
    dx = next_xy[0] - curr_xy[0]
    dy = next_xy[1] - curr_xy[1]
    if dy == -1:
        return 0  # up
    if dy == 1:
        return 1  # down
    if dx == -1:
        return 2  # left
    if dx == 1:
        return 3  # right
    return 0  # fallback (shouldn't happen with 4-connected A*)


# ----------------- Static planner run -----------------

def run_planner_once(
    scenario_id: str,
    planner_id: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Plan a static path and validate against the map (no env stepping)."""
    cfg = load_scenario(scenario_path(scenario_id))
    env = UrbanEnv(cfg)

    env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")

    planner = PLANNERS[planner_id](heightmap, no_fly)
    t0 = time.perf_counter()
    path = planner.plan(start_xy, goal_xy)  # [] if no path
    planning_time = time.perf_counter() - t0

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
        "planning_time": planning_time,
        "map_source": cfg.map_source,
        "osm_tile_id": cfg.osm_tile_id,
        "config": cfg,
    }


# ----------------- Dynamic episode run -----------------

def run_dynamic_episode(
    scenario_id: str,
    planner_id: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run a full episode stepping through the environment.

    Unlike run_planner_once (static validation), this:
    1. Plans an initial path
    2. Steps through the env action-by-action
    3. For adaptive planners: checks should_replan each step, replans if needed
    4. For static planners: follows the initial path blindly (may get stuck)
    """
    cfg = load_scenario(scenario_path(scenario_id))
    env = UrbanEnv(cfg)

    env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    if planner_id not in PLANNERS:
        raise ValueError(f"Unknown planner '{planner_id}'. Available: {list(PLANNERS.keys())}")

    planner = PLANNERS[planner_id](heightmap, no_fly)
    t0 = time.perf_counter()
    path = planner.plan(start_xy, goal_xy)
    planning_time = time.perf_counter() - t0

    is_adaptive = hasattr(planner, "should_replan")

    if not path:
        return {
            "scenario": scenario_id,
            "planner": planner_id,
            "seed": int(seed),
            "success": False,
            "constraint_violations": 0,
            "path_length": 0,
            "path": None,
            "heightmap": heightmap,
            "no_fly": no_fly,
            "start": start_xy,
            "goal": goal_xy,
            "planning_time": planning_time,
            "map_source": cfg.map_source,
            "osm_tile_id": cfg.osm_tile_id,
            "config": cfg,
            "episode_steps": 0,
            "total_replans": 0,
        }

    # Episode execution
    path_idx = 0
    actual_trajectory = [start_xy]
    total_reward = 0.0
    violations = 0
    max_steps = 4 * int(cfg.map_size)
    stuck_counter = 0
    reached_goal = False
    collision_terminated = False
    termination_reason = "timeout"
    episode_steps = 0

    for step in range(max_steps):
        episode_steps = step + 1

        # End of current path
        if path_idx >= len(path) - 1:
            break

        action = _waypoint_action(path[path_idx], path[path_idx + 1])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Check if agent actually moved to expected next waypoint
        current_pos = tuple(int(v) for v in env._agent_pos)  # (x, y, z)
        current_xy = (current_pos[0], current_pos[1])

        if current_xy == path[path_idx + 1]:
            path_idx += 1
            actual_trajectory.append(current_xy)
            stuck_counter = 0
        else:
            stuck_counter += 1

        if terminated:
            reached_goal = info.get("reached_goal", False)
            collision_terminated = info.get("collision_terminated", False)
            termination_reason = info.get("termination_reason", "unknown")
            break

        if truncated:
            break

        # Adaptive replanning
        if is_adaptive:
            dyn = env.get_dynamic_state()

            # Build merged extra_obstacles from new dynamic layers
            extra_obs = None
            for key in ("moving_target_buffer", "intruder_buffer", "dynamic_nfz_mask"):
                layer = dyn.get(key)
                if layer is not None:
                    extra_obs = layer if extra_obs is None else (extra_obs | layer)

            should, reason = planner.should_replan(
                current_pos, dyn["fire_mask"], dyn["traffic_positions"],
                smoke_mask=dyn["smoke_mask"],
                extra_obstacles=extra_obs,
            )
            # Also replan if stuck (move was rejected 3+ times)
            if should or stuck_counter >= 3:
                new_path = planner.replan(
                    current_pos, goal_xy,
                    dyn["fire_mask"], dyn["traffic_positions"],
                    reason or "stuck",
                    smoke_mask=dyn["smoke_mask"],
                    extra_obstacles=extra_obs,
                )
                if new_path:
                    path = new_path
                    path_idx = 0
                    stuck_counter = 0
        elif stuck_counter >= 10:
            # Static planner: give up if stuck too long
            break

    # Count real constraint violations (NFZ, building) — fire blocks are expected
    events = env.events
    nfz_violations = sum(1 for e in events if e.get("type") == "no_fly_violation_attempt")
    building_violations = sum(1 for e in events if e.get("type") == "collision_building_attempt")
    fire_blocks = sum(1 for e in events if e.get("type") == "fire_block")
    traffic_blocks = sum(1 for e in events if e.get("type") == "traffic_block")
    target_blocks = sum(1 for e in events if e.get("type") == "target_block")
    intruder_blocks = sum(1 for e in events if e.get("type") == "intruder_block")
    nfz_dyn_blocks = sum(1 for e in events if e.get("type") == "dynamic_nfz_block")
    violations = nfz_violations + building_violations

    # Also check if we reached the goal position even without terminated flag
    final_xy = (int(env._agent_pos[0]), int(env._agent_pos[1]))
    if final_xy == goal_xy and not reached_goal:
        reached_goal = True

    success = reached_goal

    replan_metrics = planner.get_replan_metrics() if is_adaptive else {"total_replans": 0}

    return {
        "scenario": scenario_id,
        "planner": planner_id,
        "seed": int(seed),
        "success": success,
        "constraint_violations": int(violations),
        "path_length": int(len(actual_trajectory)) if success else 0,
        "path": actual_trajectory if success else None,
        "heightmap": heightmap,
        "no_fly": no_fly,
        "start": start_xy,
        "goal": goal_xy,
        "planning_time": planning_time,
        "map_source": cfg.map_source,
        "osm_tile_id": cfg.osm_tile_id,
        "config": cfg,
        "episode_steps": episode_steps,
        "total_replans": replan_metrics.get("total_replans", 0),
        "total_reward": total_reward,
        "fire_blocks": fire_blocks,
        "traffic_blocks": traffic_blocks,
        "target_blocks": target_blocks,
        "intruder_blocks": intruder_blocks,
        "dynamic_nfz_blocks": nfz_dyn_blocks,
        "total_dynamic_blocks": fire_blocks + traffic_blocks + target_blocks + intruder_blocks + nfz_dyn_blocks,
        "termination_reason": termination_reason,
        "collision_terminated": collision_terminated,
    }


# ----------------- Metrics aggregation -----------------

def aggregate(results: list[dict[str, Any]], _metric_ids: list[str] | None = None) -> dict[str, float]:
    """Aggregate per-trial results into metrics.

    Uses compute_all_metrics() for per-trial operational metrics, then
    averages across trials.
    """
    if not results:
        return {}

    # Per-trial operational metrics
    per_trial_metrics = [compute_all_metrics(r) for r in results]

    out: dict[str, float] = {}

    # Core metrics (always computed for backward compat)
    successes = np.array([1.0 if r["success"] else 0.0 for r in results], dtype=float)
    out["success_rate"] = float(successes.mean())

    # Collision and timeout rates (dynamic episodes)
    collision_flags = [r.get("collision_terminated", False) for r in results]
    if any(collision_flags):
        out["collision_rate"] = float(np.mean([1.0 if c else 0.0 for c in collision_flags]))

    timeout_flags = [r.get("termination_reason") == "timeout" for r in results]
    if any(timeout_flags):
        out["timeout_rate"] = float(np.mean([1.0 if t else 0.0 for t in timeout_flags]))

    lengths = np.array([float(r["path_length"]) for r in results if r["success"]], dtype=float)
    out["avg_path_length"] = float(lengths.mean()) if len(lengths) else float("nan")

    violations = np.array([float(r["constraint_violations"]) for r in results], dtype=float)
    out["avg_constraint_violations"] = float(violations.mean())

    # Replanning metrics (if present)
    replans = [r.get("total_replans", 0) for r in results]
    if any(r > 0 for r in replans):
        out["avg_replans"] = round(float(np.mean(replans)), 1)

    ep_steps = [r.get("episode_steps", 0) for r in results]
    if any(s > 0 for s in ep_steps):
        out["avg_episode_steps"] = round(float(np.mean(ep_steps)), 0)

    # Dynamic blocking metrics
    dyn_blocks = [r.get("total_dynamic_blocks", 0) for r in results]
    if any(b > 0 for b in dyn_blocks):
        out["avg_dynamic_blocks"] = round(float(np.mean(dyn_blocks)), 1)

    f_blocks = [r.get("fire_blocks", 0) for r in results]
    if any(f > 0 for f in f_blocks):
        out["avg_fire_blocks"] = round(float(np.mean(f_blocks)), 1)

    t_blocks = [r.get("traffic_blocks", 0) for r in results]
    if any(t > 0 for t in t_blocks):
        out["avg_traffic_blocks"] = round(float(np.mean(t_blocks)), 1)

    # Operational metrics (averaged across trials)
    all_keys = set()
    for m in per_trial_metrics:
        all_keys.update(m.keys())

    for key in sorted(all_keys):
        vals = [m[key] for m in per_trial_metrics if key in m]
        if vals:
            out[f"avg_{key}"] = round(float(np.mean(vals)), 4)

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
        help="Comma-separated list of planner IDs to run (e.g. astar,adaptive_astar).",
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
        help="Save best/worst/both successful paths as MP4/GIF to 'videos/' directory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on any exception (V&V mode).",
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

        # Check if scenario needs dynamic episode execution
        cfg = load_scenario(sp)
        use_dynamic = (cfg.fire_blocks_movement or cfg.traffic_blocks_movement
                       or cfg.enable_moving_target or cfg.enable_intruders
                       or cfg.enable_dynamic_nfz)

        for planner_id in planner_ids:
            per_trial: list[dict[str, Any]] = []
            run_fn = run_dynamic_episode if use_dynamic else run_planner_once

            for t in range(args.trials):
                seed = args.seed_base + (hash(scenario_id) & 0xFFFF) + (hash(planner_id) & 0x0FFF) + t
                try:
                    r = run_fn(scenario_id, planner_id, seed=seed)
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
                        "planning_time": 0.0,
                        "error": str(e),
                    }
                per_trial.append(r)

            # Print metrics
            metrics = aggregate(per_trial, metric_ids)

            print("\n------------------------------")
            print(f"Scenario: {scenario_id}")
            print(f"Planner : {planner_id}")
            print(f"Trials  : {args.trials}")
            if use_dynamic:
                print(f"Mode    : dynamic episode")
            for k, v in metrics.items():
                if isinstance(v, float) and np.isnan(v):
                    print(f"{k:>24}: n/a")
                else:
                    print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")
            print("------------------------------")

            # AFTER TRIALS: visualization (play / save-videos)
            successful = [r for r in per_trial if r.get("success", False)]
            want_viz = args.play or args.save_videos

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
                if args.play in ("best",) or args.save_videos in ("best", "both"):
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
                            viz_cfg = chosen["config"]
                            sim = simulate_dynamics_along_path(
                                tile_path=tile_path,
                                path=chosen["path"],
                                enable_fire=viz_cfg.enable_fire,
                                enable_traffic=viz_cfg.enable_traffic,
                                fire_ignition_points=viz_cfg.fire_ignition_points,
                                wind_direction=viz_cfg.wind_direction,
                                wind_speed=viz_cfg.wind_speed,
                                num_vehicles=viz_cfg.num_emergency_vehicles,
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


if __name__ == "__main__":
    main()
