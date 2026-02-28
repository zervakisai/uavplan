"""Enhanced benchmark runner with oracle/non-oracle protocols and determinism.

This module orchestrates full benchmark runs:
- Load scenarios and initialize environment
- Run planners with time budgets
- Collect metrics and events
- Support oracle vs non-oracle modes
- Aggregate results across seeds
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.cli.benchmark import run_dynamic_episode, run_planner_once
from uavbench.metrics.comprehensive import (
    EpisodeMetrics, AggregateMetrics, compute_episode_metrics,
    aggregate_episode_metrics, save_episode_metrics_jsonl, save_aggregate_metrics_csv,
    print_aggregate_metrics_table
)
from uavbench.missions.engine import generate_briefing
from uavbench.planners import PLANNERS
from uavbench.planners.astar import AStarPlanner
from uavbench.scenarios.loader import load_scenario

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    # Scenarios and planners
    scenario_ids: list[str]
    planner_ids: list[str]
    
    # Seeds and runs
    seeds: list[int]
    max_episode_steps: int = 1000
    
    # Oracle mode
    oracle_horizon: int = 0  # 0 = non-oracle, >0 = oracle with horizon
    oracle_planner_id: Optional[str] = "astar"
    
    # Determinism
    deterministic: bool = True
    
    # Output
    output_dir: Path = Path("./benchmark_results")
    save_jsonl: bool = True
    save_csv: bool = True
    
    # Debug
    verbose: bool = False
    early_stop_seed: Optional[int] = None  # Stop after first N seeds (for testing)


class BenchmarkRunner:
    """Orchestrates benchmark runs across scenarios, planners, and seeds."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if config.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    
    def run(self) -> dict[str, AggregateMetrics]:
        """Run full benchmark suite.
        
        Returns:
            Dict mapping "scenario_id/planner_id" -> AggregateMetrics
        """
        all_episodes: dict[tuple[str, str], list[EpisodeMetrics]] = {}
        
        scenario_count = len(self.config.scenario_ids)
        planner_count = len(self.config.planner_ids)
        seed_count = len(self.config.seeds)
        
        total_runs = scenario_count * planner_count * seed_count
        run_idx = 0
        
        logger.info(f"Starting benchmark: {scenario_count} scenarios × {planner_count} planners × {seed_count} seeds = {total_runs} runs")
        
        for scenario_id in self.config.scenario_ids:
            for planner_id in self.config.planner_ids:
                episodes = []
                
                for seed in self.config.seeds:
                    run_idx += 1
                    progress = f"[{run_idx}/{total_runs}]"
                    
                    logger.info(f"{progress} Running {scenario_id} / {planner_id} / seed={seed}")
                    
                    try:
                        episode = self._run_episode(scenario_id, planner_id, seed)
                        episodes.append(episode)
                        
                        status = "✓" if episode.success else "✗"
                        logger.info(
                            f"{progress} {status} {scenario_id} / {planner_id} "
                            f"path={episode.path_length:.0f} "
                            f"time={episode.planning_time_ms:.1f}ms"
                        )
                        
                        if self.config.early_stop_seed is not None and run_idx >= self.config.early_stop_seed:
                            logger.info("Early stop requested")
                            return {}
                    
                    except Exception as e:
                        logger.error(f"{progress} Exception: {e}", exc_info=True)
                        episodes.append(
                            EpisodeMetrics(
                                scenario_id=scenario_id,
                                planner_id=planner_id,
                                seed=seed,
                                episode_step=0,
                                success=False,
                                termination_reason=f"exception: {type(e).__name__}",
                                path_length=0.0,
                                path_length_any_angle=None,
                                planning_time_ms=0.0,
                                total_time_ms=0.0,
                                replans=0,
                                first_replan_step=None,
                                blocked_path_events=0,
                                collision_count=0,
                                nfz_violations=0,
                                fire_exposure=0.0,
                                traffic_proximity_time=0.0,
                                intruder_proximity_time=0.0,
                                smoke_exposure=0.0,
                                regret_length=None,
                                regret_risk=None,
                                regret_time=None,
                                notes=f"Exception: {e}",
                            )
                        )
                
                all_episodes[(scenario_id, planner_id)] = episodes
        
        # Aggregate results
        logger.info("Aggregating results...")
        aggregates = self._aggregate_all_episodes(all_episodes)
        
        # Save results
        self._save_results(all_episodes, aggregates)
        
        # Print summary
        print_aggregate_metrics_table(list(aggregates.values()))
        
        return aggregates
    
    def _run_episode(self, scenario_id: str, planner_id: str, seed: int) -> EpisodeMetrics:
        """Run a single episode and return metrics."""
        cfg = load_scenario(Path(f"src/uavbench/scenarios/configs/{scenario_id}.yaml"))
        if planner_id not in PLANNERS:
            raise ValueError(f"Unknown planner '{planner_id}'")

        # Generate mission briefing (MC-1: every episode has a mission objective)
        briefing = generate_briefing(cfg)
        briefing_event = {
            "step": 0,
            "type": "mission_briefing",
            "payload": briefing.to_dict(),
        }

        use_dynamic = (
            cfg.paper_track == "dynamic"
            or int(cfg.force_replan_count) > 0
            or cfg.fire_blocks_movement
            or cfg.traffic_blocks_movement
            or cfg.enable_moving_target
            or cfg.enable_intruders
            or cfg.enable_dynamic_nfz
        )
        if use_dynamic:
            res = run_dynamic_episode(
                scenario_id,
                planner_id,
                seed=seed,
                protocol_variant="default",
                episode_horizon_steps=self.config.max_episode_steps,
            )
            path = list(res.get("path") or [res.get("start")])
            start_xy = tuple(res.get("start", (0, 0)))
            goal_xy = tuple(res.get("goal", (0, 0)))
            heightmap = np.asarray(res.get("heightmap"))
            no_fly = np.asarray(res.get("no_fly"))
            planning_time_ms = float(res.get("planning_time_ms", float(res.get("planning_time", 0.0) * 1000.0)))
            episode_duration_ms = float(res.get("episode_steps", 0)) * 1.0
            collisions = 1 if bool(res.get("collision_terminated", False)) else 0
            nfz_violations = int(res.get("constraint_violations", 0))
            termination_reason = str(res.get("termination_reason", "unknown"))
            success = bool(res.get("success", False))
            replans = int(res.get("total_replans", 0))
            env_events = [briefing_event] + list(res.get("events", []))
        else:
            res = run_planner_once(scenario_id, planner_id, seed=seed)
            path = list(res.get("path") or [res.get("start")])
            start_xy = tuple(res.get("start", (0, 0)))
            goal_xy = tuple(res.get("goal", (0, 0)))
            heightmap = np.asarray(res.get("heightmap"))
            no_fly = np.asarray(res.get("no_fly"))
            planning_time_ms = float(res.get("planning_time_ms", float(res.get("planning_time", 0.0) * 1000.0)))
            episode_duration_ms = planning_time_ms
            collisions = 0
            nfz_violations = int(res.get("constraint_violations", 0))
            termination_reason = "success" if bool(res.get("success", False)) else "static_validation_failed"
            success = bool(res.get("success", False))
            replans = 0
            env_events = [briefing_event]

        return compute_episode_metrics(
            scenario_id=scenario_id,
            planner_id=planner_id,
            seed=seed,
            success=success,
            path=path,
            start=start_xy,
            goal=goal_xy,
            heightmap=heightmap,
            no_fly=no_fly,
            planning_time_ms=planning_time_ms,
            episode_duration_ms=episode_duration_ms,
            replans=replans,
            first_replan_step=None,
            env_events=env_events,
            collisions=collisions,
            nfz_violations=nfz_violations,
            termination_reason=termination_reason,
        )
    
    def _aggregate_all_episodes(
        self,
        all_episodes: dict[tuple[str, str], list[EpisodeMetrics]],
    ) -> dict[str, AggregateMetrics]:
        """Aggregate episode metrics across all runs."""
        aggregates = {}
        
        for (scenario_id, planner_id), episodes in all_episodes.items():
            if not episodes:
                continue
            
            agg = aggregate_episode_metrics(
                episodes,
                oracle_planner_id=self.config.oracle_planner_id,
            )
            aggregates[f"{scenario_id}/{planner_id}"] = agg
        
        return aggregates
    
    def _save_results(
        self,
        all_episodes: dict[tuple[str, str], list[EpisodeMetrics]],
        aggregates: dict[str, AggregateMetrics],
    ) -> None:
        """Save results to disk."""
        output_dir = self.config.output_dir
        
        # Flatten episodes
        all_episode_list = []
        for episodes in all_episodes.values():
            all_episode_list.extend(episodes)
        
        # Save JSONL
        if self.config.save_jsonl:
            jsonl_path = output_dir / "episodes.jsonl"
            save_episode_metrics_jsonl(all_episode_list, jsonl_path)
            logger.info(f"Saved episode metrics: {jsonl_path}")
        
        # Save CSV
        if self.config.save_csv:
            csv_path = output_dir / "aggregates.csv"
            save_aggregate_metrics_csv(list(aggregates.values()), csv_path)
            logger.info(f"Saved aggregate metrics: {csv_path}")

        # Save reproducibility metadata
        metadata_path = output_dir / "run_metadata.json"
        metadata = {
            "scenario_ids": self.config.scenario_ids,
            "planner_ids": self.config.planner_ids,
            "seeds": self.config.seeds,
            "deterministic": self.config.deterministic,
            "oracle_horizon": self.config.oracle_horizon,
            "oracle_planner_id": self.config.oracle_planner_id,
            "max_episode_steps": self.config.max_episode_steps,
            "python_version": platform.python_version(),
            "num_episode_records": len(all_episode_list),
            "num_aggregate_rows": len(aggregates),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Saved run metadata: {metadata_path}")


def _waypoint_to_action(current: tuple[int, int], next_pos: tuple[int, int]) -> int:
    """Convert waypoints to discrete action (0-5)."""
    dx = next_pos[0] - current[0]
    dy = next_pos[1] - current[1]
    
    if dx == 0 and dy == -1:
        return 0  # up
    elif dx == 0 and dy == 1:
        return 1  # down
    elif dx == -1 and dy == 0:
        return 2  # left
    elif dx == 1 and dy == 0:
        return 3  # right
    else:
        return 0  # fallback


def _expand_execution_path(
    path: list[tuple[int, int]],
    heightmap: Any,
    no_fly: Any,
) -> list[tuple[int, int]]:
    """Expand sparse/any-angle waypoints into 4-connected executable steps."""
    if len(path) < 2:
        return list(path)

    expanded: list[tuple[int, int]] = [path[0]]
    segment_planner = AStarPlanner(heightmap, no_fly)
    for waypoint in path[1:]:
        prev = expanded[-1]
        dx = abs(waypoint[0] - prev[0])
        dy = abs(waypoint[1] - prev[1])
        if dx + dy == 1:
            expanded.append(waypoint)
            continue
        seg = segment_planner.plan(prev, waypoint)
        if seg.success and len(seg.path) >= 2:
            expanded.extend(seg.path[1:])
        else:
            # Keep original waypoint if local expansion unexpectedly fails.
            expanded.append(waypoint)
    return expanded


def main():
    """Main entry point for benchmark CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run UAVBench benchmarks")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["osm_athens_wildfire_easy"],
        help="Scenario IDs to run",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["astar"],
        help="Planner IDs to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=0,
        help="Oracle horizon (0 = non-oracle)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=None,
        help="Stop after N runs (for testing)",
    )
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        scenario_ids=args.scenarios,
        planner_ids=args.planners,
        seeds=args.seeds,
        output_dir=args.output_dir,
        oracle_horizon=args.oracle_horizon,
        verbose=args.verbose,
        early_stop_seed=args.early_stop,
    )
    
    runner = BenchmarkRunner(config)
    aggregates = runner.run()
    
    print(f"\n✓ Benchmark complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
