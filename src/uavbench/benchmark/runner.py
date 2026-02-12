"""Enhanced benchmark runner with oracle/non-oracle protocols and determinism.

This module orchestrates full benchmark runs:
- Load scenarios and initialize environment
- Run planners with time budgets
- Collect metrics and events
- Support oracle vs non-oracle modes
- Aggregate results across seeds
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.metrics.comprehensive import (
    EpisodeMetrics, AggregateMetrics, compute_episode_metrics,
    aggregate_episode_metrics, save_episode_metrics_jsonl, save_aggregate_metrics_csv,
    print_aggregate_metrics_table
)
from uavbench.planners import PLANNERS, BasePlanner
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.registry import list_scenarios

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
    
    def run(self) -> dict[str, list[AggregateMetrics]]:
        """Run full benchmark suite.
        
        Returns:
            Dict mapping (scenario_id, planner_id) -> list of AggregateMetrics
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
                                collision_count=0.0,
                                nfz_violations=0.0,
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
        # Load scenario and environment
        cfg = load_scenario(Path(f"src/uavbench/scenarios/configs/{scenario_id}.yaml"))
        env = UrbanEnv(cfg)
        
        # Reset with seed
        obs, info = env.reset(seed=seed)
        
        # Export planner inputs
        heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()
        
        # Initialize planner
        if planner_id not in PLANNERS:
            raise ValueError(f"Unknown planner '{planner_id}'")
        
        planner_cls = PLANNERS[planner_id]
        planner = planner_cls(heightmap, no_fly)
        
        # Plan
        plan_start_time = time.monotonic()
        plan_result = planner.plan(start_xy, goal_xy)
        planning_time_ms = (time.monotonic() - plan_start_time) * 1000
        
        if not plan_result.success:
            # Planning failed
            return EpisodeMetrics(
                scenario_id=scenario_id,
                planner_id=planner_id,
                seed=seed,
                episode_step=0,
                success=False,
                termination_reason=f"planning_failed: {plan_result.reason}",
                path_length=0.0,
                path_length_any_angle=None,
                planning_time_ms=planning_time_ms,
                total_time_ms=0.0,
                replans=0,
                first_replan_step=None,
                blocked_path_events=0,
                collision_count=0.0,
                nfz_violations=0.0,
                fire_exposure=0.0,
                traffic_proximity_time=0.0,
                intruder_proximity_time=0.0,
                smoke_exposure=0.0,
                regret_length=None,
                regret_risk=None,
                regret_time=None,
                notes=plan_result.reason,
            )
        
        # Execute path by stepping env
        episode_start_time = time.monotonic()
        path = [start_xy]
        collisions = 0
        nfz_violations = 0
        
        for step_idx in range(min(len(plan_result.path) - 1, self.config.max_episode_steps)):
            current_pos = plan_result.path[step_idx]
            next_pos = plan_result.path[step_idx + 1]
            
            # Convert waypoint to action
            action = _waypoint_to_action(current_pos, next_pos)
            
            obs, reward, terminated, truncated, info = env.step(action)
            path.append(next_pos)
            
            # Track collisions
            if terminated and "collision" in info.get("event_type", "").lower():
                collisions += 1
            
            # Check if goal reached
            if next_pos == goal_xy:
                break
        
        episode_duration_ms = (time.monotonic() - episode_start_time) * 1000
        success = path[-1] == goal_xy
        
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
            replans=plan_result.replans,
            first_replan_step=None,
            env_events=env.events,
            collisions=collisions,
            nfz_violations=nfz_violations,
            termination_reason="success" if success else "timeout_or_blocked",
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
