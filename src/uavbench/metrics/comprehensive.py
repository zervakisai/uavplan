"""Comprehensive metrics suite for UAVBench benchmarks.

Includes:
- Efficiency: path length, optimality, planning time
- Safety: collisions, NFZ violations, risk exposure (fire, traffic, intruders)
- Dynamic replanning: replan count, first replan step, blocked path events
- Regret vs oracle: path length regret, risk regret, time regret
- Statistical aggregation: mean, std, 95% CI across multiple seeds
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


GridPos = tuple[int, int]  # (x, y)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode run."""
    # Identity
    scenario_id: str
    planner_id: str
    seed: int
    episode_step: int
    
    # Success / termination
    success: bool
    termination_reason: str
    
    # Efficiency
    path_length: float  # L1 distance
    path_length_any_angle: Optional[float]  # Euclidean if tracked
    planning_time_ms: float
    total_time_ms: float
    replans: int
    first_replan_step: Optional[int]
    blocked_path_events: int
    
    # Safety: collisions and constraints
    collision_count: int
    nfz_violations: int
    fire_exposure: float  # Integral of fire intensity along path
    traffic_proximity_time: float  # Steps near vehicle
    intruder_proximity_time: float  # Steps near intruder
    smoke_exposure: float  # Integral of smoke along path
    
    # Regret vs oracle (if oracle ran)
    regret_length: Optional[float]  # (planner path - oracle path) / oracle path
    regret_risk: Optional[float]  # (planner risk - oracle risk) / (oracle risk + eps)
    regret_time: Optional[float]  # (planner steps - oracle steps)

    # Advanced dynamics (best-paper protocol)
    risk_integral_tls: float = 0.0  # combined threat-load score integral
    stability_index: float = 0.0  # normalized heading-change magnitude

    # Metadata
    notes: str = ""


@dataclass
class AggregateMetrics:
    """Aggregated metrics across N trials."""
    scenario_id: str
    planner_id: str
    num_seeds: int
    
    # Success rate
    success_count: int
    success_rate: float
    
    # Efficiency (mean +/- std over successful runs)
    path_length_mean: float
    path_length_std: float
    path_length_min: float
    path_length_max: float
    path_length_ci_lower: float  # 95% CI
    path_length_ci_upper: float
    
    planning_time_mean_ms: float
    planning_time_std_ms: float
    
    replans_mean: float
    replans_std: float
    
    # Safety
    collision_mean: float
    collision_rate: float
    nfz_violation_mean: float
    nfz_violation_rate: float
    fire_exposure_mean: float
    traffic_proximity_mean: float
    intruder_proximity_mean: float
    
    # Regret (vs oracle, if available)
    regret_length_mean: Optional[float]
    regret_length_std: Optional[float]
    regret_risk_mean: Optional[float]
    regret_risk_std: Optional[float]
    
    # Metadata
    oracle_planner_id: Optional[str] = None


def compute_episode_metrics(
    scenario_id: str,
    planner_id: str,
    seed: int,
    success: bool,
    path: list[GridPos],
    start: GridPos,
    goal: GridPos,
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    planning_time_ms: float,
    episode_duration_ms: float,
    replans: int = 0,
    first_replan_step: Optional[int] = None,
    env_events: Optional[list[dict[str, Any]]] = None,
    collisions: int = 0,
    nfz_violations: int = 0,
    fire_exposure: float = 0.0,
    traffic_proximity_time: float = 0.0,
    intruder_proximity_time: float = 0.0,
    smoke_exposure: float = 0.0,
    termination_reason: str = "unknown",
    oracle_path: Optional[list[GridPos]] = None,
    oracle_risk: Optional[float] = None,
) -> EpisodeMetrics:
    """Compute metrics for a single episode.
    
    Args:
        scenario_id: Scenario name
        planner_id: Planner name
        seed: RNG seed
        success: Whether goal was reached
        path: List of (x, y) waypoints traveled
        start: Start position
        goal: Goal position
        heightmap: Building height map
        no_fly: No-fly zone mask
        planning_time_ms: Planning computation time
        episode_duration_ms: Total episode time
        replans: Number of times planner re-ran
        first_replan_step: Step at which first replan occurred
        env_events: Event log from environment
        collisions: Collision count
        nfz_violations: NFZ violation count
        fire_exposure, traffic_proximity_time, etc.: Dynamic hazard metrics
        termination_reason: Why episode ended (success, collision, timeout, etc.)
        oracle_path: Path from oracle planner (for regret computation)
        oracle_risk: Risk from oracle (for regret computation)
    
    Returns:
        EpisodeMetrics dataclass
    """
    # Efficiency: path length (L1)
    path_length = len(path) if path else 0
    
    # Any-angle path length (Euclidean)
    path_length_any_angle = None
    if path:
        path_length_any_angle = sum(
            ((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)**0.5
            for i in range(len(path) - 1)
        )
    
    # Blocked path events (from env_events)
    blocked_path_events = 0
    if env_events:
        blocked_path_events = len([e for e in env_events if e.get("type") == "path_blocked"])
    
    # Regret computation
    regret_length = None
    regret_risk = None
    regret_time = None
    if oracle_path is not None:
        oracle_len = len(oracle_path)
        if oracle_len > 0:
            regret_length = (path_length - oracle_len) / oracle_len
    if oracle_risk is not None:
        eps = 1e-6
        regret_risk = (fire_exposure - oracle_risk) / (oracle_risk + eps)

    # Risk integral (TLS proxy): aggregate dynamic threat exposures
    risk_integral_tls = float(fire_exposure + smoke_exposure + traffic_proximity_time + intruder_proximity_time)

    # Stability index: normalized cumulative heading changes across path
    stability_index = 0.0
    if path and len(path) >= 3:
        changes = 0
        prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        for i in range(1, len(path) - 1):
            cur = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if cur != prev:
                changes += 1
            prev = cur
        stability_index = float(changes / max(1, len(path) - 2))

    return EpisodeMetrics(
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
        episode_step=len(path),
        success=success,
        termination_reason=termination_reason,
        path_length=float(path_length),
        path_length_any_angle=path_length_any_angle,
        planning_time_ms=planning_time_ms,
        total_time_ms=episode_duration_ms,
        replans=replans,
        first_replan_step=first_replan_step,
        blocked_path_events=blocked_path_events,
        collision_count=int(collisions),
        nfz_violations=int(nfz_violations),
        fire_exposure=fire_exposure,
        traffic_proximity_time=traffic_proximity_time,
        intruder_proximity_time=intruder_proximity_time,
        smoke_exposure=smoke_exposure,
        regret_length=regret_length,
        regret_risk=regret_risk,
        regret_time=regret_time,
        risk_integral_tls=risk_integral_tls,
        stability_index=stability_index,
    )


def aggregate_episode_metrics(
    episodes: list[EpisodeMetrics],
    oracle_planner_id: Optional[str] = None,
) -> AggregateMetrics:
    """Aggregate metrics across multiple episodes (seeds).
    
    Args:
        episodes: List of EpisodeMetrics from multiple seeds
        oracle_planner_id: Name of oracle planner (if used)
    
    Returns:
        AggregateMetrics with statistical summaries
    """
    if not episodes:
        raise ValueError("episodes list is empty")
    
    scenario_id = episodes[0].scenario_id
    planner_id = episodes[0].planner_id
    num_seeds = len(episodes)
    
    # Success rate
    successes = [e for e in episodes if e.success]
    success_count = len(successes)
    success_rate = success_count / num_seeds
    
    # Efficiency metrics (over successful runs only)
    if successes:
        path_lengths = [e.path_length for e in successes]
        planning_times = [e.planning_time_ms for e in successes]
        replan_counts = [float(e.replans) for e in successes]
        
        path_length_mean = float(np.mean(path_lengths))
        path_length_std = float(np.std(path_lengths))
        path_length_min = float(np.min(path_lengths))
        path_length_max = float(np.max(path_lengths))
        path_length_ci_lower, path_length_ci_upper = _bootstrap_ci(path_lengths, 0.95)
        
        planning_time_mean_ms = float(np.mean(planning_times))
        planning_time_std_ms = float(np.std(planning_times))
        
        replans_mean = float(np.mean(replan_counts))
        replans_std = float(np.std(replan_counts))
    else:
        path_length_mean = float('nan')
        path_length_std = float('nan')
        path_length_min = float('nan')
        path_length_max = float('nan')
        path_length_ci_lower = float('nan')
        path_length_ci_upper = float('nan')
        planning_time_mean_ms = float('nan')
        planning_time_std_ms = float('nan')
        replans_mean = float('nan')
        replans_std = float('nan')
    
    # Safety metrics (over all runs)
    collisions = [e.collision_count for e in episodes]
    nfz_violations = [e.nfz_violations for e in episodes]
    fire_exposures = [e.fire_exposure for e in episodes]
    traffic_proximities = [e.traffic_proximity_time for e in episodes]
    intruder_proximities = [e.intruder_proximity_time for e in episodes]
    
    collision_mean = float(np.mean(collisions))
    collision_rate = float(np.sum([1 for e in episodes if e.collision_count > 0]) / num_seeds)
    nfz_violation_mean = float(np.mean(nfz_violations))
    nfz_violation_rate = float(np.sum([1 for e in episodes if e.nfz_violations > 0]) / num_seeds)
    fire_exposure_mean = float(np.mean(fire_exposures))
    traffic_proximity_mean = float(np.mean(traffic_proximities))
    intruder_proximity_mean = float(np.mean(intruder_proximities))
    
    # Regret (if oracle data available)
    regret_lengths = [e.regret_length for e in episodes if e.regret_length is not None]
    regret_risks = [e.regret_risk for e in episodes if e.regret_risk is not None]
    
    regret_length_mean = float(np.mean(regret_lengths)) if regret_lengths else None
    regret_length_std = float(np.std(regret_lengths)) if regret_lengths else None
    regret_risk_mean = float(np.mean(regret_risks)) if regret_risks else None
    regret_risk_std = float(np.std(regret_risks)) if regret_risks else None
    
    return AggregateMetrics(
        scenario_id=scenario_id,
        planner_id=planner_id,
        num_seeds=num_seeds,
        success_count=success_count,
        success_rate=success_rate,
        path_length_mean=path_length_mean,
        path_length_std=path_length_std,
        path_length_min=path_length_min,
        path_length_max=path_length_max,
        path_length_ci_lower=path_length_ci_lower,
        path_length_ci_upper=path_length_ci_upper,
        planning_time_mean_ms=planning_time_mean_ms,
        planning_time_std_ms=planning_time_std_ms,
        replans_mean=replans_mean,
        replans_std=replans_std,
        collision_mean=collision_mean,
        collision_rate=collision_rate,
        nfz_violation_mean=nfz_violation_mean,
        nfz_violation_rate=nfz_violation_rate,
        fire_exposure_mean=fire_exposure_mean,
        traffic_proximity_mean=traffic_proximity_mean,
        intruder_proximity_mean=intruder_proximity_mean,
        regret_length_mean=regret_length_mean,
        regret_length_std=regret_length_std,
        regret_risk_mean=regret_risk_mean,
        regret_risk_std=regret_risk_std,
        oracle_planner_id=oracle_planner_id,
    )


def _bootstrap_ci(data: list[float], confidence: float = 0.95, num_bootstrap: int = 10000) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    if len(data) < 2:
        val = float(data[0]) if data else float('nan')
        return val, val
    
    bootstrap_means = []
    rng = np.random.default_rng(42)  # Deterministic
    data_arr = np.array(data)
    
    for _ in range(num_bootstrap):
        sample = rng.choice(data_arr, size=len(data), replace=True)
        bootstrap_means.append(float(np.mean(sample)))
    
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(bootstrap_means, alpha))
    upper = float(np.quantile(bootstrap_means, 1 - alpha))
    
    return lower, upper


def save_episode_metrics_jsonl(metrics_list: list[EpisodeMetrics], filepath: Path) -> None:
    """Save episode metrics to JSONL file."""
    with open(filepath, 'w') as f:
        for metrics in metrics_list:
            json_obj = {k: v for k, v in asdict(metrics).items()}
            # Convert None to null
            json_obj = {k: (v if v is not None else None) for k, v in json_obj.items()}
            f.write(json.dumps(json_obj) + '\n')


def save_aggregate_metrics_csv(agg_list: list[AggregateMetrics], filepath: Path) -> None:
    """Save aggregated metrics to CSV file."""
    import csv
    
    if not agg_list:
        return
    
    # Get all field names
    fieldnames = list(asdict(agg_list[0]).keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agg in agg_list:
            row = asdict(agg)
            writer.writerow(row)


def print_aggregate_metrics_table(agg_list: list[AggregateMetrics]) -> None:
    """Print human-readable table of aggregated metrics."""
    if not agg_list:
        print("No metrics to display")
        return
    
    print("\n" + "=" * 150)
    print("BENCHMARK RESULTS: AGGREGATED METRICS")
    print("=" * 150)
    
    print(f"\n{'Scenario':<40s} {'Planner':<20s} {'Success':<10s} {'Path Len':<15s} {'Plan Time':<12s} {'Replans':<10s} {'Fire Exp':<12s}")
    print("-" * 150)
    
    for agg in agg_list:
        success_str = f"{agg.success_rate*100:.1f}% ({agg.success_count}/{agg.num_seeds})"
        if np.isnan(agg.path_length_mean):
            path_str = "N/A"
            time_str = "N/A"
            replan_str = "N/A"
        else:
            path_str = f"{agg.path_length_mean:.1f}±{agg.path_length_std:.1f}"
            time_str = f"{agg.planning_time_mean_ms:.1f}ms"
            replan_str = f"{agg.replans_mean:.1f}±{agg.replans_std:.1f}"
        
        fire_str = f"{agg.fire_exposure_mean:.2f}"
        
        print(
            f"{agg.scenario_id:<40s} {agg.planner_id:<20s} {success_str:<10s} "
            f"{path_str:<15s} {time_str:<12s} {replan_str:<10s} {fire_str:<12s}"
        )
    
    print("=" * 150 + "\n")
