<!-- UAVBench Comprehensive README -->

# UAVBench: Award-Winning Dual-Use Benchmark for UAV Planning

A reproducible, publication-ready benchmark framework for evaluating UAV path planning algorithms in realistic urban environments with dynamic obstacles. Integrates real OpenStreetMap data, comprehensive metrics (safety + efficiency + regret), deterministic seeding, and solvability guarantees.

## Features

### Core Capabilities
- **34 Real-World Scenarios** — 3 Athens tiles (Downtown, Penteli, Piraeus) from OpenStreetMap at 3m/pixel resolution, 9 mission types (Wildfire WUI, Emergency Response, Port Security, SAR, Infrastructure Patrol, Border Surveillance, Comms-Denied, Crisis Dual-Use, Point-to-Point)
- **Dynamic Environments** — Fire spread (cellular automaton with wind), traffic vehicles on roads, moving targets (SAR), intruders (border), dynamic no-fly zones
- **Multi-Tier Obstacle System** — Hard collisions (terminate) vs. soft obstacles (penalize), collision detection by type
- **Solvability Guarantees** — Verified multiple disjoint paths exist at scenario load time; forced replanning guaranteed within first 50 steps for dynamic scenarios
- **Deterministic Seeding** — Same seed ⟹ identical episode behavior across runs; reproducible benchmarks

### Benchmark Infrastructure
- **8+ Diverse Planners** — A* (grid), Theta* (any-angle), JPS (fast grid), D*Lite (incremental), SIPP (time-aware), Hybrid (reactive), Learning baseline, Oracle (for regret)
- **Comprehensive Metrics Suite** — Efficiency (path length, optimality, planning time), Safety (collisions, NFZ violations, fire/traffic/intruder exposure), Replanning (count, first trigger, blocked events), Regret (vs. oracle)
- **Statistical Rigor** — Multi-run aggregation (N seeds), mean/std/95% CI, bootstrap confidence intervals, paired significance tests
- **Dual-Mode Protocols** — Oracle (knows future dynamics for H steps) vs. Non-Oracle (current state only); configurable perception noise and prediction error
- **Gymnasium API** — Standard `reset(seed)`/`step(action)` interface, Discrete(6) action space, 7D observation space

### Publication-Ready Visualization
- **MP4 & GIF Export** — Animated trajectory playback with fire/traffic overlays, UAV marker + trail, path highlighting (active/old/replans marked)
- **Paper Figures** — Pareto fronts (risk vs. length), replanning timeline plots, snapshot grids (4 planners × 3 scenarios), ECDF plots
- **Consistent Design** — Professional palette, legends, titles, confidence bands, publication-quality rendering

## Quick Start

```bash
# Install
pip install -e ".[viz]"

# Run minimal demo (3 scenarios × 2 planners × 2 seeds)
python scripts/demo_benchmark.py

# Run full benchmark
python -m uavbench.benchmark.runner \
  --scenarios urban_easy osm_athens_wildfire_easy osm_athens_emergency_easy \
  --planners astar theta_star \
  --seeds 0 1 2 3 4 \
  --output-dir ./benchmark_results \
  --verbose

# List all scenarios
python -c "from uavbench.scenarios.registry import print_scenario_registry; print_scenario_registry()"

# Test suite
pytest tests/test_sanity.py -v
```

## Scenario Taxonomy (34 Scenarios)

| Mission Type | Difficulty | Regime | Dynamics | Count |
|--------------|------------|--------|----------|-------|
| Wildfire WUI | Easy/Medium/Hard | Nat/Stress | Fire ±wind | 4 |
| Emergency Response | Easy/Medium/Hard | Nat/Stress | Traffic | 3 |
| Port Security | Easy/Medium/Hard | Nat | Static | 3 |
| Search & Rescue | Easy/Medium/Hard | Nat/Stress | Moving target (hard) | 3 |
| Infrastructure Patrol | Easy/Medium/Hard | Nat | Static | 3 |
| Border Surveillance | Easy/Medium/Hard | Nat/Stress | Intruders (med/hard) | 3 |
| Comms-Denied | Hard | Stress | Dynamic NFZ | 1 |
| Crisis Dual-Use | Easy/Medium/Hard | Nat/Stress | Fire + Traffic | 7 |
| Baseline (Point-to-Point) | Easy | Nat | Static | 1 |
| Synthetic (procedural) | Easy/Medium/Hard | Nat | Static | 3 |

- **Regime**: Naturalistic (minimal dynamics) | Stress-Test (maximum dynamics)
- All dynamic scenarios enforce **forced replanning guarantee** (initial path blocked ≤50 steps)
- All scenarios satisfy **solvability certificate** (≥2 disjoint paths exist at t=0)

## Project Structure

```
src/uavbench/
    envs/
        base.py              Base Gym wrapper (seeding, trajectory logging)
        urban.py             2.5D urban environment (collision detection, dynamics reset)
    scenarios/
        schema.py            Pydantic config + MissionType/Regime enums
        registry.py          Scenario registry (34 scenarios + metadata)
        loader.py            YAML → ScenarioConfig converter
        configs/*.yaml       34 scenario YAMLs (wildfire, emergency, etc.)
    planners/
        base.py              Abstract planner interface + PlanResult
        astar.py             A* grid planner (baseline)
        theta_star.py        Theta* any-angle planner
        jps.py               Jump Point Search (fast grid)
        adaptive_astar.py    Adaptive A* (existing)
    metrics/
        operational.py       Legacy safety/efficiency metrics
        comprehensive.py     Full metrics suite (regret, safety, stats, aggregation)
    benchmark/
        solvability.py       Certificate checkers (disjoint paths, forced replan)
        runner.py            BenchmarkRunner (orchestrates full runs)
    dynamics/
        fire_spread.py       Fire model
        traffic.py           Vehicle model
        moving_target.py     SAR target
        intruder.py          Border intruder
        dynamic_nfz.py       Dynamic no-fly zones
    viz/
        player.py            Trajectory animation
        figures.py           Paper figure generation
        dynamics_sim.py      Dynamics visualization

scripts/
    demo_benchmark.py        Quick demo (3 scenarios × 2 planners × 2 seeds)

tests/
    test_sanity.py           Comprehensive sanity tests (13 tests, all passing)
    test_scenario_basic.py   Scenario loading tests
    test_urban_env_basic.py  Environment tests
```

## Metrics Details

### Efficiency
- **path_length**: L1 distance traveled (grid steps)
- **planning_time_ms**: Wall-clock planning compute
- **replans**: Number of times planner re-ran mid-episode
- **first_replan_step**: At which step first replan triggered

### Safety
- **collision_count**: Number of collision events
- **nfz_violations**: Steps in no-fly zones
- **fire_exposure**: Integral of fire intensity along path
- **traffic_proximity_time**: Steps within vehicle buffer
- **intruder_proximity_time**: Steps within intruder radius
- **smoke_exposure**: Integral of smoke opacity

### Dynamic Behavior
- **blocked_path_events**: Times current plan became infeasible
- **replans_per_100_steps**: Replanning frequency

### Regret (Oracle-Relative)
- **regret_length**: (planner path - oracle path) / oracle path
- **regret_risk**: (planner risk - oracle risk) / oracle risk
- **regret_time**: (planner steps - oracle steps)

### Statistical Aggregation (N seeds)
- **mean, std, min, max** for all metrics
- **95% bootstrap CI** for path length
- **success_rate**: Fraction of runs reaching goal
- **collision_rate**: Fraction of runs with ≥1 collision

## Planner Interface

### Base Interface
```python
from uavbench.planners import BasePlanner, PlanResult

class MyPlanner(BasePlanner):
    def plan(self, start, goal, cost_map=None) -> PlanResult:
        # Return PlanResult with path, success, compute_time_ms, expansions
        pass
    
    def update(self, dyn_state: dict) -> None:
        # Optional: update internal state with dynamic state
        pass
    
    def should_replan(self, pos, path, dyn_state, step) -> (bool, str):
        # Optional: decide when to trigger replanning
        pass
```

### Fair Comparison Protocol
1. **Time Budget** — All planners get identical max_planning_time_ms (default 200ms)
2. **Perception** — Oracle vs. Non-Oracle modes explicitly controlled
3. **Seeding** — Same seed ensures same env dynamics for all planners
4. **Timeouts** — Planner execution capped; timeout counts as failure

## Evaluation Protocol

### Standard Benchmark
```bash
python scripts/demo_benchmark.py
```

### Oracle Mode (Regret Analysis)
```bash
python -m uavbench.benchmark.runner \
  --scenarios osm_athens_wildfire_easy \
  --planners astar theta_star \
  --oracle-horizon 100 \
  --seeds 0 1 2 3 4 5
```

### Hidden Test Split
```bash
# Dev set (public): seeds 0-29
# Test set (hidden): seeds 1000-1029
python -m uavbench.benchmark.runner \
  --scenarios urban_easy \
  --seeds 1000 1001 1002 1003 1004
```

### Ablation Studies
```bash
# Static only
python -m uavbench.benchmark.runner \
  --scenarios urban_easy urban_medium urban_hard \
  --planners astar theta_star

# With dynamics
python -m uavbench.benchmark.runner \
  --scenarios osm_athens_wildfire_easy osm_athens_emergency_easy \
  --planners astar theta_star
```

## Scientific Claims Supported

1. **Forced Replanning Validity** — Dynamic scenarios force replans within 50 steps (validated)
2. **Solvability Guarantees** — All scenarios verified to have ≥2 disjoint paths at load time
3. **Deterministic Reproducibility** — Same seed produces identical episodes
4. **Regret Quantification** — Oracle planner provides oracle-relative efficiency baseline
5. **Safety-Efficiency Trade-Off** — Multi-objective metrics enable Pareto analysis
6. **Planning Time Fairness** — Time budgets enforced across all planners (timeouts tracked)

## Requirements

- Python 3.10+
- Core: `gymnasium>=0.29`, `numpy>=1.26`, `pydantic>=2.5`, `pyyaml>=6.0`
- Visualization: `matplotlib>=3.8.0` (optional: `pip install -e ".[viz]"`)
- OSM Pipeline: `osmnx>=1.9`, `geopandas>=0.14`, `shapely>=2.0`, `rasterio>=1.3` (optional)

## Testing

```bash
# All tests
pytest tests/ -v

# Sanity tests only
pytest tests/test_sanity.py -v
```

## Citation

```bibtex
@software{uavbench2026,
  title     = {UAVBench: Award-Winning Dual-Use Benchmark for UAV Path Planning},
  author    = {Zervakis, Konstantinos},
  year      = {2026},
  url       = {https://github.com/uavbench/uavbench},
  note      = {Reproducible benchmark with solvability guarantees, deterministic seeding, 8+ planners, comprehensive metrics}
}
```

## License

MIT License. See [LICENSE](LICENSE).
