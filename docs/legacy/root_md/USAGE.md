# Usage Guide

## CLI Reference

### Basic Benchmark

```bash
uavbench --scenarios <ids> --planners <ids> --trials <n> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scenarios` | `urban_easy` | Comma-separated scenario IDs |
| `--planners` | `astar` | Comma-separated planner IDs |
| `--trials` | `1` | Trials per scenario/planner pair |
| `--seed-base` | `0` | Base random seed for reproducibility |
| `--metrics` | `path_length,success_rate,constraint_violations` | Metrics to display |
| `--play` | | Open window: `best` or `worst` path |
| `--fps` | `8` | Playback speed |
| `--save-videos` | | Save MP4: `best`, `worst`, or `both` |
| `--save-figures` | | Save PNG figures to directory |
| `--with-dynamics` | | Include fire/traffic overlays in viz |
| `--fail-fast` | | Stop on first exception |

### Examples

```bash
# Single OSM scenario
uavbench --scenarios osm_athens_wildfire_easy --planners astar --trials 5

# Multiple scenarios
uavbench --scenarios osm_athens_wildfire_easy,osm_athens_emergency_easy \
    --planners astar --trials 3

# With visualization
uavbench --scenarios osm_athens_crisis_hard --planners astar --trials 1 \
    --save-figures outputs/ --with-dynamics

# Reproducible run
uavbench --scenarios osm_athens_sar_hard --planners astar --trials 10 --seed-base 42
```

### Scenario Pack Runner

Runs all 20 OSM scenarios with formatted comparison table and CSV export:

```bash
python tools/benchmark_scenario_pack.py --planners astar --trials 3 --output outputs/
```

Output CSV columns: `scenario, planner, trials, success_rate, avg_path_length, avg_violations, path_optimality, planning_time_ms, path_smoothness, risk_exposure, time_s`

### Paper Figure Generator

Produces 5 publication-quality figures:

```bash
python tools/generate_paper_figures.py
```

## Custom Scenarios

Create a YAML file in `src/uavbench/scenarios/configs/`:

```yaml
name: "my_custom_scenario"
domain: "urban"
difficulty: "medium"

# Map source: "osm" or "synthetic"
map_source: "osm"
osm_tile_id: "downtown"
map_size: 500
max_altitude: 8

# Synthetic map params (ignored for OSM)
building_density: 0.0
building_level: 0

# Start/goal placement
start_altitude: 2
safe_altitude: 6
min_start_goal_l1: 200

# Dynamic layers
enable_fire: true
fire_ignition_points: 5
wind_direction: 0.0
wind_speed: 0.5

enable_traffic: true
num_emergency_vehicles: 8

wind: "none"
traffic: "none"
```

Run it:
```bash
uavbench --scenarios my_custom_scenario --planners astar --trials 3
```

## Custom Planners

1. Create a planner class implementing the `plan(start, goal)` interface:

```python
# src/uavbench/planners/my_planner.py

class MyPlanner:
    def __init__(self, heightmap, no_fly_mask):
        self.heightmap = heightmap
        self.no_fly = no_fly_mask

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Return list of (x, y) waypoints from start to goal, or [] if no path."""
        # Your planning algorithm here
        ...
```

2. Register it in `src/uavbench/planners/__init__.py`:

```python
from uavbench.planners.my_planner import MyPlanner

PLANNERS = {
    "astar": AStarPlanner,
    "my_planner": MyPlanner,
}
```

3. Use it:
```bash
uavbench --scenarios osm_athens_wildfire_easy --planners my_planner --trials 5
```

## Programmatic Usage

```python
from uavbench.cli.benchmark import run_planner_once
from uavbench.metrics.operational import compute_all_metrics

# Run a single trial
result = run_planner_once("osm_athens_wildfire_easy", "astar", seed=42)

# Compute metrics
metrics = compute_all_metrics(result)
print(f"Path length: {result['path_length']}")
print(f"Optimality: {metrics['path_optimality']:.3f}")
print(f"Smoothness: {metrics['path_smoothness']:.3f}")
print(f"Risk exposure: {metrics['risk_exposure_sum']:.4f}")
print(f"Planning time: {metrics['planning_time_ms']:.1f}ms")
```

## Available Scenarios

### Synthetic (no tiles required)
- `urban_easy`, `urban_medium`, `urban_hard`

### OSM Athens (requires tiles in `data/maps/`)
See [scenarios README](src/uavbench/scenarios/configs/README_SCENARIOS.md) for full details.

| ID | Tile | Dynamics |
|----|------|----------|
| `osm_athens_wildfire_{easy,medium,hard}` | penteli | Fire |
| `osm_athens_emergency_{easy,medium,hard}` | downtown | Traffic |
| `osm_athens_port_{easy,medium,hard}` | piraeus | Static |
| `osm_athens_crisis_hard` | downtown | Fire + Traffic |
| `osm_athens_sar_{easy,medium,hard}` | penteli | Static |
| `osm_athens_infrastructure_{easy,medium,hard}` | downtown | Static |
| `osm_athens_border_{easy,medium,hard}` | penteli | Static |
| `osm_athens_comms_denied_hard` | downtown | Static |
