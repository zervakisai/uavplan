# API Reference

## Environments

### `UAVBenchEnv` (base class)

```python
from uavbench.envs.base import UAVBenchEnv
```

Abstract base class extending `gymnasium.Env`. Provides:
- Per-instance RNG (`self._rng`) — all randomness must use this
- Step counting (`self._step_count`)
- Trajectory logging (`self._trajectory`)
- Event logging (`self._events`, `self.log_event()`)

**Methods:**
| Method | Description |
|--------|-------------|
| `reset(seed=None, options=None)` | Reset episode, re-seed RNG, delegate to `_reset_impl` |
| `step(action)` | Increment step counter, delegate to `_step_impl`, log trajectory |
| `log_event(event_type, **payload)` | Record a structured event |
| `get_trajectory()` | Return copy of trajectory list |
| `get_events()` | Return copy of events list |

### `UrbanEnv`

```python
from uavbench.envs.urban import UrbanEnv
```

2.5D urban grid environment with buildings, no-fly zones, and dynamic layers.

**Constructor:** `UrbanEnv(config: ScenarioConfig)`

**Spaces:**
- Observation: `Box(7,)` — `[x, y, z, gx, gy, gz, h(x,y)]`
- Action: `Discrete(6)` — 0:up 1:down 2:left 3:right 4:ascend 5:descend

**Key methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `reset(seed=, options=)` | `(obs, info)` | Load map, init dynamics, sample start/goal |
| `step(action)` | `(obs, reward, terminated, truncated, info)` | Move agent, advance dynamics |
| `export_planner_inputs()` | `(heightmap, no_fly, start_xy, goal_xy)` | Extract planner-ready data |

**Step info dict:**
```python
{
    "reached_goal": bool,
    "distance_to_goal": float,
    "attempted_building_collision": bool,
    "attempted_no_fly": bool,
    "accepted_move": bool,
    "agent_pos": (x, y, z),
    "goal_pos": (x, y, z),
    "fire_active": bool,
    "traffic_active": bool,
    "fire_exposure": bool,
    "traffic_proximity": bool,
    "fire_cells": int,
    "vehicles_near": int,
}
```

**Collision model:** `z <= heightmap[y, x]` means collision. Height 0 = free space.

**Height conversion (OSM):** `level = ceil(meters / 10.0)`, clamped to `max_altitude`.

## Scenarios

### `ScenarioConfig`

```python
from uavbench.scenarios.schema import ScenarioConfig
```

Frozen dataclass with validation. Key fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Scenario identifier |
| `domain` | `Domain` | required | `"urban"` |
| `difficulty` | `Difficulty` | required | `"easy"` / `"medium"` / `"hard"` |
| `map_source` | `str` | `"synthetic"` | `"synthetic"` or `"osm"` |
| `osm_tile_id` | `str\|None` | `None` | `"downtown"`, `"penteli"`, `"piraeus"` |
| `map_size` | `int` | `25` | Grid dimension (overridden by tile for OSM) |
| `max_altitude` | `int` | `3` | Maximum flight level |
| `safe_altitude` | `int` | `3` | Start/goal altitude |
| `min_start_goal_l1` | `int` | `10` | Minimum Manhattan distance |
| `enable_fire` | `bool` | `False` | Activate fire spread model |
| `enable_traffic` | `bool` | `False` | Activate traffic model |
| `fire_ignition_points` | `int` | `3` | Number of initial fires |
| `num_emergency_vehicles` | `int` | `5` | Number of traffic vehicles |
| `wind_direction` | `float` | `0.0` | Wind direction in radians (0 = North) |
| `wind_speed` | `float` | `0.5` | Wind speed [0, 1] |

### `load_scenario`

```python
from uavbench.scenarios.loader import load_scenario
cfg = load_scenario(Path("src/uavbench/scenarios/configs/osm_athens_wildfire_easy.yaml"))
```

## Planners

### `AStarPlanner`

```python
from uavbench.planners import PLANNERS
planner = PLANNERS["astar"](heightmap, no_fly_mask)
path = planner.plan(start_xy, goal_xy)  # list[(x, y)] or []
```

**Constructor:** `AStarPlanner(heightmap, no_fly, config=None)`

**Config:** `AStarConfig(allow_diagonal=False, block_buildings=True, max_expansions=200_000)`

**Returns:** List of `(x, y)` tuples from start to goal, or `[]` if no path found.

### Registry

```python
from uavbench.planners import PLANNERS
print(PLANNERS.keys())  # dict_keys(['astar'])
```

## Dynamics

### `FireSpreadModel`

```python
from uavbench.dynamics.fire_spread import FireSpreadModel
```

Cellular automaton with land-use-dependent spread probabilities and wind effects.

| Method/Property | Description |
|-----------------|-------------|
| `__init__(landuse_map, roads_mask, wind_dir, wind_speed, rng, n_ignition)` | Initialize with random ignition points |
| `step(dt=1.0)` | Advance one timestep |
| `fire_mask` | Boolean array of currently burning cells |
| `burned_mask` | Boolean array of burned-out cells |
| `total_affected` | Total cells ever on fire |

Spread probabilities: Forest=0.3, Urban=0.1, Industrial=0.05, Water=0.0, Empty=0.02. Roads act as firebreaks.

### `TrafficModel`

```python
from uavbench.dynamics.traffic import TrafficModel
```

Pixel-space vehicle movement on road network.

| Method/Property | Description |
|-----------------|-------------|
| `__init__(roads_mask, num_vehicles, rng)` | Spawn vehicles on road cells |
| `step(dt=1.0)` | Move vehicles 1 pixel toward destinations |
| `vehicle_positions` | `ndarray [N, 2]` of `(y, x)` positions |
| `get_occupancy_mask(shape, buffer_radius=3)` | Boolean buffer around vehicles |

## Metrics

### `compute_all_metrics`

```python
from uavbench.metrics.operational import compute_all_metrics
result = run_planner_once("osm_athens_wildfire_easy", "astar", seed=42)
metrics = compute_all_metrics(result)
```

Returns dict with all metrics computed from a single trial result:

| Metric | Category | Description |
|--------|----------|-------------|
| `nfz_violation_count` | Safety | Path cells on no-fly zones |
| `building_violation_count` | Safety | Path cells on buildings |
| `risk_exposure_sum` | Safety | Cumulative risk from tile risk_map |
| `path_optimality` | Efficiency | `manhattan_dist / path_length` (1.0 = optimal) |
| `planning_time_ms` | Efficiency | Wall-clock planning time |
| `steps_per_meter` | Efficiency | `path_length / manhattan_dist` |
| `success` | Feasibility | 1.0 if path found with no violations |
| `constraint_violation_rate` | Feasibility | `violations / path_length` |
| `path_smoothness` | Feasibility | `1 - direction_changes / (n-2)` |

### Individual metric functions

```python
from uavbench.metrics.operational import (
    compute_safety_metrics,
    compute_efficiency_metrics,
    compute_feasibility_metrics,
)
```

## CLI / Benchmark

### `run_planner_once`

```python
from uavbench.cli.benchmark import run_planner_once
result = run_planner_once(scenario_id, planner_id, seed=42)
```

Returns dict:
```python
{
    "scenario": str,
    "planner": str,
    "seed": int,
    "success": bool,
    "constraint_violations": int,
    "path_length": int,
    "path": list[(x, y)] | None,
    "heightmap": ndarray,
    "no_fly": ndarray,
    "start": (x, y),
    "goal": (x, y),
    "planning_time": float,  # seconds
    "map_source": str,
    "osm_tile_id": str | None,
    "config": ScenarioConfig,
}
```

### `aggregate`

```python
from uavbench.cli.benchmark import aggregate
metrics = aggregate(list_of_results)
```

Averages all operational metrics across trials. Returns dict with `success_rate`, `avg_path_length`, `avg_path_optimality`, `avg_path_smoothness`, etc.

## Data Formats

### `.npz` Tile Files

Located in `data/maps/{tile_id}.npz`. Arrays:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `heightmap` | `(500, 500)` | `float32` | Building heights in meters |
| `roads_mask` | `(500, 500)` | `bool` | Road network |
| `landuse_map` | `(500, 500)` | `int8` | Land use categories |
| `risk_map` | `(500, 500)` | `float32` | Risk heatmap [0, 1] |
| `nfz_mask` | `(500, 500)` | `bool` | No-fly zones |
| `roads_graph_nodes` | `(N, 3)` | `float64` | `[node_id, lat, lon]` |
| `roads_graph_edges` | `(M, 2)` | `int64` | `[source_id, target_id]` |

### Coordinate Conventions

- **Path entries:** `(x, y)` — column, row
- **Heightmap indexing:** `heightmap[y, x]` — row, column
- **TrafficModel positions:** `(y, x)` — row, column
- **Scatter plots:** `scatter(x_array, y_array)` — standard matplotlib
