# UAVBench AI Coding Agent Instructions

## Project Overview
**UAVBench** is a reproducible UAV (Unmanned Aerial Vehicle) planning benchmark framework for evaluating path planning algorithms in realistic urban environments. It's structured as a Python package built on Gymnasium (RL environments framework) with extensible scenario configurations.

## Architecture & Key Components

### Core Design Pattern: Environment Abstraction
UAVBench follows Gymnasium's `gym.Env` API. All domain logic extends from two base classes:
- **`uavbench.envs.base.UAVBenchEnv`** — Abstract base handling common RL concerns:
  - Deterministic seeding via per-instance `np.random.Generator` (never use `np.random` directly)
  - Episode lifecycle: trajectory + event logging
  - Validation guards (e.g., enforcing `terminated` and `truncated` are never both True)
- **`uavbench.envs.urban.UrbanEnv`** — Concrete domain implementation for 2.5D urban environments

### State Representation (Urban Domain)
- **Observation space (7D float32)**: `[x, y, z, goal_x, goal_y, goal_z, heightmap[x,y]]`
  - Position + goal are in grid coordinates (0..map_size-1)
  - `z` is discrete altitude levels (0..max_altitude), **not continuous meters**
  - All values are always within bounds—validated by test assertions
- **Action space (Discrete-6)**: 0=up, 1=down, 2=left, 3=right, 4/5=altitude change

### Scenario Configuration System
Scenarios are defined as YAML + Pydantic models:
- **YAML files** → `src/uavbench/scenarios/configs/*.yaml` (e.g., `urban_easy.yaml`)
- **Loader**: `uavbench.scenarios.loader.load_scenario(path)` → `ScenarioConfig` object
- **Schema**: `uavbench.scenarios.schema` defines enums:
  - `Domain`: URBAN, MOUNTAIN, COASTAL, INFRASTRUCTURE, ATHENSFIR
  - `Difficulty`: EASY, MEDIUM, HARD (adjusts map density + obstacles)
  - `WindLevel`, `TrafficLevel`: model environmental stress (NONE, LOW, MEDIUM, HIGH)
  - `NoFlyZone`: axis-aligned rectangles in grid coordinates

### Urban Environment Specifics
The `UrbanEnv._reset_impl()` builds scenario state with difficulty-based tweaks:
- **EASY**: Base building density from config (e.g., 30%)
- **MEDIUM**: Add `extra_density_medium` (e.g., +10%) more random buildings
- **HARD**: Extra buildings + central no-fly circle; both are separate constraints (heightmap vs. no-fly mask)
- **Collision detection**: `z <= heightmap[x,y]` OR grid is in no-fly zone
- **Free cell guarantee**: Environment raises `ValueError` if not enough free cells to place start/goal with min Manhattan distance enforced

## Developer Workflows

### Running Tests
```bash
pytest tests/
pytest tests/test_urban_env_basic.py -v  # Specific test file
pytest -k "trajectory" -v                 # Pattern matching
```

### CLI Entrypoint
```bash
uavbench-benchmark --scenarios=urban_easy,urban_medium \
                   --planners=astar,rrtstar \
                   --trials=3 \
                   --seed-base=42
```
Current implementation parses args and prints them; planner/metrics computation is TODO.

### Development Setup
- **Python requirement**: >=3.10
- **Dependencies**: `gymnasium>=0.29`, `numpy>=1.26`, `pydantic>=2.5`
- **Dev tools**: pytest, mypy, sphinx (see `pyproject.toml`)
- **Package layout**: Editable install via setuptools; entry point is `uavbench-benchmark`

## Critical Patterns & Conventions

### Random Number Generation
- **NEVER** use `np.random.default_rng()` or `np.random.seed()` directly in domain logic
- Always use the **per-instance `self._rng`** (initialized in `UAVBenchEnv.__init__`)
- Example:
  ```python
  base_mask = (self._rng.random((H, W)) < building_density)
  free_idx = self._rng.choice(len(free_cells))
  ```
- This ensures determinism when `env.reset(seed=...)` is called

### Trajectory & Event Logging
- **Trajectory**: Auto-logged by `UAVBenchEnv.step()` — every transition as dict with step#, action, obs, reward, termination flags, info
- **Events**: Domain logs high-level facts (collisions, violations) via `self.log_event(event_type, **payload)`
- **Return copies**: Properties return shallow copies to prevent external mutation (e.g., `return list(self._trajectory)`)

### Type Safety & Validation
- Use **Pydantic models** for all config objects (see `ScenarioConfig` in `schema.py`)
- **Gymnasium returns**: `(obs, reward, terminated, truncated, info)` — **obs must be valid in `observation_space`**
- **Guard clauses**: Raise descriptive `ValueError` early (e.g., invalid map_size, insufficient free cells)
- **Normalize types**: Always convert numpy scalars to Python natives for JSON/metrics (e.g., `float(reward)`, `bool(terminated)`)

### Map Representation
- Grids are indexed as `heightmap[y, x]` (row-major, following numpy convention)
- `argwhere()` returns `[y, x]` pairs; extract carefully
- Heightmap values: 0.0 = free cell, `building_level` (float) = building; separate `no_fly_mask` for restricted zones
- Building levels are in discrete **altitude levels** (0..max_altitude), not meters

## Code Locations & Exemplars

| Pattern | Location | Example |
|---------|----------|---------|
| Base env class | `src/uavbench/envs/base.py` | `UAVBenchEnv`, seeding discipline |
| Domain implementation | `src/uavbench/envs/urban.py` | `UrbanEnv._reset_impl()`, collision logic |
| Config schema | `src/uavbench/scenarios/schema.py` | `ScenarioConfig`, enums |
| Scenario loading | `src/uavbench/scenarios/loader.py` | `load_scenario()` YAML→Pydantic |
| CLI | `src/uavbench/cli/benchmark.py` | Argument parsing (metrics computation TODO) |
| Test trajectory validation | `tests/test_urban_env_basic.py` | Asserting observations stay in bounds |
| Test config loading | `tests/test_scenario_basic.py` | Difficulty-specific assertions |

## Integration Points & Extensibility

1. **Adding new domains** (e.g., `MountainEnv`):
   - Subclass `UAVBenchEnv`, implement `_reset_impl()` and `_step_impl()`
   - Define domain-specific enums if needed; extend `Domain` enum
   - Add YAML config files and tests
   - Register in CLI if needed

2. **Planners** (future):
   - Will consume `ScenarioConfig` and `UrbanEnv` instances
   - Should respect determinism (use fixed seed for reproducible benchmarks)
   - Metrics computed from `env.trajectory` and `env.events`

3. **Metrics** (future):
   - Derive from trajectory (path length = trajectory length × step_distance)
   - Derive from events (safety = collision count)
   - Aggregate over trials for statistical summaries

## Common Pitfalls

- ❌ Mutating trajectory or events externally; ✅ use the immutable copies returned by properties
- ❌ Forgetting `z <= h(x,y)` is a level-based collision (not continuous altitude); ✅ think in discrete altitude levels
- ❌ Using global random state; ✅ always use `self._rng`
- ❌ Allowing both `terminated=True` and `truncated=True`; ✅ base class enforces this guard
- ❌ Grid indexing confusion (x, y) vs (y, x); ✅ follow numpy row-major convention: `grid[y, x]`
