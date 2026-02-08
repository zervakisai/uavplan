# UAVBench Project Analysis
## Complete Architecture Overview & File Interconnections

**Project Type**: Python package (Gymnasium-based RL environment)  
**Current Version**: 0.0.1  
**Python Requirement**: >=3.10  
**Core Dependencies**: numpy, gymnasium, pydantic, matplotlib, pyyaml  

---

## 1. PROJECT STRUCTURE & ARCHITECTURE

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     UAVBench Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌──────────────────┐             │
│  │  CLI Layer      │────────▶│  Benchmark.py    │             │
│  │  (Entry Point)  │         │  (Orchestration) │             │
│  └─────────────────┘         └──────────────────┘             │
│           │                           │                        │
│           │                           ├──────────┐            │
│           │                           │          │            │
│           ▼                           ▼          ▼            │
│  ┌──────────────────────┐   ┌──────────────┐  ┌──────────┐  │
│  │  Scenario System     │   │  Planners    │  │ Viz/Play │  │
│  │  ┌────────────────┐  │   │  ┌────────┐  │  │          │  │
│  │  │ schema.py      │  │   │  │ astar  │  │  │ player   │  │
│  │  │ (Pydantic)     │  │   │  │ .py    │  │  │ .py      │  │
│  │  ├────────────────┤  │   │  └────────┘  │  │          │  │
│  │  │ loader.py      │  │   │              │  └──────────┘  │
│  │  │ (YAML→Config)  │  │   │              │                │
│  │  └────────────────┘  │   └──────────────┘                │
│  └──────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────────────────────────────┐               │
│  │         Environment Layer                │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  base.py (UAVBenchEnv ABC)        │  │               │
│  │  │  - Seeding discipline             │  │               │
│  │  │  - Trajectory logging             │  │               │
│  │  │  - Event system                   │  │               │
│  │  └────────────────────────────────────┘  │               │
│  │           ▲                               │               │
│  │           │ (inherits)                    │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  urban.py (UrbanEnv)               │  │               │
│  │  │  - 2.5D urban simulation           │  │               │
│  │  │  - Building/no-fly collision       │  │               │
│  │  │  - Discrete altitude levels        │  │               │
│  │  └────────────────────────────────────┘  │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. CORE LAYERS

### 2.1 Environment Layer (Gym API Foundation)

#### **File: `src/uavbench/envs/base.py`** (139 lines)

**Role**: Abstract base class for all UAVBench environments, implementing Gymnasium API

**Key Responsibilities**:
- **Seeding Discipline**: Maintains per-instance `self._rng` (np.random.Generator) for deterministic randomness
  - Initialized in `__init__` with `np.random.default_rng()`
  - Re-seeded in `reset(seed=...)` 
  - **Critical**: Domain logic MUST use `self._rng`, never global `np.random`
  
- **Episode Lifecycle Management**:
  - `reset(seed=None)` → calls `_reset_impl()` (implemented by subclasses)
  - `step(action)` → calls `_step_impl()` (implemented by subclasses)
  - Auto-increments `_step_count` after each step
  - Maintains trajectory log (`_trajectory` list)
  
- **Trajectory Logging**:
  - Every step logged to `_trajectory` as dict: `{"step", "action", "obs", "reward", "terminated", "truncated", "info"}`
  - Values normalized to Python types (float, bool) for JSON/metrics stability
  
- **Event System**:
  - `log_event(event_type, **payload)` for structured logging (collisions, violations)
  - Events stored in `_events` list with timestamp/step index
  
- **Validation Guards**:
  - Enforces `terminated` and `truncated` never both True
  - Checks observation space containment at Python level

**Public API**:
```python
env.reset(seed=42)                    # Returns (obs, info)
obs, r, term, trunc, info = env.step(action)  # Gymnasium API
env.trajectory                         # Property: list of dicts
env.events                             # Property: list of events
env.log_event("collision", x=5, y=3)  # Structured event
```

**Design Pattern**: Template Method
- Base class defines `reset()` and `step()` flow
- Subclasses implement `_reset_impl()` and `_step_impl()`

---

#### **File: `src/uavbench/envs/urban.py`** (369 lines)

**Role**: Concrete implementation for 2.5D Urban environment with buildings and no-fly zones

**State Representation** (7D observation space):
```
obs = [x, y, z, goal_x, goal_y, goal_z, heightmap[y,x]]
      (grid coords)      (grid coords)  (current terrain height)
```

**Key Concepts**:

1. **Altitude Model (Discrete Levels)**:
   - Z-axis is discrete: 0 to `max_altitude` (default 3 levels)
   - Buildings also occupy discrete levels (heightmap values: 0 or `building_level`)
   - **Collision rule**: `z <= heightmap[y,x]` OR in no-fly zone → blocked
   - **Important**: NOT continuous altitude in meters—discrete level-based

2. **Map Representation**:
   - 2D grid (x, y) with size `map_size × map_size` (default 25×25)
   - `_heightmap`: 2D array [H,W] of building heights (0.0 = free, building_level = obstacle)
   - `_no_fly_mask`: 2D boolean array [H,W] for restricted zones
   - Grid indexing: `heightmap[y, x]` (row-major, NumPy convention)

3. **Action Space** (Discrete-6):
   - 0: Move up (y-1)
   - 1: Move down (y+1)
   - 2: Move left (x-1)
   - 3: Move right (x+1)
   - 4: Ascend (z+1)
   - 5: Descend (z-1)
   - All actions clamp to bounds

4. **Scenario Generation** (`_reset_impl()`):
   - **EASY**: Base building density from config (e.g., 30%)
   - **MEDIUM**: Extra buildings added (+10% more random buildings)
   - **HARD**: Even more buildings (+20%) + central no-fly circle
   - Start position: Always at `safe_altitude` (default level 3)
   - Goal position: Sampled with minimum Manhattan distance (L1 norm)
   - **Free cell guarantee**: Must have ≥2 free cells to place start/goal

5. **Reward Structure** (`_step_impl()`):
   - Base: `-1.0` per step (encourage short paths)
   - Progress shaping: `+0.2 × (prev_dist - new_dist)` (dense signal)
   - Collision penalty: `-5.0` for building, `-8.0` for no-fly
   - Goal bonus: `+50.0` when reaching goal
   - Weighted L1 distance: `abs(x-gx) + abs(y-gy) + 1.0*abs(z-gz)`

6. **Termination**:
   - `terminated=True` when agent reaches goal
   - `truncated=True` when step count exceeds `4*map_size` (without reaching goal)

**Key Methods**:

```python
env._reset_impl(options)           # Build new map + sample start/goal
env._step_impl(action)             # Execute action, return (obs, r, term, trunc, info)
env._build_observation()           # Construct 7D observation vector
env.export_planner_inputs()        # Return (heightmap, no_fly, start_xy, goal_xy)
```

**Integration Points**:
- Called by `base.reset()` and `base.step()` (template pattern)
- Uses `self._rng` for random generation (seeding from base)
- Logs events via `self.log_event()` for collisions/violations

---

### 2.2 Scenario Configuration System

#### **File: `src/uavbench/scenarios/schema.py`** (100 lines)

**Role**: Pydantic-based configuration schema for scenarios

**Enums**:
```python
Domain: URBAN (extensible for MOUNTAIN, MARITIME, etc.)
Difficulty: EASY, MEDIUM, HARD
Wind: NONE, LOW, MEDIUM, HIGH
Traffic: NONE, LOW, MEDIUM, HIGH
```

**ScenarioConfig Dataclass** (frozen=True, slots=True for immutability):

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `name` | str | (required) | Scenario ID |
| `domain` | Domain | (required) | Environment type |
| `difficulty` | Difficulty | (required) | Obstacle/challenge level |
| `wind` | Wind | NONE | Environmental disturbance |
| `traffic` | Traffic | NONE | Dynamic obstacles |
| `map_size` | int | 25 | Grid dimensions (25×25) |
| `max_altitude` | int | 3 | Discrete altitude levels (0..3) |
| `building_density` | float | 0.30 | % cells with buildings (0..1) |
| `building_level` | int | 2 | Building height in levels |
| `start_altitude` | int | 1 | Agent spawn altitude |
| `safe_altitude` | int | 3 | Goal altitude (default cruise height) |
| `min_start_goal_l1` | int | 10 | Minimum Manhattan distance |
| `extra_density_medium` | float | 0.10 | +% buildings in MEDIUM |
| `extra_density_hard` | float | 0.20 | +% buildings in HARD |
| `no_fly_radius` | int | 3 | Central circle radius (HARD mode) |
| `downtown_window` | int | 7 | Odd value for spawn region (odd ≥3) |
| `spawn_clearance` | int | 1 | Buffer zone around start/goal |
| `debug` | bool | False | Enable debug output |
| `extra` | dict | None | Extensibility dict |

**Validation**: `config.validate()` raises `ValueError` for:
- map_size < 5
- max_altitude < 1
- building_density not in [0,1]
- building_level not in [0, max_altitude]
- Altitude parameters out of range
- Invalid window/clearance dimensions

---

#### **File: `src/uavbench/scenarios/loader.py`** (48 lines)

**Role**: YAML→Pydantic conversion for scenario files

**Function**: `load_scenario(path: Path) → ScenarioConfig`

**Process**:
1. Read YAML file (e.g., `urban_easy.yaml`)
2. Extract required fields: `name`, `domain`, `difficulty`
3. Extract optional fields with defaults
4. Map string values to Enum types
5. Construct frozen `ScenarioConfig` object
6. Call `.validate()` for early error detection
7. Return config object

**Example YAML Structure**:
```yaml
name: urban_easy
domain: urban
difficulty: easy
wind: none
traffic: none
map_size: 50
building_density: 0.30
building_level: 2
safe_altitude: 3
min_start_goal_l1: 10
```

**Integration**: Called by CLI and tests to load `.yaml` files from `src/uavbench/scenarios/configs/`

---

### 2.3 Planner Layer

#### **File: `src/uavbench/planners/astar.py`** (163 lines)

**Role**: A* pathfinding algorithm on 2D grid (x,y) coordinates

**Algorithm**: Classic A* with:
- Manhattan/Octile heuristic
- 4-connectivity (no diagonals) by default
- Building collision detection (heightmap > 0)
- No-fly zone enforcement

**Class: `AStarPlanner`**

**Constructor**:
```python
AStarPlanner(heightmap, no_fly, config=None)
```
- `heightmap`: float32 [H,W] array (0.0=free, >0=building)
- `no_fly`: bool [H,W] array (True=blocked)
- `config`: Optional `AStarConfig` (allow_diagonal, block_buildings, max_expansions)

**Public Method**:
```python
path: List[GridPos] = planner.plan(start=(x,y), goal=(x,y))
```
- Returns list of (x,y) tuples from start to goal (inclusive)
- Empty list [] if no path found

**Internal Implementation**:

1. **Collision Detection** (`_is_blocked()`):
   - Blocks if `no_fly[y,x] == True`
   - Blocks if `heightmap[y,x] > 0` (building) when `block_buildings=True`

2. **Heuristic** (`_heuristic()`):
   - 4-connectivity: Manhattan distance `|dx| + |dy|`
   - 8-connectivity: Octile distance (when diagonal enabled)

3. **Movement** (`_neighbors()`):
   - 4-connectivity: up, down, left, right (cost 1 each)
   - 8-connectivity: +diagonals (cost √2 each)

4. **Expansion**:
   - Standard A* with open/closed sets
   - Tie-breaking for determinism
   - Max expansion limit (200,000) to prevent infinite loops

5. **Path Reconstruction**:
   - Backtrack through came_from dict
   - Reverse to get start→goal order

**Type Definitions**:
```python
GridPos = Tuple[int, int]  # (x, y)

@dataclass(frozen=True)
class AStarConfig:
    allow_diagonal: bool = False
    block_buildings: bool = True
    max_expansions: int = 200_000
```

**Design Pattern**: Strategy (configurable via AStarConfig)

**Validation**:
- Checks start/goal in bounds
- Checks heightmap/no_fly same shape
- Checks 2D array dimensions

---

#### **File: `src/uavbench/planners/__init__.py`** (4 lines)

**Role**: Planner registry

```python
PLANNERS = {
    "astar": AStarPlanner,
}
```

**Pattern**: Allows CLI to dynamically instantiate planners: `planner_class = PLANNERS[planner_id]`

---

### 2.4 CLI & Orchestration

#### **File: `src/uavbench/cli/benchmark.py`** (304 lines)

**Role**: Command-line interface for running benchmarks and generating videos

**Entry Point**: `python -m uavbench.cli.benchmark [args]`

**CLI Arguments**:

| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--scenarios` | str | urban_easy | Comma-separated scenario IDs |
| `--planners` | str | astar | Comma-separated planner IDs |
| `--metrics` | str | path_length,success_rate,constraint_violations | Metrics to compute |
| `--trials` | int | 1 | Trials per (scenario, planner) |
| `--seed-base` | int | 0 | Base random seed |
| `--play` | str (best\|worst) | "" | Show animation in window |
| `--fps` | int | 8 | Playback frames per second |
| `--save-videos` | str (best\|worst\|both) | "" | Save to videos/ directory |
| `--fail-fast` | bool | False | Stop on first error (V&V mode) |

**Main Functions**:

1. **`run_planner_once(scenario_id, planner_id, seed)`** → `dict`:
   - Load scenario config
   - Create UrbanEnv instance
   - Reset with seed
   - Export planner inputs (heightmap, no_fly, start, goal)
   - Instantiate planner
   - Call `planner.plan(start, goal)`
   - Validate path (collision, bounds, start/goal endpoints)
   - Return results dict with metrics
   
2. **`aggregate(results, metric_ids)`** → `dict`:
   - Compute success_rate: % of trials with valid path
   - Compute avg_path_length: mean over successful trials
   - Compute avg_constraint_violations: mean violations
   - Return float-valued metric dict
   
3. **`main()`**:
   - Parse CLI arguments
   - Loop over scenario/planner pairs
   - For each pair, run trials with different seeds
   - Collect results
   - Compute aggregated metrics
   - Print results table
   - If `--play` or `--save-videos`: trigger visualization

**Seeding Strategy**:
```python
seed = base_seed + hash(scenario_id) + hash(planner_id) + trial_index
```
Ensures reproducibility while varying seeds across trials

**Video/Animation Handling**:
```python
if args.save_videos:
    best_path = min(successful_trials, key=lambda r: r["path_length"])
    worst_path = max(successful_trials, key=lambda r: r["path_length"])
    
    save_path_video(heightmap, no_fly, start, goal, path, output_path)
```

**Integration Points**:
- Imports `UrbanEnv` from `envs.urban`
- Imports `load_scenario` from `scenarios.loader`
- Imports `PLANNERS` registry from `planners`
- Lazy-imports `player` module only if visualization needed

---

### 2.5 Visualization & Animation

#### **File: `src/uavbench/viz/player.py`** (207 lines)

**Role**: Matplotlib-based visualization for path animations

**Two Main Functions**:

1. **`play_path_window(heightmap, no_fly, start, goal, path, title="", fps=8)`**:
   - Opens interactive window with animated path visualization
   - **Layers**:
     - Background: Building footprint (grayscale)
     - Overlay: No-fly zones (red with alpha)
     - Markers: Start (green), Goal (gold star)
     - Animated line: Path trace (blue line with markers)
   
   - **Rendering**:
     - Uses matplotlib TkAgg backend (cross-platform)
     - Frame-by-frame update with `fig.canvas.draw_idle()`
     - Sleeps between frames based on FPS
     - Keeps window open until user closes
   
   - **Figure Size**: 8×8 inches (reasonable for 25-50 px grids)

2. **`save_path_video(heightmap, no_fly, start, goal, path, output_path, title="", fps=8, dpi=100)`**:
   - Save animated path to file (MP4 or GIF)
   - **Attempt 1 (MP4)**: Uses ffmpeg writer if available
   - **Fallback (GIF)**: Uses Pillow writer (no external dependency)
   - **Output**: Creates `videos/` directory if missing
   - **DPI**: Controls resolution (default 100, higher = finer detail)
   
   - **Animation Details**:
     - `FuncAnimation` with frame callbacks
     - Frame rate controlled by interval_ms = 1000/fps
     - Blit=True for efficiency
     - No looping (repeat=False)

**Matplotlib Customization**:
```python
# Grid rendering
ax.imshow(heightmap > 0, cmap="gray", alpha=0.8)  # Building silhouettes
ax.imshow(no_fly, cmap="Reds", alpha=0.5)          # No-fly overlay

# Markers
ax.scatter(*start, c="green", s=120, label="Start")      # Start point
ax.scatter(*goal, c="gold", s=150, marker="*", label="Goal")  # Goal

# Path animation
(line,) = ax.plot([], [], linewidth=2, color="blue")     # Animated line

# Axes
ax.set_xlim(-0.5, W - 0.5)                                # X bounds
ax.set_ylim(H - 0.5, -0.5)                                # Y inverted (0 at top)
ax.set_aspect("equal")                                    # Square pixels
```

**Error Handling**:
- Catches ffmpeg failures gracefully
- Falls back to GIF if MP4 fails
- Handles window close events
- Keyboard interrupt support

---

### 2.6 Testing

#### **File: `tests/test_urban_env_basic.py`** (12 lines)

**Role**: Basic smoke tests for environment

**Tests**:
1. `test_urban_env_reset_step()`:
   - Load urban_easy scenario
   - Reset env
   - Check observation in space
   - Step 5 times with random actions
   - Verify observations remain valid

2. `test_urban_env_logs_trajectory()`:
   - Reset env
   - Step 3 times
   - Check trajectory has 3 entries
   - Verify step indices (1, 2, 3)

---

#### **File: `tests/test_scenario_basic.py`** (18 lines)

**Role**: Configuration loading tests

**Tests**:
1. `test_urban_hard_levels()`:
   - Load urban_hard.yaml
   - Verify wind=high, traffic=high

2. `test_urban_medium_no_fly()`:
   - Load urban_medium.yaml
   - Check no-fly zones list
   - Verify first zone coordinates

3. `test_load_urban_easy()`:
   - Load urban_easy.yaml
   - Verify domain=urban, difficulty=easy
   - Verify map_size=50

**Testing Pattern**: Arrange-Act-Assert (pytest style)

---

## 3. DATA FLOW & INTERCONNECTIONS

### 3.1 Scenario Load → Environment Creation

```
CLI (benchmark.py)
  ├─ parse --scenarios=urban_easy
  └─ call scenario_path("urban_easy")
     └─ return src/uavbench/scenarios/configs/urban_easy.yaml
        
loader.py: load_scenario(path)
  ├─ read YAML file
  ├─ construct ScenarioConfig Pydantic object
  └─ validate() → ScenarioConfig instance
     
CLI → instantiate UrbanEnv(config)
  ├─ base.py: __init__() → setup spaces, rng
  └─ ready for reset()
```

### 3.2 Environment Reset Flow

```
CLI: env.reset(seed=42)
  ↓
base.py: reset(seed)
  ├─ super().reset(seed=seed)     # Gymnasium seeding
  ├─ self._rng = np.random.default_rng(seed)  # CRITICAL: per-instance RNG
  ├─ reset bookkeeping (_step_count, _trajectory, _events)
  └─ call self._reset_impl(options)
     ↓
     urban.py: _reset_impl()
     ├─ validate map_size, altitude constraints
     ├─ generate building density mask using self._rng
     ├─ add difficulty-based extra buildings
     ├─ sample start/goal with min L1 distance
     ├─ set self._agent_pos, self._goal_pos
     └─ call _build_observation()
        ├─ extract [x, y, z, gx, gy, gz, h]
        └─ return obs as ndarray
     
     Return (obs, info) to CLI
```

### 3.3 Planning & Path Validation Flow

```
CLI: run_planner_once(scenario_id, planner_id, seed)
  ├─ load scenario → ScenarioConfig
  ├─ create env, reset(seed)
  ├─ env.export_planner_inputs()
  │  └─ return (heightmap, no_fly, start_xy, goal_xy)
  │
  ├─ planner = PLANNERS[planner_id](heightmap, no_fly)
  │  └─ instantiate AStarPlanner
  │
  ├─ path = planner.plan(start_xy, goal_xy)
  │  └─ A* search on 2D grid
  │
  └─ validate path:
     ├─ check path[0] == start_xy
     ├─ check path[-1] == goal_xy
     ├─ for each (x,y) in path:
     │  ├─ check 0 <= x,y < map_size
     │  ├─ check not no_fly[y,x]
     │  └─ check heightmap[y,x] == 0 (no building)
     └─ count violations
     
     Return: {success, path_length, violations, ...}
```

### 3.4 Benchmark Loop

```
CLI: main()
  ├─ parse arguments
  │  ├─ scenarios = ["urban_easy", "urban_medium"]
  │  ├─ planners = ["astar"]
  │  └─ trials = 3
  │
  └─ for scenario in scenarios:
     └─ for planner in planners:
        ├─ results = []
        ├─ for trial in 1..trials:
        │  ├─ seed = base_seed + hash(scenario) + hash(planner) + trial
        │  ├─ result = run_planner_once(scenario, planner, seed)
        │  └─ results.append(result)
        │
        ├─ metrics = aggregate(results, ["success_rate", "path_length"])
        ├─ print metrics table
        │
        └─ if --save-videos:
           ├─ best = min(successful, key=path_length)
           ├─ save_path_video(best)  → videos/urban_easy_astar_best.gif
           └─ if --save-videos=worst:
              └─ worst = max(successful, key=path_length)
                 └─ save_path_video(worst) → videos/urban_easy_astar_worst.gif
```

### 3.5 Visualization Flow

```
CLI: --save-videos=best

benchmark.py: save_path_video(heightmap, no_fly, start, goal, path, output_path)
  ↓
player.py: save_path_video(...)
  ├─ create output_path parent directories
  ├─ setup matplotlib figure + axes
  ├─ render background layers:
  │  ├─ imshow(heightmap > 0) → building footprints
  │  └─ imshow(no_fly) → no-fly zones
  ├─ add markers:
  │  ├─ scatter(start, green)
  │  └─ scatter(goal, gold)
  ├─ create FuncAnimation with path trace
  ├─ try anim.save(MP4) with ffmpeg:
  │  ├─ success → done
  │  └─ failure → fallback
  └─ anim.save(GIF) with pillow
     └─ output_path.gif created

Video file saved to: videos/urban_easy_astar_best_seedXXXXX.gif
```

---

## 4. KEY DESIGN PATTERNS

### 4.1 Template Method Pattern
**Location**: base.py (UAVBenchEnv)
- Base class defines `reset()` and `step()` algorithms
- Calls abstract `_reset_impl()` and `_step_impl()`
- Subclass (UrbanEnv) implements domain-specific logic
- Ensures consistent bookkeeping, seeding, event logging

### 4.2 Strategy Pattern
**Location**: planners/__init__.py, benchmark.py
- PLANNERS registry maps IDs to classes
- CLI dynamically instantiates: `planner = PLANNERS[id](...)`
- Easy to add new planners without changing CLI code

### 4.3 Factory Pattern
**Location**: loader.py
- `load_scenario(path)` factory creates ScenarioConfig from YAML
- Encapsulates YAML parsing, validation, enum mapping

### 4.4 Lazy Initialization
**Location**: benchmark.py (visualization)
- Matplotlib imported only if `--play` or `--save-videos` used
- Reduces startup time for non-visualization runs
- Graceful error if viz dependencies missing

---

## 5. SEEDING & REPRODUCIBILITY

**Critical Design: Per-Instance RNG**

```python
# BAD (global state):
np.random.seed(42)
mask = np.random.rand() < density  # NOT reproducible across instances

# GOOD (per-instance):
self._rng = np.random.default_rng(seed)
mask = self._rng.random() < density  # Reproducible, isolated
```

**Flow**:
1. CLI calls `env.reset(seed=42)`
2. base.py re-seeds: `self._rng = np.random.default_rng(42)`
3. urban.py uses `self._rng.random()` for all randomness
4. Every reset with seed=42 produces identical map/start/goal
5. Ensures benchmarks are reproducible across runs

---

## 6. CONSTRAINT SYSTEM

### 6.1 Observation Space Bounds
```python
# UrbanEnv.__init__()
observation_space = spaces.Box(
    low  = [0, 0, 0, 0, 0, 0, 0],
    high = [map_size-1, map_size-1, max_alt, map_size-1, map_size-1, max_alt, max_height],
    dtype = np.float32
)
# Ensures all observation values always in bounds
```

### 6.2 Collision Detection
```python
# urban.py: _step_impl()
terrain_h = self._heightmap[ny, nx]
attempted_building = (nz <= terrain_h)      # Altitude collision
attempted_no_fly = self._no_fly_mask[ny, nx]  # Zone violation
```

### 6.3 Path Validation
```python
# benchmark.py: run_planner_once()
for (x, y) in path:
    if not (0 <= x < W and 0 <= y < H): violations += 1
    if no_fly[y, x]: violations += 1
    if heightmap[y, x] > 0: violations += 1
```

---

## 7. CONFIGURATION SYSTEM DETAILS

### 7.1 Scenario YAML Example (urban_easy.yaml)

```yaml
name: urban_easy
domain: urban
difficulty: easy
wind: none
traffic: none

map_size: 50
max_altitude: 3

building_density: 0.30
building_level: 2

start_altitude: 1
safe_altitude: 3
min_start_goal_l1: 10

extra_density_medium: 0.10
extra_density_hard: 0.20
no_fly_radius: 3

downtown_window: 7
spawn_clearance: 1

debug: false
```

### 7.2 Configuration Validation Chain
```
YAML file
  ↓ (loader.py)
  Dict (parsed by PyYAML)
  ↓
  ScenarioConfig (Pydantic)
  ↓
  validate()  # Checks constraints
  ↓
  UrbanEnv.__init__()  # Uses config
  ↓
  reset()     # Applies config to environment
```

---

## 8. TRAJECTORY & METRICS

### 8.1 Trajectory Structure
```python
trajectory = [
    {
        "step": 1,
        "action": 3,  # right
        "obs": ndarray([x, y, z, gx, gy, gz, h]),
        "reward": -1.2,
        "terminated": False,
        "truncated": False,
        "info": {...}
    },
    {
        "step": 2,
        "action": 3,
        "obs": ...,
        ...
    },
    ...
]
```

### 8.2 Metrics Computation
```
Input: List of trial results
  ├─ success_rate = % trials with valid path
  ├─ avg_path_length = mean steps over successful paths
  └─ avg_constraint_violations = mean violations across all paths

Output: Dict[metric_name → float]
```

---

## 9. FILE DEPENDENCIES GRAPH

```
┌─ benchmark.py (CLI)
│  ├─ imports → envs.urban (UrbanEnv)
│  │  └─ imports → envs.base (UAVBenchEnv)
│  │     └─ imports → gymnasium
│  │     └─ imports → numpy
│  │
│  ├─ imports → scenarios.loader (load_scenario)
│  │  └─ imports → scenarios.schema (ScenarioConfig, Domain, etc.)
│  │     └─ imports → pydantic
│  │     └─ imports → yaml
│  │
│  ├─ imports → planners/__init__ (PLANNERS registry)
│  │  └─ imports → planners.astar (AStarPlanner)
│  │
│  └─ lazy imports → viz.player (visualization)
│     └─ imports → matplotlib
│        └─ imports → numpy
│
└─ tests/
   ├─ test_urban_env_basic.py
   │  └─ imports → envs.urban, scenarios.loader
   │
   └─ test_scenario_basic.py
      └─ imports → scenarios.loader
```

---

## 10. EXECUTION WORKFLOW EXAMPLE

**Command**: 
```bash
python -m uavbench.cli.benchmark \
  --scenarios=urban_easy \
  --planners=astar \
  --trials=2 \
  --seed-base=100 \
  --save-videos=best
```

**Execution Steps**:

1. **CLI Entry**
   - `benchmark.py:main()` parses arguments
   - scenario_ids = ["urban_easy"]
   - planner_ids = ["astar"]
   - trials = 2

2. **Trial 1** (seed = 100 + hash("urban_easy") + hash("astar") + 0 = 53283)
   - Load `urban_easy.yaml` → ScenarioConfig
   - Create `UrbanEnv(config)`
   - `env.reset(seed=53283)` → random map generated deterministically
   - Export inputs: (heightmap, no_fly, start, goal)
   - Create `AStarPlanner(heightmap, no_fly)`
   - `path = planner.plan(start, goal)` → 18 steps
   - Validate path → success=True, violations=0
   - Record: {scenario, planner, seed, success, path_length, path, ...}

3. **Trial 2** (seed ≈ 53284)
   - Same process with different seed
   - Different map, but same algorithm
   - Record results

4. **Aggregation**
   - Success rate = 2/2 = 1.0
   - Avg path length = (18 + 20) / 2 = 19
   - Avg violations = 0

5. **Video Generation** (--save-videos=best)
   - best_result = result with min path_length (18 steps)
   - Call `save_path_video(heightmap, no_fly, start, goal, path, "videos/urban_easy_astar_best_seed53283.mp4")`
   - Try MP4 with ffmpeg → fails (not installed)
   - Fallback to GIF with Pillow → success
   - File: `videos/urban_easy_astar_best_seed53283.gif` (48 KB)

6. **Output**
   ```
   [UAVBench] scenarios=['urban_easy'], planners=['astar'], ...
   
   Scenario: urban_easy
   Planner : astar
   Trials  : 2
       success_rate: 1.000
    avg_path_length: 19.000
   avg_constraint_violations: 0.000
   
   ✓ Animation saved: videos/urban_easy_astar_best_seed53283.gif
   ```

---

## 11. EXTENSIBILITY & FUTURE WORK

### 11.1 Adding a New Environment
```python
# src/uavbench/envs/mountain.py
class MountainEnv(UAVBenchEnv):
    def _reset_impl(self, options):
        # Mountain-specific logic
        pass
    
    def _step_impl(self, action):
        # Mountain-specific physics
        pass
```

### 11.2 Adding a New Planner
```python
# src/uavbench/planners/rrtstar.py
class RRTStarPlanner:
    def __init__(self, heightmap, no_fly):
        self.heightmap = heightmap
        self.no_fly = no_fly
    
    def plan(self, start, goal):
        # RRT* algorithm
        pass

# src/uavbench/planners/__init__.py
PLANNERS["rrtstar"] = RRTStarPlanner
```

### 11.3 Adding Metrics
```python
# benchmark.py: aggregate()
if "energy" in metric_ids:
    energy = [sum(abs(a) for a in path) for path in paths]  # Simple proxy
    out["energy"] = np.mean(energy)
```

---

## 12. SUMMARY TABLE

| Component | File | Lines | Purpose | Key Exports |
|-----------|------|-------|---------|------------|
| **Base Env** | envs/base.py | 139 | Gymnasium API abstraction | UAVBenchEnv |
| **Urban Env** | envs/urban.py | 369 | 2.5D urban simulator | UrbanEnv |
| **Schema** | scenarios/schema.py | 100 | Config Pydantic models | ScenarioConfig, Domain, Difficulty |
| **Loader** | scenarios/loader.py | 48 | YAML→Config factory | load_scenario() |
| **A* Planner** | planners/astar.py | 163 | A* pathfinding | AStarPlanner, AStarConfig |
| **Planner Registry** | planners/__init__.py | 4 | Dynamic planner loader | PLANNERS dict |
| **CLI Benchmark** | cli/benchmark.py | 304 | Orchestration & metrics | main(), run_planner_once(), aggregate() |
| **Visualization** | viz/player.py | 207 | Matplotlib animation | play_path_window(), save_path_video() |
| **Tests** | tests/test_*.py | 30 | Pytest smoke tests | test_* functions |

---

## 13. CRITICAL PRINCIPLES

1. **Determinism**: Per-instance RNG (`self._rng`) ensures reproducibility
2. **Immutability**: Config frozen dataclass prevents accidental mutations
3. **V&V**: Early validation (guards) catch invalid configs/states
4. **Type Safety**: Pydantic + static typing enable IDE support
5. **Extensibility**: Registry pattern (PLANNERS) and abstract base (UAVBenchEnv)
6. **Logging**: Trajectory + events enable post-hoc analysis
7. **Modularity**: Clear separation (envs, scenarios, planners, viz)
8. **Graceful Degradation**: Viz falls back to GIF if MP4 unavailable

---

**End of Analysis**

Generated: Feb 2, 2026  
Project: UAVBench v0.0.1  
Scope: Complete architecture overview with interconnections
