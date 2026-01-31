# UAVBench Project - Complete File Structure & Documentation

## 📋 Project Overview

**UAVBench** is a reproducible UAV (Unmanned Aerial Vehicle) path planning benchmark framework. It provides:
- Realistic 2.5D urban environments with buildings and no-fly zones
- Path planning algorithms (A*, with extensibility for more)
- Comprehensive metrics evaluation
- Interactive visualization and video export
- CLI interface for running benchmarks

---

## 🗂️ Complete File Structure

```
uavbench/
├── pyproject.toml                          # Project configuration & dependencies
├── .github/
│   └── copilot-instructions.md            # AI coding agent instructions
├── src/uavbench/
│   ├── __init__.py                        # Package root
│   ├── envs/                              # Environment implementations
│   │   ├── base.py                        # Abstract base class for all environments
│   │   └── urban.py                       # Urban 2.5D environment (main domain)
│   ├── scenarios/                         # Scenario configuration system
│   │   ├── schema.py                      # Pydantic models for configurations
│   │   ├── loader.py                      # YAML→Config loader
│   │   └── configs/
│   │       ├── urban_easy.yaml            # Easy difficulty scenario
│   │       ├── urban_medium.yaml          # Medium difficulty scenario
│   │       └── urban_hard.yaml            # Hard difficulty scenario
│   ├── planners/                          # Path planning algorithms
│   │   ├── __init__.py                    # Planner registry
│   │   └── astar.py                       # A* grid planner implementation
│   ├── cli/                               # Command-line interface
│   │   └── benchmark.py                   # Main CLI entry point
│   └── viz/                               # Visualization & video export
│       └── player.py                      # Animation playback & GIF/MP4 export
├── tests/                                 # Unit tests
│   ├── test_scenario_basic.py             # Scenario loading tests
│   └── test_urban_env_basic.py            # Environment tests
├── videos/                                # Output directory for saved animations
│   ├── *.gif                              # Exported animations (auto-created)
│   └── *.mp4                              # MP4 exports (if ffmpeg installed)
└── Documentation Files (various .md files) # See below
```

---

## 📄 Core Source Files

### 1. **`pyproject.toml`** - Project Configuration
**Purpose**: Defines package metadata, dependencies, and build configuration

**Contains**:
- Project name, version, description
- Python version requirement (≥3.10)
- Core dependencies:
  - `gymnasium≥0.29` - RL environment framework
  - `numpy≥1.26` - Numerical computing
  - `pydantic≥2.5` - Data validation
- Optional dependencies:
  - `[viz]`: matplotlib for visualization
- Entry point: `uavbench` CLI command

**Key Lines**:
```toml
[project.scripts]
uavbench = "uavbench.cli.benchmark:main"
```

---

### 2. **`src/uavbench/__init__.py`** - Package Root
**Purpose**: Initializes package namespace

**Contains**:
- Package version string
- `__all__` export list

---

## 🏗️ Architecture Files

### 3. **`src/uavbench/envs/base.py`** - Abstract Base Environment
**Purpose**: Common RL environment logic for all domain implementations

**Key Classes**:
- `UAVBenchEnv` (abstract base class)

**Responsibilities**:
- Gymnasium API compliance (`reset()`, `step()`)
- Per-instance RNG (random number generator) for reproducibility
- Trajectory logging (every step recorded)
- Event logging (collisions, violations)
- Type validation and guards

**Key Methods**:
- `reset(seed)` - Initialize episode with deterministic seeding
- `step(action)` - Execute action, return (obs, reward, terminated, truncated, info)
- `log_event(type, **payload)` - Record domain-specific events
- Properties: `trajectory`, `events`, `step_count`

**Critical Pattern**: Uses `self._rng` (per-instance numpy Generator) for all randomness, never global `np.random`

---

### 4. **`src/uavbench/envs/urban.py`** - Urban 2.5D Environment
**Purpose**: Concrete implementation of 2.5D urban path planning environment

**Key Classes**:
- `UrbanEnv(UAVBenchEnv)` - Main environment for 2D grid navigation with discrete altitude

**Domain Details**:
- **State**: 7D observation `[x, y, z, goal_x, goal_y, goal_z, heightmap[x,y]]`
  - Position (x, y) in grid coordinates
  - Altitude z in discrete levels (0..max_altitude)
  - Goal position and current cell height
- **Action Space**: Discrete(6)
  - 0=up (y-1), 1=down (y+1), 2=left (x-1), 3=right (x+1), 4=ascend (z+1), 5=descend (z-1)
- **Collision Detection**:
  - Building collision: `z <= heightmap[x,y]`
  - No-fly zone: `grid[y,x] in no_fly_mask`
- **Reward Function**:
  - Step cost: -1.0 per action
  - Progress shaping: +0.2 per unit distance reduction
  - Safety penalties: -5 (building), -8 (no-fly)
  - Goal bonus: +50

**Key Methods**:
- `_reset_impl()` - Generate random map + place start/goal
  - Builds heightmap with configurable density
  - Applies difficulty-specific tweaks (EASY/MEDIUM/HARD)
  - Ensures minimum start-goal distance
- `_step_impl(action)` - Handle movement and collisions
- `_build_observation()` - Construct observation vector

**Difficulty Scaling**:
- **EASY**: Base building density (30%)
- **MEDIUM**: Extra buildings (+10%)
- **HARD**: Extra buildings (+20%) + central no-fly circle

---

### 5. **`src/uavbench/scenarios/schema.py`** - Configuration Schema
**Purpose**: Pydantic models for scenario configuration validation

**Key Enums**:
- `Domain`: URBAN, MOUNTAIN, COASTAL, INFRASTRUCTURE, ATHENSFIR
- `Difficulty`: EASY, MEDIUM, HARD
- `WindLevel`: NONE, LOW, MEDIUM, HIGH
- `TrafficLevel`: NONE, LOW, MEDIUM, HIGH

**Key Models**:
- `NoFlyZone` - Rectangle in grid coordinates (x_min, y_min, x_max, y_max)
- `ScenarioConfig` - Complete scenario specification
  - name, domain, difficulty, map_size
  - building_density, building_level, altitude settings
  - no-fly zones, environmental parameters

---

### 6. **`src/uavbench/scenarios/loader.py`** - Scenario Loader
**Purpose**: Load YAML scenario files into validated Pydantic models

**Key Function**:
- `load_scenario(path)` → `ScenarioConfig`
  - Reads YAML file
  - Validates with Pydantic
  - Returns config object

---

### 7. **`src/uavbench/scenarios/configs/*.yaml`** - Scenario Definitions

#### **`urban_easy.yaml`**
- Map size: 25×25
- Building density: 25%
- Building level: 2
- Wind: NONE, Traffic: NONE
- Sparse obstacles, clear weather

#### **`urban_medium.yaml`**
- Map size: 35×35
- Building density: 35%
- Building level: 2
- Wind: LOW, Traffic: LOW
- Moderate obstacles, some environmental stress

#### **`urban_hard.yaml`**
- Map size: 50×50
- Building density: 45%
- Building level: 3
- Wind: HIGH, Traffic: HIGH
- Dense obstacles, central no-fly zone, extreme conditions

---

### 8. **`src/uavbench/planners/__init__.py`** - Planner Registry
**Purpose**: Central registry of available planners

**Contains**:
```python
PLANNERS = {
    "astar": AStarPlanner,
}
```

**Extensibility**: Add new planners by:
1. Implementing planner class
2. Registering in `PLANNERS` dict

---

### 9. **`src/uavbench/planners/astar.py`** - A* Path Planner
**Purpose**: A* grid-based path planning algorithm

**Key Classes**:
- `AStarConfig` - Configuration dataclass
  - `allow_diagonal`: Enable diagonal movement
  - `block_buildings`: Treat buildings as obstacles
  - `max_expansions`: Safety limit to prevent pathological cases
- `AStarPlanner` - Implementation

**Algorithm**:
- Standard A* search on 2D grid
- Heuristic: Manhattan distance to goal
- Open set: priority queue of (f_score, tie, position)
- Closed set: visited cells
- Returns: `List[GridPos]` path from start to goal (or empty if no path)

**Obstacles**:
- No-fly zones (always blocked)
- Buildings (if `block_buildings=True`)

---

### 10. **`src/uavbench/cli/benchmark.py`** - Command-Line Interface
**Purpose**: Main entry point for running benchmarks

**Key Functions**:
- `run_planner_once(scenario_id, planner_id, seed)` - Run single trial
  - Loads scenario, creates environment
  - Calls planner, validates path
  - Returns detailed results dict
- `aggregate(results, metric_ids)` - Compute metrics across trials
  - Success rate
  - Average path length
  - Constraint violations
- `main()` - CLI argument parsing and benchmark execution

**Command-Line Options**:
- `--scenarios` - Comma-separated scenario IDs
- `--planners` - Comma-separated planner IDs
- `--trials` - Number of trials per scenario/planner
- `--seed-base` - Random seed for reproducibility
- `--play [best|worst]` - Open interactive visualization
- `--save-videos [best|worst|both]` - Export animations
- `--fps` - Playback speed (frames per second)
- `--metrics` - Which metrics to compute
- `--fail-fast` - Stop on first error (V&V mode)

**Output**:
- Prints metrics for each scenario/planner combination
- Saves videos to `videos/` directory if requested
- Opens animation windows if `--play` specified

---

### 11. **`src/uavbench/viz/player.py`** - Visualization & Video Export
**Purpose**: Interactive animation playback and video export

**Key Functions**:
- `play_path_window(heightmap, no_fly, start, goal, path, fps, title)` - Interactive playback
  - Opens matplotlib window with TkAgg backend
  - Animates path frame-by-frame (smooth movement)
  - Shows buildings (gray), no-fly zones (red)
  - Start (green dot), goal (gold star)
  - FPS-controlled playback speed

- `save_path_video(heightmap, no_fly, start, goal, path, output_path, fps, dpi)` - Export animation
  - Tries MP4 format (requires ffmpeg)
  - Falls back to GIF (Pillow only)
  - Auto-creates `videos/` directory
  - Descriptive filename: `{scenario}_{planner}_{best|worst}_seed{seed}.{ext}`

**Visualization Elements**:
- **Gray regions**: Buildings/obstacles
- **Red regions**: No-fly zones
- **Green dot**: Start position
- **Gold star**: Goal position
- **Blue line**: Path (animates from start to end)

---

## 🧪 Test Files

### 12. **`tests/test_scenario_basic.py`** - Scenario Configuration Tests
**Test Cases**:
- `test_load_urban_easy()` - Load and validate easy scenario
- `test_urban_medium_no_fly()` - Validate no-fly zone configuration
- `test_urban_hard_levels()` - Validate difficulty-specific settings

**Validates**:
- YAML loading works correctly
- Pydantic validation enforces constraints
- Difficulty levels have correct parameters

---

### 13. **`tests/test_urban_env_basic.py`** - Environment Tests
**Test Cases**:
- `test_urban_env_reset_step()` - Basic reset/step cycle
  - Verify observations stay in bounds
  - Run 5 random steps
  - Validate observation space compliance
- `test_urban_env_logs_trajectory()` - Trajectory recording
  - Run 3 steps
  - Verify trajectory has 3 entries
  - Check step indices

**Validates**:
- Environment works with Gymnasium API
- Trajectory logging functional
- Observations always valid

---

## 📚 Documentation Files

### 14. **`.github/copilot-instructions.md`** - AI Coding Guide
**Purpose**: Instructions for AI coding agents to work in this project

**Contains**:
- Architecture overview
- Critical patterns (RNG discipline, trajectory logging)
- Common pitfalls and solutions
- Code locations and exemplars
- Integration points for extensibility

---

### 15. **`CODE_REVIEW.md`** - Code Review Summary
**Purpose**: Document of code review and fixes performed

**Contains**:
- Errors found and fixed
- Indentation issues resolved
- Test results
- Recommendations for preventing future issues

---

### 16. **`MATPLOTLIB_RESOLUTION.md`** - Dependency Fix Documentation
**Purpose**: Record of matplotlib dependency resolution

**Contains**:
- Issue: matplotlib not installed
- Solution: Lazy import + optional dependencies
- How visualization is now optional but available

---

### 17. **`VISUALIZATION_GUIDE.md`** - Complete Visualization Guide
**Purpose**: Comprehensive guide for users on visualization features

**Contains**:
- Quick start commands
- Option reference
- Video output documentation
- Troubleshooting guide
- Advanced usage examples
- File organization

---

### 18. **`VIDEO_FEATURE_SUMMARY.md`** - Video Feature Overview
**Purpose**: Technical summary of video export implementation

**Contains**:
- Feature highlights
- File format handling
- Real-world usage examples
- File size information
- Next steps for users

---

### 19. **`ANIMATION_WORKING.md`** - Animation Feature Documentation
**Purpose**: Complete guide showing animation is now fully functional

**Contains**:
- What works (interactive + save modes)
- Quick start commands
- Command reference
- Visualization guide
- Troubleshooting
- Real-world examples

---

### 20. **`QUICK_VIDEO_REFERENCE.md`** - Quick Reference Card
**Purpose**: One-page quick reference for video features

**Contains**:
- One-liner commands
- Options summary table
- Output file information
- Installation tips
- Common tasks
- Troubleshooting matrix

---

## 🔄 Data Flow & Integration

### Typical Benchmark Run Flow

```
user runs: uavbench --scenarios urban_easy --planners astar --trials 2 --play best

↓

benchmark.py:main()
  ├── Parse arguments
  ├── For each scenario:
  │   └── scenario_path() → loads YAML
  │       loader.load_scenario() → ScenarioConfig (pydantic validated)
  │
  ├── For each planner:
  │   └── For each trial:
  │       └── run_planner_once()
  │           ├── Create UrbanEnv(config)
  │           ├── env.reset(seed) 
  │           │   └── _reset_impl() → generates map, places start/goal
  │           ├── Get heightmap + no_fly from env
  │           ├── Create AStarPlanner(heightmap, no_fly)
  │           ├── planner.plan(start, goal) → List[GridPos]
  │           └── Validate path → return results dict
  │
  ├── aggregate(results, metrics) → compute statistics
  ├── Print metrics
  │
  └── If --play best:
      └── play_path_window() → opens matplotlib window
          ├── Draw environment (buildings, no-fly)
          ├── Animate path frame-by-frame
          └── Wait for user to close window
```

### Video Export Flow

```
uavbench --save-videos best --fps 10

↓ (same as above until visualization)

├── successful_paths = filter(result["success"] == True)
├── chosen = min(successful_paths, key=path_length)
│
└── save_path_video()
    ├── Check if ffmpeg available
    ├── Create figure with environment + path
    ├── For each frame:
    │   ├── Draw path up to position i
    │   └── Append to animation
    ├── Save as MP4 (or GIF fallback)
    └── videos/urban_easy_astar_best_seed0.mp4 ✓
```

---

## 🔌 Extension Points

### Adding a New Planner

1. Create `src/uavbench/planners/myplanner.py`
2. Implement class inheriting from planner base pattern
3. Register in `src/uavbench/planners/__init__.py`:
   ```python
   from .myplanner import MyPlanner
   PLANNERS = {
       "astar": AStarPlanner,
       "myplanner": MyPlanner,
   }
   ```
4. Use with CLI: `uavbench --planners myplanner`

### Adding a New Scenario

1. Create `src/uavbench/scenarios/configs/my_scenario.yaml`
2. Define configuration matching `ScenarioConfig` schema
3. Use with CLI: `uavbench --scenarios my_scenario`

### Adding a New Domain (e.g., MountainEnv)

1. Create `src/uavbench/envs/mountain.py`
2. Subclass `UAVBenchEnv`, implement `_reset_impl()` and `_step_impl()`
3. Define in `scenarios/schema.py`: add to `Domain` enum
4. Add scenario YAML files
5. Add tests in `tests/test_mountain_*.py`

---

## 🎯 Quick Reference: What Each File Does

| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `pyproject.toml` | Project config | Build metadata |
| `envs/base.py` | RL base class | `UAVBenchEnv` |
| `envs/urban.py` | 2.5D urban env | `UrbanEnv` |
| `scenarios/schema.py` | Config validation | `ScenarioConfig` |
| `scenarios/loader.py` | YAML→Config | `load_scenario()` |
| `planners/__init__.py` | Planner registry | `PLANNERS` dict |
| `planners/astar.py` | A* algorithm | `AStarPlanner` |
| `cli/benchmark.py` | CLI interface | `main()` |
| `viz/player.py` | Visualization | `play_path_window()`, `save_path_video()` |
| `tests/test_*.py` | Unit tests | Various test functions |

---

## 🚀 Quick Start

```bash
# Install with visualization support
pip install -e ".[viz]"

# Run simple benchmark
uavbench --scenarios urban_easy --planners astar --trials 5

# Watch best path animation
uavbench --scenarios urban_easy --planners astar --trials 10 --play best --fps 8

# Save animations
uavbench --scenarios urban_easy --planners astar --trials 10 --save-videos both

# Full example with multiple scenarios
uavbench \
  --scenarios urban_easy,urban_medium,urban_hard \
  --planners astar \
  --trials 20 \
  --seed-base 42 \
  --play best \
  --save-videos worst \
  --fps 10
```

---

## 📊 File Statistics

- **Total Python files**: 11 core + 2 test = 13
- **Total documentation files**: 7 markdown files
- **Configuration files**: 4 (pyproject.toml + 3 YAML scenarios)
- **Lines of code**: ~2,500 (core logic)
- **Test coverage**: Basic environment + scenario tests

---

## ✅ Status

- ✅ Core environments working (UrbanEnv)
- ✅ A* planner implemented
- ✅ Scenario loading functional
- ✅ CLI working with all options
- ✅ Interactive visualization (animation pop-up)
- ✅ Video export (GIF + MP4 fallback)
- ✅ Tests passing
- ✅ Documentation complete

---

## 🔗 Dependencies

### Core
- `gymnasium >= 0.29` - RL environment framework
- `numpy >= 1.26` - Numerical computing
- `pydantic >= 2.5` - Data validation
- `PyYAML` - YAML parsing

### Optional [viz]
- `matplotlib >= 3.8.0` - Visualization

### Development [dev]
- `pytest >= 7.0` - Testing
- `mypy >= 1.0` - Type checking
- `sphinx >= 7.0` - Documentation

---

This is your complete project reference! All files work together to provide a reproducible UAV planning benchmark framework with interactive visualization.
