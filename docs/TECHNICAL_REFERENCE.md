# UAVBench — Complete Technical Reference

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Episode Lifecycle](#2-episode-lifecycle)
3. [Environment (UrbanEnvV2)](#3-environment)
4. [Planners](#4-planners)
5. [Dynamics](#5-dynamics)
6. [Scenarios & Configs](#6-scenarios)
7. [Missions](#7-missions)
8. [Metrics](#8-metrics)
9. [Visualization](#9-visualization)
10. [Guardrail & Calibration](#10-guardrail--calibration)
11. [Sanity Check](#11-sanity-check)
12. [Contracts](#12-contracts)
13. [Paper Pipeline](#13-paper-pipeline)
14. [Debugging Guide](#14-debugging-guide)

---

## 1. System Overview

### Architecture
```
src/uavbench/
├── cli/benchmark.py              # CLI: uavbench run --scenarios X --planners Y --trials N
├── benchmark/
│   ├── runner.py                 # Episode orchestrator (owns step_idx)
│   ├── determinism.py            # SHA-256 episode hashing (DC-2)
│   └── sanity_check.py           # Post-run SC-1/SC-2/SC-4 analysis
├── envs/
│   ├── base.py                   # Enums: TerminationReason, RejectReason, TaskStatus
│   └── urban.py                  # UrbanEnvV2 (Gymnasium env)
├── blocking.py                   # ONE compute_blocking_mask() (MP-1)
├── scenarios/
│   ├── schema.py                 # ScenarioConfig frozen dataclass
│   ├── loader.py                 # YAML → ScenarioConfig
│   ├── registry.py               # SCENARIO_IDS list
│   ├── calibration.py            # Feasibility pre-check (CC-1..4)
│   └── configs/*.yaml            # 3 scenario configs
├── missions/
│   ├── schema.py                 # TaskSpec, MissionBriefing
│   └── engine.py                 # MissionEngine (task lifecycle)
├── dynamics/
│   ├── fire_ca.py                # Fire CA: 8-neighbor Moore, isotropic, no wind
│   ├── traffic.py                # Emergency vehicle traffic
│   ├── restriction_zones.py      # Dynamic NFZ (staggered activation)
│   └── interaction_engine.py     # Fire↔traffic coupling → traffic_closure_mask
├── planners/
│   ├── base.py                   # PlannerBase ABC + PlanResult
│   ├── astar.py                  # A* (4-connected, static)
│   ├── apf.py                    # APF (artificial potential field, reactive)
│   ├── periodic_replan.py        # Time-triggered replan
│   ├── aggressive_replan.py      # Event-driven replan
│   ├── dstar_lite.py             # Incremental (simplified)
│   └── __init__.py               # PLANNERS registry (5 planners)
├── guardrail/feasibility.py      # Multi-depth relaxation (GC-1..4)
├── metrics/
│   ├── schema.py                 # EpisodeMetrics dataclass
│   └── compute.py                # compute_episode_metrics()
└── visualization/
    ├── renderer.py               # Frame renderer (paper_min / ops_full)
    ├── overlays.py               # Path, fire, NFZ, traffic, agent, markers
    └── hud.py                    # HUD badges + text (VC-2, MC-3)
```

### 5 Planners (3 families)
| ID | Family | Replanning | Algorithm |
|----|--------|------------|-----------|
| `astar` | Search (static) | Never | A* 4-connected, Manhattan heuristic |
| `periodic_replan` | Search (adaptive) | Every N steps | A* on dynamic blocking mask |
| `aggressive_replan` | Search (adaptive) | On mask change | A* on dynamic blocking mask |
| `dstar_lite` | Search (adaptive) | On path blocked | A* internal (simplified incremental) |
| `apf` | Reactive | Every step | Artificial potential field with A* fallback |

### 3 Scenarios (OSM-based, all medium difficulty)
| Scenario ID | Mission Type | OSM Tile | Dynamics |
|-------------|-------------|----------|----------|
| `osm_penteli_pharma_delivery_medium` | pharma_delivery | Penteli, Attica | fire=2, traffic=8+2, nfz=1 |
| `osm_piraeus_urban_rescue_medium` | urban_rescue | Piraeus port | fire=2, traffic=8+2, nfz=1 |
| `osm_downtown_fire_surveillance_medium` | fire_surveillance | Athens center | fire=1, traffic=8+2, nfz=1 |

---

## 2. Episode Lifecycle

### Full Step Loop (runner.py)
```
run_episode(scenario_id, planner_id, seed, frame_callback=None):
  1. config = load_scenario(scenario_id)
  2. env = UrbanEnvV2(config)
  3. obs, info = env.reset(seed=seed)           # DC-1: creates ONE root RNG
  4. heightmap, no_fly, start, goal = env.export_planner_inputs()
  5. planner = PLANNERS[planner_id](heightmap, no_fly, config)
  6. planner.set_seed(seed)
  7. plan_result = planner.plan(start, goal)     # Initial plan
  8. path = plan_result.path; path_idx = 0

  LOOP (until terminated/truncated):
    a. action = _path_to_action(agent_xy, path, path_idx)
    b. obs, reward, terminated, truncated, info = env.step(action)
    c. trajectory.append(env.agent_xy)
    d. advance path_idx if reached next waypoint
    e. dyn_state = env.get_dynamic_state()       # Fire/traffic/NFZ state
    f. planner.update(dyn_state)                 # Planner receives observation
    g. should, reason = planner.should_replan(...)
    h. if should: plan_result = planner.plan(agent_xy, goal)
       → update path, path_idx=0, replan_count++
    i. frame_callback(heightmap, state, dyn_state, config)  # Optional viz

  9. metrics = compute_episode_metrics(...)
  10. return EpisodeResult(events, trajectory, metrics, frame_hashes)
```

### Critical: FD-3 Fire Timing (within env.step())
```
step_idx += 1
├── 1. dyn_state = get_dynamic_state()          # CURRENT fire (frozen)
├── 2. blocked = compute_blocking_mask(...)      # Mask from CURRENT state
├── 3. Validate action → accept/reject move     # Against CURRENT mask
├── 4. Execute move (if valid)
├── 5. _step_dynamics()                          # Fire advances AFTER move
│      ├── fire.step()
│      ├── traffic.step(fire_mask)
│      ├── nfz.step()
│      └── interaction.update()
├── 6. mission.step(agent_xy, action, step_idx)
├── 7. Check goal/timeout/collision
└── 8. Return (obs, reward, terminated, truncated, info)
```

### RNG Tree (DC-1)
```
root_rng = np.random.default_rng(seed)
  ├── env_rng      → heightmap, roads, start/goal placement
  ├── fire_rng     → FireSpreadModel (ignition, spread)
  ├── traffic_rng  → TrafficModel (positions, targets)
  ├── nfz_rng      → RestrictionZoneModel (zone centers, radii)
  └── reserved_rng → (reserved for future use)
```

---

## 3. Environment

### UrbanEnvV2 (`envs/urban.py`)

**Observation Space:** `[agent_x, agent_y, goal_x, goal_y, terrain_height]` (5D float32)

**Action Space:** Discrete(5) — UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4

**Reward:**
- Per step: -1.0
- Progress shaping: +0.2 × (dist_before - dist_after)
- Goal reached: +50.0
- Collision terminal: -25.0

**Termination Reasons:**
- `SUCCESS` — reached goal
- `COLLISION_BUILDING` — hit building (if terminate_on_collision=True)
- `COLLISION_NFZ` — hit NFZ
- `TIMEOUT` — step_idx >= max_steps
- `INFEASIBLE` — no path exists
- `IN_PROGRESS` — still running

**Rejection Reasons (priority order for _classify_block):**
1. BUILDING (heightmap > 0)
2. NO_FLY (static NFZ)
3. FIRE (burning cell)
4. FIRE_BUFFER (safety buffer around fire)
5. SMOKE (smoke >= 0.5)
6. TRAFFIC_CLOSURE (fire-road interaction)
7. TRAFFIC_BUFFER (vehicle occupancy)
8. DYNAMIC_NFZ (mission zone)
9. OUT_OF_BOUNDS (off-map)

**Key Methods:**
- `reset(seed)` → obs, info
- `step(action)` → obs, reward, terminated, truncated, info
- `export_planner_inputs()` → heightmap, no_fly, start_xy, goal_xy
- `get_dynamic_state()` → dict with fire_mask, smoke_mask, traffic_*, dynamic_nfz_mask

**Info Dict (returned by step):**
```python
{
    "agent_xy", "goal_xy", "step_idx",
    "termination_reason", "objective_completed",
    "objective_poi", "objective_reason", "objective_label",
    "mission_domain", "distance_to_task", "task_progress",
    "deliverable_name", "service_time_s",
    "origin_name", "destination_name", "priority",
}
```

### Reference Corridor

The environment computes a reference corridor at `reset()` using **A*** (same algorithm as planners). This ensures fire corridor closures, vehicle roadblock placement, and fire ignition intersect the actual paths planners take. BFS serves as fallback if A* fails.

### Blocking Mask (`blocking.py`)
```python
compute_blocking_mask(heightmap, no_fly, config, dynamic_state=None) → bool[H,W]

Layers (OR-merged):
  1. heightmap > 0 (buildings)
  2. no_fly (static)
  3. traffic_closure_mask
  4. fire_mask + fire_buffer (if fire_blocks_movement)
  5. smoke >= 0.5 (if fire_blocks_movement)
  6. traffic_occupancy_mask (if traffic_blocks_movement)
  7. dynamic_nfz_mask
```

Fire buffer uses `scipy.ndimage.binary_dilation` with 4-connected structure, `iterations=fire_buffer_radius`.

---

## 4. Planners

### Base Interface (`planners/base.py`)
```python
class PlannerBase(ABC):
    def __init__(heightmap, no_fly, config)
    def plan(start, goal, cost_map=None) → PlanResult     # ABSTRACT
    def set_seed(seed)                                      # Optional (no-op)
    def update(dyn_state: dict)                             # Optional (no-op)
    def should_replan(pos, path, dyn_state, step) → (bool, str)  # Optional (False, "")

@dataclass
class PlanResult:
    path: list[tuple[int, int]]    # (x,y) waypoints, start+goal inclusive
    success: bool
    compute_time_ms: float
    expansions: int = 0
    replans: int = 0
    reason: str = ""
```

### A* (`planners/astar.py`)
- 4-connected grid, Manhattan heuristic
- max_expansions = 200,000
- Uniform cost (all moves cost 1)
- Never replans. Deterministic. Optimal.

### APF (`planners/apf.py`)
- Artificial potential field: attractive (quadratic Euclidean) + repulsive (distance transform)
- Quadratic attractive: `U_att = 0.5 * k_att * ||pos - goal||²` (Khatib 1986)
- Repulsive: `scipy.distance_transform_cdt` from blocking mask
- Greedy descent on combined field, random perturbation on plateau
- Fallback: full A* when stuck in local minimum
- Reactive planner: effectively replans every step

### Periodic Replan (`planners/periodic_replan.py`)
- Wraps A* internally
- Replans every `replan_every_steps` (default 6)
- Cooldown: 3 steps minimum between replans
- Builds dynamic heightmap: obstacles → height=999.0
- **RS-1 storm prevention:** SHA-256 mask hash + position check → skip naive replans

### Aggressive Replan (`planners/aggressive_replan.py`)
- Wraps A* internally
- Replans when blocking mask changes (SHA-256 hash comparison)
- Same cooldown and RS-1 storm prevention
- More responsive than periodic (event-driven vs time-triggered)
- Calibration: first call records baseline mask hash (no replan)

### D* Lite (`planners/dstar_lite.py`)
- **Simplified implementation** (uses A* internally, not true incremental D* Lite)
- Only replans if current path is blocked by new obstacles
- First call: calibration (baseline mask hash)
- If mask changed but path clear → skip replan
- If mask changed AND path blocked → replan
- Expected to underperform full-replan planners in fire scenarios (mass cell changes),
  but uses far fewer replans (incremental efficiency). See SC-4.

---

## 5. Dynamics

### Fire CA (`dynamics/fire_ca.py`)

**Cell States (FD-1):** UNBURNED=0, BURNING=1, BURNED_OUT=2

**8-Neighbor Moore (FD-2, isotropic, NO wind):**
```python
_NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
```

**Spread Probabilities (by landuse):**
| Landuse | Code | p_spread |
|---------|------|----------|
| Empty | 0 | 0.02 |
| Forest | 1 | 0.15 |
| Urban | 2 | 0.06 |
| Industrial | 3 | 0.03 |
| Water | 4 | 0.00 |

**Default landuse:** Urban (code 2, p_spread=0.06) for synthetic grids.

- Roads: 50% firebreak (prob *= 0.5)
- Burnout: after 100-200 steps per cell (random)
- Smoke: diffused from fire via 3×3 box blur (2 passes), persistence=0.7

**Ignition:** Random placement on burnable cells. Fire acts as a distributed environmental hazard that adaptive planners must detect and route around.

**Fire Corridor Guarantee (FC-1):**
- `guarantee_targets`: corridor cells that must be BURNING by `guarantee_step` (= event_t1)
- Approach ignitions placed 8-15 Manhattan distance from each target
- Safety net: force-ignites any UNBURNED targets at `guarantee_step`
- Extended burnout: guarantee targets have `_burnout_time = 999999.0`, so they
  stay BURNING for the entire episode (no re-ignition, no burnout gap — see BUG-8)

**Properties:** fire_mask, burned_mask, smoke_mask, total_affected

### Traffic (`dynamics/traffic.py`)
- Emergency vehicles on road network
- Greedy movement toward random targets (avoid fire)
- Buffer radius: 5 cells (Manhattan distance) for blocking
- Target refresh: when within 3 cells of target
- Fire avoidance events tracked

### Dynamic NFZ (`dynamics/restriction_zones.py`)
- Staggered activation between event_t1 and midpoint(t1, t2)
- Manhattan-distance circular zones, radius ∈ [8, 20]
- Max coverage cap: 30% of map
- Supports guardrail relaxation (shrink_px)

### Physical Corridor Interdictions (FC-1)
The abstract `ForcedBlockManager` has been replaced by physical interdiction
mechanisms using existing dynamics layers:
- **Fire corridor closures** (all scenarios): Fire ignition is guaranteed on
  corridor cells via `fire_ca.py`, creating a physical barrier that blocks the
  reference path. Extended burnout (999999 steps) ensures fire never burns out.
  Adaptive planners detect fire and reroute; static planners fail.
- **Vehicle roadblocks** (piraeus, additional): Traffic vehicles are positioned on
  corridor cells via `traffic.py`, blocking the path with physical vehicle occupancy.
- Fairness: same corridor for all planners (computed at reset, seed-dependent)
- Guardrail D1 clears roadblock vehicles instead of abstract forced blocks

### Interaction Engine (`dynamics/interaction_engine.py`)
- Couples fire and traffic → traffic_closure_mask
- Dilates fire mask, intersects with roads
- coupling_strength=1.0 → dilation=2 cells

---

## 6. Scenarios

### ScenarioConfig Fields (schema.py)

**Core:**
| Field | Type | Default |
|-------|------|---------|
| name | str | required |
| mission_type | MissionType | required |
| difficulty | Difficulty | required |
| domain | Domain | URBAN |
| regime | Regime | NATURALISTIC |
| paper_track | "static"/"dynamic" | "static" |

**Map:**
| Field | Type | Default |
|-------|------|---------|
| map_size | int | 500 |
| map_source | "synthetic"/"osm" | "synthetic" |
| building_density | float | 0.18 |
| fixed_start_xy | tuple\|None | None |
| fixed_goal_xy | tuple\|None | None |
| min_start_goal_l1 | int | 100 |

**Dynamics:**
| Field | Type | Default |
|-------|------|---------|
| enable_fire | bool | False |
| enable_traffic | bool | False |
| enable_dynamic_nfz | bool | False |
| fire_blocks_movement | bool | False |
| traffic_blocks_movement | bool | False |

**Episode:**
| Field | Type | Default |
|-------|------|---------|
| max_episode_steps | int\|None | None → 4×map_size |
| terminate_on_collision | bool | True |
| event_t1 | int\|None | None |
| event_t2 | int\|None | None |

**Obstacles:**
| Field | Type | Default |
|-------|------|---------|
| fire_ignition_points | int | 0 |
| fire_buffer_radius | int | 3 |
| num_emergency_vehicles | int | 0 |
| num_nfz_zones | int | 0 |

**Planning:**
| Field | Type | Default |
|-------|------|---------|
| replan_every_steps | int | 6 |
| max_replans_per_episode | int | 2000 |
| plan_budget_static_ms | float | 500.0 |
| plan_budget_dynamic_ms | float | 200.0 |

### All 3 Scenario Configs (OSM-based, medium difficulty)

| Scenario | Map | Density | Fire | Buffer | Traffic | NFZ | Interdiction | t1 | t2 |
|----------|-----|---------|------|--------|---------|-----|-------------|----|----|
| osm_penteli_pharma_delivery_medium | 500 | 0.18 | 2 | 1 | 8+2 | 1 | fire corridor | 40 | 120 |
| osm_piraeus_urban_rescue_medium | 500 | 0.29 | 2 | 1 | 8+2 | 1 | fire corridor + roadblock | 40 | 120 |
| osm_downtown_fire_surveillance_medium | 500 | 0.50 | 1 | 1 | 8+2 | 1 | fire corridor | 40 | 120 |

**All scenarios:** Dynamic track, OSM-based maps, 500x500, fire_buffer_radius=1, event window t1=40 t2=120. Traffic=8 emergency + 2 corridor-aware vehicles.

---

## 7. Missions

### Mission Types

| Type | Service Time | Task Placement | Deliverable | Priority |
|------|-------------|----------------|-------------|----------|
| pharma_delivery | 0 (fly-through) | At goal | medical_supplies | critical |
| urban_rescue | 2 steps (STAY) | At midpoint | rescue_assessment | critical |
| fire_surveillance | 3 steps (STAY) | At midpoint | perimeter_report | high |

### Mission Metadata

| Type | Origin | Destination | Objective Label |
|------|--------|-------------|----------------|
| pharma_delivery | Hospital Depot Alpha | Fire-Isolated Settlement | Emergency Medical Supply Delivery |
| urban_rescue | Emergency Operations Center | Stranded Area | Urban Search & Rescue Assessment |
| fire_surveillance | Fire Command Post | Active Fire Perimeter | Aerial Fire Perimeter Survey |

### Mission Engine (`missions/engine.py`)

**Completion (MC-2):** Agent must reach objective POI + stay for service_time consecutive steps.
- pharma_delivery: arrive at goal = complete (service_time=0)
- urban_rescue: arrive at midpoint, STAY 2 steps
- fire_surveillance: arrive at midpoint, STAY 3 steps

**Properties:** objective_poi, objective_reason, objective_label, deliverable_name, service_time_s, task_progress, all_tasks_completed, distance_to_task(xy), origin_name, destination_name, priority

**Events:** mission_briefing (at reset), task_completed (when service_time met)

---

## 8. Metrics

### EpisodeMetrics (metrics/schema.py)
| Metric | Type | Description |
|--------|------|-------------|
| success | bool | Reached goal |
| termination_reason | str | success/collision/timeout/infeasible |
| objective_completed | bool | Mission objective met (MC-2) |
| path_length | int | Trajectory length |
| executed_steps_len | int | Steps taken |
| planned_waypoints_len | int | Initial plan length |
| planning_time_ms | float | Computation time |
| replans | int | Number of replans |
| collision_count | int | Failed move attempts |
| nfz_violations | int | NFZ violations |
| distance_to_goal_final | float | Manhattan distance to goal at end |
| fire_exposure_steps | int | Steps where fire/buffer rejected move |

### Paper Table Mapping
- **Table 1:** success_rate per (scenario, planner) × 30 seeds
- **Table 2:** avg path_length, replans, planning_time_ms
- **Figure 1:** success_rate bar chart (planner × difficulty)
- **Figure 2:** fire spread visualization
- **Figure 3:** trajectory comparison (static vs adaptive)

---

## 9. Visualization

### Renderer (`visualization/renderer.py`)

**Two modes:**
- `paper_min`: 15 px/cell, minimal HUD (rows 1-2 only)
- `ops_full`: auto-scaled (~1200px max dimension), full HUD (rows 1-4)

**Z-order (bottom to top):**
| Z | Layer | Color |
|---|-------|-------|
| 1 | Basemap (landuse + roads + buildings) | Cream ground, asphalt grey roads, dark grey buildings |
| 3.5 | Smoke overlay | Grey (160,160,160) |
| 3.8 | Fire buffer zone | Light orange (255,180,100) semi-transparent |
| 4 | Fire overlay | Red-orange (255,80,20) |
| 5 | Dynamic NFZ | Purple (204,121,167) with diagonal hatching |
| 6 | Traffic closures/occupancy | Orange (230,159,0) |
| 9 | Trajectory trail | Blue (0,114,178) |
| 9.5 | Planned path | Cyan (86,180,233) |
| 9.6 | Start marker (green) / Goal marker (gold) | |
| 10 | Agent icon | Blue (0,114,178) circle + white border |
| 12 | HUD badges + text | |
| 13 | Color legend bar | |

### HUD Badges (hud.py)

**Plan Badge (VC-2):**
- `"NO PLAN"` — plan_len <= 1
- `"STALE PLAN (reason)"` — plan_age > 2 × replan_every
- `"PLAN: Nwp"` — active plan with N waypoints

**HUD Lines (ops_full):**
```
Row 1 (yellow):  MISSION: {objective_label} [{PRIORITY}]
Row 2 (light):   {origin_name} > {destination_name}  |  PLN: {planner}
Row 3 (white):   T: {step_idx}  |  REP: {replans}  |  {plan_badge}
Row 4 (blue):    DIST: {distance}  |  TASKS: {progress}  |  {deliverable}
```

### Mission Briefing Title Card
Shown before GIF animation (3 seconds default). Displays: MISSION BRIEFING, objective, FROM/TO, PLANNER [DIFFICULTY], CARGO, constraints.

---

## 10. Guardrail & Calibration

### Feasibility Guardrail (`guardrail/feasibility.py`)

**Multi-depth relaxation (GC-1):**

| Depth | Action | Effect |
|-------|--------|--------|
| D0 | None | Already reachable |
| D1 | Clear roadblock vehicles | Remove corridor vehicle roadblocks |
| D2 | Shrink NFZ | Erode zones by 2px |
| D3 | Remove traffic | Clear traffic occupancy + closure |

Uses `compute_blocking_mask()` + BFS reachability check at each depth.
Returns `GuardrailResult(feasible, depth, relaxations)`.

### Calibration (`scenarios/calibration.py`)

**CC-1: Feasibility pre-check:**
- Simulates dynamics forward (STAY action), BFS at each step
- Returns first_infeasible_step or "always_feasible"

**CC-2: Difficulty thresholds:**
| Difficulty | Min Feasibility Rate |
|-----------|---------------------|
| Easy | >= 80% |
| Medium | >= 50% |
| Hard | >= 15% |

### Medium Calibration Results (10 seeds, osm_penteli_pharma_delivery_medium)
| Planner | Success Rate | Avg Replans |
|---------|-------------|-------------|
| A* (static) | 0% | 0 |
| Aggressive Replan | 70% | ~94 |
| Periodic Replan | ~67% | ~36 |
| D* Lite | ~67% | ~4 |

A* blocked by corridor interdiction (fire/vehicle roadblock). Adaptive planners detect and route around.

---

## 11. Sanity Check

### Post-run analysis (`benchmark/sanity_check.py`)

| Check | Severity | Condition |
|-------|----------|-----------|
| SC-1 | ERROR | Static beats ALL adaptive in fire scenario |
| SC-2 | WARNING | Hard success > Medium success + 5% |
| SC-4 | WARNING | A* success > D*Lite success + 5% |

**Usage:**
```python
from uavbench.benchmark.sanity_check import run_sanity_check
report = run_sanity_check(results_list)
assert report.passed  # No ERROR-level violations
```

---

## 12. Contracts (32 total, 13 families)

| Family | IDs | Module | Tests |
|--------|-----|--------|-------|
| DC Determinism | DC-1, DC-2 | runner, env | contract_test_determinism.py |
| FC Fairness | FC-1, FC-2 | blocking, fire_ca, traffic | contract_test_fairness.py |
| EC Events | EC-1, EC-2 | env (reject/accept) | contract_test_decision_record.py |
| GC Guardrail | GC-1, GC-2 | feasibility.py | contract_test_guardrail.py |
| EV Events | EV-1 | runner (step_idx) | contract_test_event_semantics.py |
| VC Visual | VC-1, VC-2 | renderer, hud | contract_test_visual_truth.py |
| MC Mission | MC-1..MC-4 | engine, schema | contract_test_mission_story.py |
| PC Planner | PC-1, PC-2 | planners | integration_test_runner_e2e.py |
| FD Fire | FD-1..FD-5 | fire_ca, env | unit_test_fire_ca.py |
| CC Calibration | CC-1..CC-4 | calibration.py | contract_test_calibration.py |
| SC Sanity | SC-1..SC-4 | sanity_check.py | contract_test_sanity.py |
| MP Mask | MP-1 | blocking.py | contract_test_mask_parity.py |
| RS Replan Storm | RS-1 | planners | contract_test_replan_storm_regression.py |

---

## 13. Paper Pipeline

### Commands
```bash
# Run all 5 planners × 3 scenarios × 30 seeds
python scripts/run_paper_experiments.py --trials 30

# Analyze results → LaTeX tables + figures
python scripts/analyze_paper_results.py

# Generate demo GIFs (animated visualization)
python scripts/gen_demo_gifs.py [--easy] [--fps 10]

# Export determinism + viz evidence
python scripts/export_artifacts.py

# Single episode rendering for debugging
python scripts/render_episode.py osm_penteli_pharma_delivery_medium aggressive_replan 42
```

### Expected Output Files
```
outputs/
├── paper_results/all_episodes.csv   # Raw results (scenario, planner, seed, success, ...)
├── paper_tables/                     # LaTeX .tex files
├── paper_figures/                    # PNG 300dpi + PDF
├── demo_gifs/                        # Animated GIFs per scenario×planner
├── determinism_hashes.json           # DC-2 verification
├── viz_manifest.csv                  # Rendering manifest
├── viz_frame_checks.json             # Per-frame validation
├── repro_manifest.json               # Full reproducibility record
└── rebuild_audit.json                # Build verification
```

---

## 14. Debugging Guide

### Common Issues

**"No path found" (A* returns success=False):**
- Check building_density (>0.25 may create disconnected maps)
- Check fire/NFZ blocking too many cells
- Run calibration: `feasibility_pre_check(config, seed)`

**Replan storms (planner thrashes):**
- Check RS-1: mask hash should prevent identical replans
- Check cooldown (3-step minimum)
- Look for `naive_skip` in replan reasons

**Determinism failure (DC-2):**
- grep for `np.random.` outside reset() — violates DC-1
- Check `random.Random()` or `random.seed()` — forbidden
- Ensure no module-level RNG state

**Fire timing issues (FD-3):**
- Fire must advance AFTER move in env.step()
- Check `_step_dynamics()` is called after move validation
- Planner must see CURRENT (pre-advance) fire state

**Sanity check failures:**
- SC-1 (adaptive < static in fire): check planner replan logic, fire timing
- SC-2 (hard > medium): check difficulty knobs in YAML configs
- SC-4 (D*Lite < A*): check path blocking detection in dstar_lite.py

### Key Debug Commands
```python
# Check a specific episode
from uavbench.benchmark.runner import run_episode
result = run_episode("osm_penteli_pharma_delivery_medium", "aggressive_replan", seed=42)
print(f"Success: {result.metrics['success']}")
print(f"Steps: {result.metrics['executed_steps_len']}")
print(f"Replans: {result.metrics['replans']}")
print(f"Reason: {result.metrics['termination_reason']}")

# Check feasibility
from uavbench.scenarios.calibration import feasibility_pre_check
from uavbench.scenarios.loader import load_scenario
config = load_scenario("osm_penteli_pharma_delivery_medium")
result = feasibility_pre_check(config, seed=42)
print(f"Feasible: {result.feasible}, infeasible at step: {result.first_infeasible_step}")

# Blocking mask inspection
from uavbench.blocking import compute_blocking_mask
mask = compute_blocking_mask(heightmap, no_fly, config, dyn_state)
print(f"Blocked cells: {mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")
```

### Coordinate System
- **Paths:** (x, y) tuples — x=column, y=row
- **Heightmap indexing:** heightmap[y, x]
- **Actions:** UP=0 (y-1), DOWN=1 (y+1), LEFT=2 (x-1), RIGHT=3 (x+1), STAY=4
