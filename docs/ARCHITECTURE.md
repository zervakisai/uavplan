# UAVBench v2 — Architecture Specification

## 1. Module Map

```
src/uavbench/
├── __init__.py                          # version, top-level exports
├── __main__.py                          # python -m uavbench
│
├── cli/
│   ├── __init__.py
│   └── benchmark.py                     # ONE CLI entry point (RU-2)
│
├── scenarios/
│   ├── __init__.py
│   ├── schema.py                        # ScenarioConfig frozen dataclass (SC-2)
│   ├── loader.py                        # load_scenario(path) → ScenarioConfig (SC-3)
│   ├── registry.py                      # SCENARIO_REGISTRY + filter functions (SC-4)
│   └── configs/                         # 9 YAML files (SC-1)
│       ├── gov_civil_protection_easy.yaml
│       ├── gov_civil_protection_medium.yaml
│       ├── gov_civil_protection_hard.yaml
│       ├── gov_maritime_domain_easy.yaml
│       ├── gov_maritime_domain_medium.yaml
│       ├── gov_maritime_domain_hard.yaml
│       ├── gov_critical_infrastructure_easy.yaml
│       ├── gov_critical_infrastructure_medium.yaml
│       └── gov_critical_infrastructure_hard.yaml
│
├── missions/
│   ├── __init__.py
│   ├── schema.py                        # TaskSpec, MissionSpec, TaskStatus (MC-1)
│   ├── engine.py                        # MissionEngine: task queue + lifecycle (MC-2)
│   └── policies.py                      # GreedyPolicy, LookaheadPolicy
│
├── envs/
│   ├── __init__.py
│   ├── base.py                          # UAVBenchEnv(gymnasium.Env, ABC) (EN-1)
│   └── urban.py                         # UrbanEnvV2(UAVBenchEnv) (EN-2..EN-9)
│
├── dynamics/
│   ├── __init__.py
│   ├── fire_ca.py                       # FireSpreadModel (cellular automaton)
│   ├── traffic.py                       # TrafficModel (road-following vehicles)
│   ├── restriction_zones.py             # MissionRestrictionModel (dynamic NFZ)
│   ├── interaction_engine.py            # InteractionEngine (cross-layer coupling)
│   ├── moving_target.py                 # MovingTargetModel
│   ├── intruder.py                      # IntruderModel
│   └── population_risk.py              # PopulationRiskModel (non-blocking)
│
├── planners/
│   ├── __init__.py                      # PLANNERS registry (PL-3)
│   ├── base.py                          # PlannerBase ABC + PlanResult (PL-1, PL-2)
│   ├── astar.py                         # AStarPlanner
│   ├── theta_star.py                    # ThetaStarPlanner
│   ├── periodic_replan.py               # PeriodicReplanPlanner
│   ├── aggressive_replan.py             # AggressiveReplanPlanner
│   ├── dstar_lite.py                    # DStarLitePlanner (true incremental)
│   └── mppi_grid.py                     # GridMPPIPlanner
│
├── guardrail/
│   ├── __init__.py
│   └── feasibility.py                   # Multi-depth relaxation (GC-1, GC-2)
│
├── blocking.py                          # compute_blocking_mask(state) (MP-1)
│
├── benchmark/
│   ├── __init__.py
│   ├── runner.py                        # BenchmarkRunner (RU-1, RU-3, RU-4)
│   ├── determinism.py                   # hash_episode(), verify_determinism()
│   └── fairness.py                      # BFS reference corridor, interdiction placement
│
├── metrics/
│   ├── __init__.py
│   ├── schema.py                        # EpisodeMetrics, AggregateMetrics (ME-1..ME-4)
│   └── compute.py                       # compute_episode_metrics(), aggregate()
│
└── visualization/
    ├── __init__.py
    ├── renderer.py                      # OperationalRenderer (VZ-1, VZ-2)
    ├── overlays.py                      # Mission-specific overlays
    └── hud.py                           # HUD rendering + tokens
```

---

## 2. Single Pipeline Architecture

There is exactly ONE execution path for running an episode:

```
CLI (cli/benchmark.py)
  │
  ▼
Runner (benchmark/runner.py)              ← owns authoritative step_idx (EV-1)
  │
  ├─ load_scenario(id) → ScenarioConfig
  ├─ create UrbanEnvV2(config)
  ├─ env.reset(seed=seed) → obs, info
  ├─ env.export_planner_inputs() → (heightmap, no_fly, start_xy, goal_xy)
  ├─ create planner via PLANNERS[planner_id](heightmap, no_fly, config)
  ├─ planner.plan(start, goal) → PlanResult         ← initial plan
  │
  ├─ EPISODE LOOP (step_idx = 0 .. max_steps-1):
  │   ├─ action = path_to_action(path, path_idx)
  │   ├─ obs, reward, terminated, truncated, info = env.step(action)
  │   │   └─ env internally calls:
  │   │       ├─ compute_blocking_mask(state)         ← MP-1: ONE mask
  │   │       ├─ move proposal + constraint checks
  │   │       ├─ dynamics.step() for each enabled layer
  │   │       ├─ guardrail.check(agent_xy, goal_xy)   ← uses same mask
  │   │       └─ reward computation
  │   ├─ planner.update(env.get_dynamic_state())
  │   ├─ should_replan, reason = planner.should_replan(pos, path, dyn, step)
  │   ├─ if should_replan: planner.plan(pos, goal) → new PlanResult
  │   ├─ env.set_plan_info(plan_len, plan_age, plan_reason)
  │   ├─ if renderer: renderer.render_frame(...)
  │   └─ if terminated or truncated: break
  │
  ├─ compute_episode_metrics(trajectory, events, config) → EpisodeMetrics
  └─ return EpisodeMetrics
```

**Invariants:**
- No second runner. No second CLI. No `run_planner_once` vs `run_dynamic_episode` split.
- The same code path handles static (no dynamics) and dynamic scenarios — static scenarios
  simply have all dynamics disabled, so the dynamics loop is a no-op.
- The runner's `step_idx` is THE step counter (EV-1).

---

## 3. Key Data Contracts

### 3.1 ScenarioConfig → Environment

```python
@dataclass(frozen=True)
class ScenarioConfig:
    # Identity
    name: str
    domain: Domain                           # Enum: URBAN
    difficulty: Difficulty                   # Enum: EASY, MEDIUM, HARD
    mission_type: MissionType               # Enum: CIVIL_PROTECTION, MARITIME_DOMAIN, CRITICAL_INFRASTRUCTURE
    regime: Regime                           # Enum: NATURALISTIC, STRESS_TEST
    paper_track: Literal["static", "dynamic"]

    # Map
    map_size: int                            # Grid dimension (500 for all gov scenarios)
    map_source: Literal["osm", "synthetic"]
    osm_tile_id: str | None
    building_density: float                  # [0, 1]

    # Start/Goal
    fixed_start_xy: tuple[int, int] | None
    fixed_goal_xy: tuple[int, int] | None
    min_start_goal_l1: int

    # Dynamic layers (bool flags)
    enable_fire: bool
    enable_traffic: bool
    enable_dynamic_nfz: bool
    fire_blocks_movement: bool
    traffic_blocks_movement: bool
    terminate_on_collision: bool

    # ... (full field list in scenarios/schema.py)
```

### 3.2 Environment → Planner

```python
# Via export_planner_inputs()
heightmap: np.ndarray          # [H, W] float32; > 0 = building
no_fly: np.ndarray             # [H, W] bool
start_xy: tuple[int, int]     # (x, y)
goal_xy: tuple[int, int]      # (x, y)

# Via get_dynamic_state()
dynamic_state: dict[str, Any]  # fire_mask, smoke_mask, traffic_positions, etc.
```

### 3.3 Planner → Runner

```python
@dataclass
class PlanResult:
    path: list[tuple[int, int]]    # (x, y) waypoints, inclusive start+goal
    success: bool
    compute_time_ms: float
    expansions: int = 0
    replans: int = 0
    reason: str = ""               # failure explanation
```

### 3.4 Runner → Metrics

```python
@dataclass
class EpisodeMetrics:
    scenario_id: str
    planner_id: str
    seed: int
    success: bool
    termination_reason: TerminationReason
    objective_completed: bool
    path_length: int
    executed_steps_len: int
    planned_waypoints_len: int
    planning_time_ms: float
    replans: int
    collision_count: int
    nfz_violations: int
    # ... (full field list in metrics/schema.py)
```

### 3.5 Environment → Renderer

```python
# Via info dict (every step)
info: dict containing:
    step_idx: int
    agent_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    accepted_move: bool
    reject_reason: RejectReason | None
    plan_len: int
    plan_age_steps: int
    plan_reason: str
    mission_domain: str
    objective_label: str
    distance_to_task: float
    task_progress: str
    deliverable_name: str
    forced_block_active: bool
    guardrail_depth: int
    feasible_after_guardrail: bool
    termination_reason: TerminationReason
    objective_completed: bool
```

---

## 4. Dependency Graph (Module Imports)

```
cli/benchmark.py
  └── benchmark/runner.py
        ├── scenarios/loader.py → scenarios/schema.py
        ├── envs/urban.py → envs/base.py
        │     ├── blocking.py              ← THE single mask function
        │     ├── dynamics/*               ← all dynamic layers
        │     ├── guardrail/feasibility.py ← uses blocking.py
        │     └── missions/engine.py
        ├── planners/__init__.py → planners/base.py + all 6 planners
        ├── metrics/compute.py → metrics/schema.py
        └── visualization/renderer.py → visualization/overlays.py, hud.py
```

**Import rules:**
- `blocking.py` is imported by `envs/urban.py` AND `guardrail/feasibility.py` — same function
- Planners NEVER import env internals; they receive `(heightmap, no_fly, cost_map)` only
- Dynamics modules NEVER import each other; interaction_engine receives all states as arguments
- Visualization NEVER imports env internals; it receives all data via the runner

---

## 5. Enum Catalog

```python
class Domain(str, Enum):
    URBAN = "urban"

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class MissionType(str, Enum):
    CIVIL_PROTECTION = "civil_protection"
    MARITIME_DOMAIN = "maritime_domain"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"

class Regime(str, Enum):
    NATURALISTIC = "naturalistic"
    STRESS_TEST = "stress_test"

class RejectReason(str, Enum):
    BUILDING = "building"
    NO_FLY = "no_fly"
    FORCED_BLOCK = "forced_block"
    TRAFFIC_CLOSURE = "traffic_closure"
    FIRE = "fire"
    TRAFFIC_BUFFER = "traffic_buffer"
    MOVING_TARGET = "moving_target"
    INTRUDER = "intruder"
    DYNAMIC_NFZ = "dynamic_nfz"

class TerminationReason(str, Enum):
    SUCCESS = "success"
    COLLISION_BUILDING = "collision_building"
    COLLISION_NFZ = "collision_nfz"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    IN_PROGRESS = "in_progress"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    SKIPPED = "skipped"

class BlockLifecycle(str, Enum):
    PENDING = "pending"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    CLEARED = "cleared"

class ReplanTrigger(str, Enum):
    INITIAL = "initial"
    FORCED = "forced"
    PATH_INVALIDATED = "path_invalidated"
    CADENCE = "cadence"
    STUCK = "stuck"
    RISK_SPIKE = "risk_spike"
```

---

## 6. RNG Threading Architecture (DC-1)

```
reset(seed=S)
  └── root_rng = np.random.default_rng(S)
        ├── env_rng = root_rng.spawn(1)[0]          # map generation, start/goal
        ├── fire_rng = root_rng.spawn(1)[0]          # fire ignition, spread
        ├── traffic_rng = root_rng.spawn(1)[0]       # vehicle spawning, movement
        ├── nfz_rng = root_rng.spawn(1)[0]           # zone placement
        ├── target_rng = root_rng.spawn(1)[0]        # target movement noise
        ├── intruder_rng = root_rng.spawn(1)[0]      # intruder spawning, noise
        ├── risk_rng = root_rng.spawn(1)[0]          # population risk init
        └── planner_rng = root_rng.spawn(1)[0]       # MPPI sampling
```

Each component receives its own child generator. No component creates its own RNG.
All stochastic calls use the assigned generator. This guarantees DC-1 by construction.

---

## 7. Blocking Mask Architecture (MP-1)

```python
# blocking.py — THE single source of truth
def compute_blocking_mask(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    config: ScenarioConfig,
    dynamic_state: dict[str, Any],
) -> np.ndarray:
    """
    Returns bool[H, W] mask where True = blocked.
    Used by BOTH env.step() legality AND guardrail BFS.
    Enforces MP-1.
    """
    mask = (heightmap > 0) | no_fly

    if dynamic_state.get("forced_block_mask") is not None:
        mask |= dynamic_state["forced_block_mask"]
    if dynamic_state.get("traffic_closure_mask") is not None:
        mask |= dynamic_state["traffic_closure_mask"]
    if config.fire_blocks_movement and dynamic_state.get("fire_mask") is not None:
        mask |= dynamic_state["fire_mask"]
    if config.fire_blocks_movement and dynamic_state.get("smoke_mask") is not None:
        mask |= (dynamic_state["smoke_mask"] >= 0.3)
    if config.traffic_blocks_movement and dynamic_state.get("traffic_occupancy_mask") is not None:
        mask |= dynamic_state["traffic_occupancy_mask"]
    if dynamic_state.get("moving_target_buffer") is not None:
        mask |= dynamic_state["moving_target_buffer"]
    if dynamic_state.get("intruder_buffer") is not None:
        mask |= dynamic_state["intruder_buffer"]
    if dynamic_state.get("dynamic_nfz_mask") is not None:
        mask |= dynamic_state["dynamic_nfz_mask"]

    return mask
```
