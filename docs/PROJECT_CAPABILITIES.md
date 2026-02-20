# PROJECT_CAPABILITIES.md — System Capability Map

**Date:** 2025-07-14  
**Scope:** Every testable capability in the codebase, with file:line citations  
**Legend:** ✅ = fully implemented + tested | ⚙️ = implemented, not wired into CLI | 🧩 = infrastructure exists, underutilized

---

## 1. Environment Layer

### 1.1 Gymnasium Environment

| Capability | File:Line | Status | Tests |
|---|---|---|---|
| `UrbanEnv(UAVBenchEnv)` — Discrete(6) action space, 4-connected grid | `envs/urban.py:22` | ✅ | `test_urban_env_basic.py` |
| 500×500 OSM tile maps (Penteli, Piraeus, Lavrio) | `envs/urban.py:50-80` | ✅ | `test_sanity.py` |
| Heightmap-based building obstacles | `envs/urban.py:85` | ✅ | `test_collision_termination.py` |
| Static no-fly zone mask | `envs/urban.py:86` | ✅ | `test_collision_termination.py` |
| Altitude levels (multi-layer) | `envs/urban.py:55` | ✅ | `test_sanity.py` |
| Emergency corridor BFS (planner-agnostic) | `envs/urban.py:929-1049` | ✅ | `test_planning_correctness.py` |
| Interdiction placement via BFS shortest path | `envs/urban.py:929-1049` | ✅ | `test_fair_protocol.py` |

### 1.2 Dynamics Models

| Dynamics | Class | File:Line | Used in CLI? | Tests |
|---|---|---|---|---|
| Fire spread (cellular automata, wind, ignition) | `FireSpreadModel` | `dynamics/fire_spread.py:39` | ✅ | `test_benchmark_cli.py` |
| Traffic (road-constrained vehicles) | `TrafficModel` | `dynamics/traffic.py:12` | ✅ | `test_benchmark_cli.py` |
| Restriction zones (expanding polygons, incident-linked) | `RestrictionZone` | `dynamics/restriction_zones.py:39` | ✅ | `test_restriction_zones.py` |
| Dynamic NFZ (time-varying no-fly) | `DynamicNFZModel` | `dynamics/dynamic_nfz.py:16` | ✅ | `test_benchmark_cli.py` |
| Moving target (mobile goal) | `MovingTargetModel` | `dynamics/moving_target.py:17` | ✅ | `test_benchmark_cli.py` |
| Intruder (adversarial obstacle) | `IntruderModel` | `dynamics/intruder.py:16` | ✅ | `test_benchmark_cli.py` |
| Adversarial UAV (pursuit dynamics) | `AdversarialUAVModel` | `dynamics/adversarial_uav.py:8` | ✅ | `test_benchmark_cli.py` |
| Population risk (density heatmap) | `PopulationRiskModel` | `dynamics/population_risk.py:12` | ✅ | `test_benchmark_cli.py` |
| Causal interaction engine (fire↔traffic couplings) | `InteractionEngine` | `dynamics/interaction_engine.py:10` | ✅ | `test_interaction_causality.py` |

---

## 2. Planner Layer

### 2.1 Planner Registry

| Key | Class | File:Line | Type | Algorithm |
|---|---|---|---|---|
| `astar` | `AStarPlanner` | `planners/astar.py:21` | `BasePlanner` | A* (Manhattan heuristic) |
| `theta_star` | `ThetaStarPlanner` | `planners/theta_star.py:27` | `BasePlanner` | Theta* (any-angle A*) |
| `periodic_replan` | `PeriodicReplanPlanner` | `planners/dstar_lite.py:34` | `AdaptiveAStarPlanner` | A* every 6 steps |
| `aggressive_replan` | `AggressiveReplanPlanner` | `planners/ad_star.py:37` | `AdaptiveAStarPlanner` | A* every 4 steps |
| `greedy_local` | `GreedyLocalPlanner` | `planners/dwa.py:38` | `BasePlanner` | 1-step greedy toward goal |
| `grid_mppi` | `GridMPPIPlanner` | `planners/mppi.py:52` | `BasePlanner` | K-rollout sampling |
| `dstar_lite` | alias → `PeriodicReplanPlanner` | `planners/__init__.py:37` | — | Legacy alias |
| `ad_star` | alias → `AggressiveReplanPlanner` | `planners/__init__.py:38` | — | Legacy alias |
| `dwa` | alias → `GreedyLocalPlanner` | `planners/__init__.py:39` | — | Legacy alias |
| `mppi` | alias → `GridMPPIPlanner` | `planners/__init__.py:40` | — | Legacy alias |

### 2.2 Planner Infrastructure

| Capability | Class/Function | File:Line | Status |
|---|---|---|---|
| Abstract planner interface | `BasePlanner` | `planners/base.py:36` | ✅ |
| Plan result dataclass | `PlanResult` | `planners/base.py:28` | ✅ |
| Optional dynamic update API | `BasePlanner.update()` | `planners/base.py:99` | ✅ |
| Optional should_replan API | `BasePlanner.should_replan()` | `planners/base.py:112` | ✅ |
| Adaptive A* base (replan via fresh A*) | `AdaptiveAStarPlanner` | `planners/adaptive_astar.py:30` | ✅ |
| Path blockage detection | `AdaptiveAStarPlanner._is_path_blocked()` | `planners/adaptive_astar.py:182` | ✅ |
| Adaptive interval (proximity-based) | `AdaptiveAStarPlanner._adaptive_interval()` | `planners/adaptive_astar.py:214` | ✅ |
| PlannerAdapter (bus integration) | `PlannerAdapter` | `planners/adapter.py:178` | ⚙️ |
| ReplanPolicy (trigger logic) | `ReplanPolicy` | `planners/adapter.py:107` | ⚙️ |
| ReplanRecord (causal logging) | `ReplanRecord` | `planners/adapter.py:55` | ⚙️ |
| Replan triggers: CADENCE, EVENT, RISK_SPIKE, FORCED | `ReplanTrigger` | `planners/adapter.py:40` | ⚙️ |

---

## 3. Updates / Event System

| Capability | Class | File:Line | Status |
|---|---|---|---|
| Typed event bus (pub/sub) | `UpdateBus` | `updates/bus.py:108` | ✅ |
| Event types: OBSTACLE, CONSTRAINT, RISK, TASK, COMMS, REPLAN | `EventType` | `updates/bus.py:40` | ✅ |
| UpdateEvent envelope (UUID, severity, mask, provenance chain) | `UpdateEvent` | `updates/bus.py:55` | ✅ |
| Conflict detector (path-obstacle intersection) | `ConflictDetector` | `updates/conflict.py:68` | ✅ |
| Merged obstacle builder | `ConflictDetector.merge_obstacles()` | `updates/conflict.py:213` | ✅ |
| Safety monitor (SC-1 to SC-4 contracts) | `SafetyMonitor` | `updates/safety.py:59` | ✅ |
| Fail-safe hover | `SafetyMonitor.fail_safe_active` | `updates/safety.py:189` | ✅ |
| Forced replan scheduler (≥2 replans guaranteed) | `ForcedReplanScheduler` | `updates/forced_replan.py:55` | ✅ |
| Dynamic obstacle manager (vehicles, vessels, work zones) | `DynamicObstacleManager` | `updates/obstacles.py:424` | ✅ |
| Vehicle layer (M1 — road-constrained) | `VehicleLayer` | `updates/obstacles.py:69` | ✅ |
| Vessel layer (M2 — AIS-like kinematics) | `VesselLayer` | `updates/obstacles.py:194` | ✅ |
| Work zone layer (M3 — slow-moving areas) | `WorkZoneLayer` | `updates/obstacles.py:318` | ✅ |

---

## 4. Mission Framework

| Capability | Class/Function | File:Line | Status |
|---|---|---|---|
| Mission spec (immutable template) | `MissionSpec` | `missions/spec.py:140` | ⚙️ |
| Task spec (POI, weight, decay, time window, service time) | `TaskSpec` | `missions/spec.py:53` | ⚙️ |
| Difficulty knobs (easy/med/hard factory methods) | `DifficultyKnobs` | `missions/spec.py:76` | ⚙️ |
| Mission engine (task tracking, injections, utility scoring) | `MissionEngine` | `missions/engine.py:1` | ⚙️ |
| Injection events (mid-episode task/constraint injection) | `InjectionEvent` | `missions/engine.py` | ⚙️ |
| Greedy policy (nearest-unvisited, decay-aware) | `GreedyPolicy` | `missions/policies.py:47` | ⚙️ |
| Lookahead OPTW policy (bounded-depth orienteering) | `LookaheadOPTWPolicy` | `missions/policies.py:82` | ⚙️ |
| Mission 1 builder (civil protection: wildfire + corridor) | `build_civil_protection()` | `missions/builders.py:53` | ⚙️ |
| Mission 2 builder (maritime: patrol + distress) | `build_maritime_domain()` | `missions/builders.py:143` | ⚙️ |
| Mission 3 builder (critical infra: inspection tour + NFZ) | `build_critical_infrastructure()` | `missions/builders.py:227` | ⚙️ |
| plan_mission() API (end-to-end standalone runner) | `plan_mission()` | `missions/runner.py:68` | ⚙️ |
| Product export (CSV/JSON) | `export_products_csv()` | `missions/runner.py:282` | ⚙️ |
| Operational products (9 product types) | `ProductType` | `missions/spec.py:35` | ⚙️ |
| Common metric keys (10 standard metrics) | `COMMON_METRICS` | `missions/spec.py:168` | ⚙️ |

**Legend for ⚙️:** Fully implemented and tested in `test_mission_bank.py` (461 LoC), but **not wired into the CLI benchmark** (`cli/benchmark.py`). The mission framework operates as a standalone library.

---

## 5. Benchmark Infrastructure

| Capability | Function/Class | File:Line | Status |
|---|---|---|---|
| Static episode runner | `run_planner_once()` | `cli/benchmark.py:231` | ✅ |
| Dynamic episode runner (main loop) | `run_dynamic_episode()` | `cli/benchmark.py:352` | ✅ |
| Scenario loading | `scenario_path()` | `cli/benchmark.py:24` | ✅ |
| Replanning fairness gate | `if True:` gate | `cli/benchmark.py:634` | ✅ |
| Harness-level replanning dispatch | native vs harness | `cli/benchmark.py:725-748` | ✅ |
| P1: constraint latency FIFO | FIFO buffer | `cli/benchmark.py:637-642` | ✅ |
| P1: comms dropout | stale delivery | `cli/benchmark.py:643-651` | ✅ |
| P1: GNSS noise | Gaussian on planner_pos | `cli/benchmark.py:679-688` | ✅ |
| Solvability guarantees (BFS corridors) | `benchmark/solvability.py` | `benchmark/solvability.py:1` | ✅ |
| Fairness audit (seed reproducibility, snapshot equality) | `benchmark/fairness_audit.py` | `benchmark/fairness_audit.py:1` | ✅ |
| Theoretical validation (H1-H7, Cohen's d, CI95) | `benchmark/theoretical_validation.py` | `benchmark/theoretical_validation.py:1` | ✅ |

---

## 6. Scenario Layer

| Capability | File:Line | Status |
|---|---|---|
| Scenario schema (ScenarioConfig dataclass) | `scenarios/schema.py:1` | ✅ |
| YAML config loader | `scenarios/loader.py` | ✅ |
| Dynamic scenario registry (9 scenarios) | `scenarios/registry.py:1` | ✅ |
| Filter by mission/regime/difficulty/dynamics/tile | `scenarios/registry.py:100-200` | ✅ |
| Paper track annotation (static/dynamic) | `scenarios/registry.py:64` | ✅ |
| Incident provenance fields (name, year, refs) | YAML `incident_*` | ✅ |

---

## 7. Visualization Layer

| Capability | Class/Function | File:Line | Status |
|---|---|---|---|
| Operational renderer (multi-panel dashboard) | `OperationalRenderer` | `visualization/operational_renderer.py:250` | ✅ |
| Stakeholder renderer (stakeholder-facing dashboards) | `StakeholderRenderer` | `visualization/stakeholder_renderer.py:298` | ✅ |
| Demo pack generator | `visualization/demo_pack.py` | — | ✅ |
| Video export | `visualization/` | — | ✅ |

---

## 8. Capability Integration Matrix

Which capabilities flow through the CLI benchmark vs standalone-only:

| Capability | CLI benchmark | Standalone | Gap? |
|---|---|---|---|
| A* / Theta* planning | ✅ | ✅ | — |
| Adaptive replanning | ✅ | ✅ | — |
| Fire / traffic dynamics | ✅ | ✅ | — |
| Restriction zones / dynamic NFZ | ✅ | ✅ | — |
| Interaction engine | ✅ | ✅ | — |
| Safety monitor | ✅ (partial) | ✅ | — |
| UpdateBus / PlannerAdapter | ⚙️ exists | ✅ | Not primary path |
| **Mission engine** | ❌ | ✅ | **G1: Not wired** |
| **Multi-task episodes** | ❌ | ✅ | **G1: Not wired** |
| **Task injection** | ❌ | ✅ | **G1: Not wired** |
| **Utility scoring** | ❌ | ✅ | **G1: Not wired** |
| **Operational products** | ❌ | ✅ | **G1: Not wired** |
| **True incremental search** | ❌ | ❌ | **G2: Not implemented** |
