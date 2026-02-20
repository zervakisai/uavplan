# Stakeholder Visualization System V2 — Architecture & Design

> Planning Correctness Infrastructure for UAVBench  
> Dynamic Obstacles · Forced Replanning · Causal Logging · Safety Contracts

---

## 1. Overview

V2 upgrades the stakeholder visualization with a **planning correctness
infrastructure** that ensures planners visibly react to dynamic obstacles.
Every mission episode now produces:

- ≥2 forced replans (deterministic, seed-controlled)
- Full causal chain: obstacle event → conflict detection → replan trigger → new path
- Safety contract enforcement with violation counting
- Rich exportable products (GeoJSON, CSV per mission type)

## 2. Architecture

```
                          ┌─────────────────┐
                          │   UpdateBus      │
                          │  (pub/sub)       │
                          └──┬───┬───┬──────┘
            ┌───────────────┘   │   └────────────────┐
            ▼                   ▼                    ▼
   ┌────────────────┐  ┌──────────────┐   ┌──────────────────┐
   │ DynamicObstacle │  │   Forced     │   │   SafetyMonitor  │
   │ Manager         │  │   Replan     │   │   SC-0..SC-3     │
   │                 │  │   Scheduler  │   │                  │
   │ • VehicleLayer  │  │   (33%/66%)  │   │ • Violations     │
   │ • VesselLayer   │  │              │   │ • Fail-safe      │
   │ • WorkZoneLayer │  └──────┬───────┘   └──────────────────┘
   └────────┬────────┘         │
            │    OBSTACLE      │ OBSTACLE (forced=True)
            ▼    events        ▼
   ┌──────────────────────────────────────────┐
   │            PlannerAdapter                 │
   │                                           │
   │  ┌─────────────────┐  ┌───────────────┐  │
   │  │ ConflictDetector │  │  ReplanPolicy │  │
   │  │ (path∩obstacles) │  │  FORCED →     │  │
   │  │                  │  │  EVENT →      │  │
   │  └──────────────────┘  │  RISK_SPIKE → │  │
   │                        │  CADENCE      │  │
   │                        └───────────────┘  │
   │                                           │
   │  ── wraps any BasePlanner ──              │
   └───────────────┬──────────────────────────┘
                   │
                   ▼ REPLAN events (causal chain)
   ┌──────────────────────────────────┐
   │        MissionEngine             │
   │   (plan_mission_v2 loop)         │
   │                                  │
   │   trajectory → StakeholderViz    │
   │   products  → GeoJSON/CSV       │
   └──────────────────────────────────┘
```

## 3. Module Reference

### 3.1 UpdateBus (`updates/bus.py`)

Central event pipeline. All dynamic changes flow through typed `UpdateEvent`
envelopes.

| Class | Description |
|-------|-------------|
| `EventType` | Enum: OBSTACLE, CONSTRAINT, RISK, TASK, COMMS, REPLAN |
| `UpdateEvent` | Typed envelope with event_id, parent_id, position, mask, severity |
| `UpdateBus` | Pub/sub: subscribe per-type or wildcard, severity filter, drain, replay |

### 3.2 ConflictDetector (`updates/conflict.py`)

Geometric intersection between planned paths and live obstacles.

| Method | Description |
|--------|-------------|
| `check_path()` | Full check against obstacle_mask, nfz, risk_map, vehicles |
| `is_path_feasible()` | Fast hard-conflict-only check |
| `merge_obstacles()` | Build merged bool mask from fire/smoke/traffic/vehicles |

### 3.3 SafetyMonitor (`updates/safety.py`)

Enforces four safety contracts every step:

| Contract | Rule |
|----------|------|
| SC-0 | Bounds check — agent within grid |
| SC-1 | No ghosting through buildings (`heightmap > 0`) |
| SC-2 | No entry into static or dynamic NFZ |
| SC-3 | No occupation of dynamic obstacle cells |

Fail-safe: after `fail_safe_threshold` (default 3) consecutive violations,
agent holds position (hover).

### 3.4 Dynamic Obstacles (`updates/obstacles.py`)

| Layer | Mission | Kinematics |
|-------|---------|------------|
| `VehicleLayer` | M1 Civil Protection | Waypoint-following, road-constrained, speed ≤ 1.5 cells/step |
| `VesselLayer` | M2 Maritime | Circular patrol, turn-rate constraint, AIS-like tracks |
| `WorkZoneLayer` | M3 Critical Infra | Slow-moving circular areas, staggered activation |
| `DynamicObstacleManager` | All | Orchestrates layers, publishes OBSTACLE events |

### 3.5 ForcedReplanScheduler (`updates/forced_replan.py`)

Guarantees ≥2 replans per episode:

1. After initial plan, picks path points at ~33% and ~66%
2. At those steps, injects circular blocking obstacle ON the path
3. Registers forced steps with `ReplanPolicy`
4. Deterministic: same seed → same injection steps

### 3.6 PlannerAdapter (`planners/adapter.py`)

Wraps any `BasePlanner` with bus integration.

| Component | Responsibility |
|-----------|---------------|
| `ReplanTrigger` | Enum: INITIAL, FORCED, EVENT, RISK_SPIKE, CADENCE |
| `ReplanRecord` | Immutable log entry with replan_id, trigger, cost delta, causal parent |
| `ReplanPolicy` | Priority evaluation: FORCED → EVENT → RISK_SPIKE → CADENCE |
| `PlannerAdapter` | Subscribes to bus, maintains merged obstacles, plan/replan with logging |

### 3.7 Mission Runner V2 (`missions/runner_v2.py`)

Episode loop pseudocode:

```
while not engine.done:
    A. obstacle_mgr.step()          # advance dynamic obstacles
    B. forced_scheduler.step()      # inject forced blockers
    C. adapter.update_dynamic_state # sync adapter
    D. if no path → pick task, plan
    E. adapter.step_check()         # mid-path conflict check
       → if triggered: adapter.try_replan()
    F. follow path one step
    G. safety.check() before move   # enforce contracts
    H. engine.step()                # advance mission state
    I. generate products on completion
```

### 3.8 Export (`visualization/export.py`)

| Function | Output |
|----------|--------|
| `export_trajectory_geojson()` | UAV trajectory as GeoJSON LineString |
| `export_points_geojson()` | Point features (fire perimeter, etc.) |
| `export_csv()` | Generic dict-list → CSV |
| `export_replan_log_csv()` | Full causal replan chain |
| `export_event_bus_csv()` | All UpdateBus events (objects or dicts) |
| `export_all()` | All-in-one: trajectory + replan_log + metrics + products |

### 3.9 StakeholderRenderer Upgrades

V2 parameters added to `render_frame()`:

| Parameter | Description |
|-----------|-------------|
| `conflict_markers` | List of `(x, y, severity)` — red X markers + collision rings |
| `violation_flash` | If `True`, red border pulse |
| `replan_annotations` | List of `(x, y, trigger_label)` — callout arrows |
| `dynamic_obstacle_mask` | `[H,W]` bool — semi-transparent orange overlay at Z4 |

Metrics panel now includes **Violations** row and updated **Replans** display.

## 4. Data Flow — Causal Chain Example

```
Step 47:  VehicleLayer moves vehicle_0 → position (30, 25)
          ↓ OBSTACLE event (severity=0.6, event_id="a1b2c3")
Step 47:  PlannerAdapter._on_obstacle() → updates _obstacle_mask
Step 48:  adapter.step_check() → ConflictDetector finds conflict at path[8]
          ↓ ReplanPolicy.evaluate() → EVENT trigger
Step 48:  adapter.try_replan() → BasePlanner.plan() with merged obstacles
          ↓ REPLAN event (parent_id="a1b2c3", severity=0.7)
          ↓ ReplanRecord logged: trigger=EVENT, reason="vehicle_at_(30,25)"
Step 48:  StakeholderRenderer receives:
          - conflict_markers=[(30, 25, 1.0)]  → red X
          - replan_annotations=[(30, 25, "EVENT")]  → callout arrow
```

## 5. Test Coverage

**85 acceptance tests** in `tests/test_planning_correctness.py`:

| Test Class | Count | Covers |
|-----------|-------|--------|
| `TestUpdateBus` | 11 | Pub/sub, drain, replay, causal chain |
| `TestConflictDetector` | 9 | Path-obstacle intersection, merge, bounds |
| `TestReplanPolicy` | 6 | Priority order, forced, cadence, risk, cap |
| `TestPlannerAdapter` | 6 | Bus integration, replan logging, state |
| `TestSafetyMonitor` | 10 | SC-0..SC-3, fail-safe, bus publish |
| `TestDynamicObstacles` | 14 | All 3 layers, manager, kinematics |
| `TestForcedReplanScheduler` | 6 | ≥2 guarantee, injection, short path |
| `TestPlanMissionV2` | 13 | E2E determinism, ≥2 replans, all missions |
| `TestExport` | 10 | GeoJSON, CSV, numpy serialization |

**Total suite: 245 tests (all passing)**

## 6. File Inventory

```
src/uavbench/
├── updates/
│   ├── __init__.py          # Package exports
│   ├── bus.py               # EventType, UpdateEvent, UpdateBus
│   ├── conflict.py          # Conflict, ConflictDetector
│   ├── safety.py            # Violation, SafetyConfig, SafetyMonitor
│   ├── obstacles.py         # Vehicle/Vessel/WorkZone layers, Manager
│   └── forced_replan.py     # ForcedReplanScheduler
├── planners/
│   └── adapter.py           # ReplanTrigger, ReplanRecord, ReplanPolicy, PlannerAdapter
├── missions/
│   └── runner_v2.py         # MissionResultV2, plan_mission_v2()
└── visualization/
    ├── stakeholder_renderer.py  # V2 conflict/violation/replan overlays
    └── export.py                # GeoJSON/CSV export
```
