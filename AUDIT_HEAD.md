# UAVBench — HEAD-Accurate Audit Report

**Standard:** ICRA / IROS / RA-L / NeurIPS Datasets & Benchmarks  
**Method:** Automated static analysis of every `.py` file and scenario YAML at HEAD  
**Date:** 2025-02-19  
**Codebase:** 18,071 LoC (`src/`), 3,805 LoC (`tests/`), 292 tests passing, 9 YAML scenarios

---

## 0. Delta vs Previous Audit

| Item | Previous audit | HEAD truth | Changed? |
|------|---------------|------------|----------|
| Source LoC | 17,039 | 18,071 | +1,032 |
| Test LoC | 3,372 | 3,805 | +433 |
| Tests passing | not run | 292/292 | — |
| Scenario YAMLs | "12" | **9** (3 missions × 3 difficulties) | ❌ Previous over-counted |
| `AdaptiveAStarPlanner` base class | `BasePlanner` subclass | **Standalone class** (no `BasePlanner` inheritance) | ❌ Previous missed |
| `BasePlanner.should_replan` | "only adaptive" | **Default impl exists** in `base.py:112-131` | ❌ Previous missed |
| D\*Lite docstring | "D* Lite inspired incremental replanner" | Unchanged (`dstar_lite.py:1`) | ✅ |
| AD\* docstring | "AD* inspired anytime-dynamic replanner" | Unchanged (`ad_star.py:1`) | ✅ |

---

## A. System Capability Map (Code-Cited)

### A1. Planner Inventory

| Registry key | Class | File:lines | Base | `should_replan`? | `replan`? | Canonical? |
|---|---|---|---|---|---|---|
| `astar` | `AStarPlanner` | `planners/astar.py:1-184` | `BasePlanner` | inherited default | ❌ | ✅ |
| `theta_star` | `ThetaStarPlanner` | `planners/theta_star.py:1-249` | `BasePlanner` | inherited default | ❌ | ✅ (Bresenham LOS, 8-conn) |
| `dstar_lite` | `DStarLitePlanner` | `planners/dstar_lite.py:1-31` | `AdaptiveAStarPlanner` | ✅ inherited | ✅ inherited | ❌ — config variant only |
| `ad_star` | `ADStarPlanner` | `planners/ad_star.py:1-32` | `AdaptiveAStarPlanner` | ✅ inherited | ✅ inherited | ❌ — config variant only |
| `dwa` | `DWAPlanner` | `planners/dwa.py:1-66` | `BasePlanner` | inherited default | ❌ | ❌ — greedy 1-step |
| `mppi` | `MPPIPlanner` | `planners/mppi.py:1-206` | `BasePlanner` | inherited default | ❌ | ⚠️ — 4-cardinal discretize |
| *(not in registry)* | `AdaptiveAStarPlanner` | `planners/adaptive_astar.py:1-295` | standalone | ✅ own impl | ✅ own impl | ✅ |

**Note:** `AdaptiveAStarPlanner` is NOT in `PLANNERS` registry (`planners/__init__.py:11-17`).
It is only used as base class for D\*Lite and AD\*.

### A2. Scenario Inventory (9 YAMLs)

| Scenario ID | Mission | Diff | Track | Tile | Fire | Traffic | Dyn NFZ | Wind | Forced replans | Incident |
|---|---|---|---|---|---|---|---|---|---|---|
| `gov_civil_protection_easy` | civil_protection | easy | static | penteli | ❌ | ❌ | ❌ | none | 0 | 2018 Attica Wildfire |
| `gov_civil_protection_medium` | civil_protection | medium | dynamic | penteli | ✅ | ✅ | ✅ | medium | 2 | 2018 Attica Wildfire |
| `gov_civil_protection_hard` | civil_protection | hard | dynamic | penteli | ✅ | ✅ | ✅ | high | 2 | 2018 Attica Wildfire |
| `gov_maritime_domain_easy` | maritime_domain | easy | static | piraeus | ❌ | ❌ | ❌ | none | 0 | — |
| `gov_maritime_domain_medium` | maritime_domain | medium | dynamic | piraeus | ✅ | ✅ | ✅ | medium | 2 | — |
| `gov_maritime_domain_hard` | maritime_domain | hard | dynamic | piraeus | ✅ | ✅ | ✅ | high | 2 | — |
| `gov_critical_infrastructure_easy` | critical_infrastructure | easy | static | downtown | ❌ | ❌ | ❌ | none | 0 | — |
| `gov_critical_infrastructure_medium` | critical_infrastructure | medium | dynamic | downtown | ✅ | ✅ | ✅ | medium | 2 | — |
| `gov_critical_infrastructure_hard` | critical_infrastructure | hard | dynamic | downtown | ✅ | ✅ | ✅ | high | 2 | — |

### A3. Dynamic Obstacles Catalogue

| Layer | Class | File | Updated in | Output mask/data | Blocks movement? |
|---|---|---|---|---|---|
| Fire | `FireSpreadModel` | `dynamics/fire_spread.py` | `urban.py` step (l.640-660) | `fire_mask`, `smoke_mask` | Yes if `fire_blocks_movement` |
| Traffic | `TrafficFlowModel` | `dynamics/traffic.py` | `urban.py` step (l.660-680) | `vehicle_positions`, `occupancy_mask` | Yes if `traffic_blocks_movement` |
| Moving Target | `MovingTargetModel` | `dynamics/moving_target.py` | `urban.py` step (l.680-695) | `moving_target_buffer` | Buffer blocks |
| Intruder | `IntruderModel` | `dynamics/intruder.py` | `urban.py` step (l.695-710) | `intruder_buffer` | Buffer blocks |
| Dynamic NFZ | `DynamicNFZModel` / `MissionRestrictionModel` | `dynamics/dynamic_nfz.py` / `dynamics/restriction_zones.py` | `urban.py` step (l.710-730) | `dynamic_nfz_mask`, `restriction_zones` | Mask blocks |
| Adversarial UAV | `AdversarialUAVModel` | `dynamics/adversarial_uav.py` | `urban.py` step (l.730-740) | risk contribution | Risk only |
| Population Risk | `PopulationRiskModel` | `dynamics/population_risk.py` | `urban.py` step (l.735) | risk contribution | Risk only |
| Forced Interdictions | inline in `UrbanEnv` | `urban.py:935-1040` | `_maybe_trigger_interdictions()` | `forced_block_mask` | Yes |
| Traffic Closures | `InteractionEngine` | `dynamics/interaction_engine.py` | `urban.py` step (l.730) | `traffic_closure_mask` | Yes |

### A4. Benchmark Harness Semantics (`cli/benchmark.py`)

**Replanning triggers** (l.636-680):
- `path_invalidation`: lookahead 6 cells ahead blocked (l.138-185)
- `forced_event`: interdiction triggered this step (l.646)
- `cadence`: `steps_since_replan >= replan_every_steps` (l.647)
- `stuck_fallback`: `stuck_counter >= 3` (l.649)
- `planner_signal`: `planner.should_replan()` returns True (l.651-660)

**Gate:** Only fires if `is_adaptive = hasattr(planner, "should_replan") and hasattr(planner, "replan")` (l.466)

**CRITICAL:** `BasePlanner` HAS a default `should_replan` (base.py:112-131) but does NOT have `replan`. So the gate evaluates to `False` for A\*, Theta\*, DWA, MPPI because none have `replan()`.

**Static planner stuck handling** (l.711-713):
```python
elif stuck_counter >= 10:
    # Static planner: give up if stuck too long
    break
```

**Time budgets:**
- Initial plan: `plan_budget_static_ms` (static track) or `plan_budget_dynamic_ms` (dynamic track)
- Replan: same `plan_budget_ms`, violations counted (l.697-706)

**Termination:**
- Goal reached (info: `reached_goal`)
- Collision (if `terminate_on_collision`)
- `max_replans_exceeded` (l.671)
- `stuck_counter >= 10` for static planners
- `max_steps = 4 * map_size` (l.503) or `episode_horizon_steps`

---

## B. P0 Findings (Confirmed Against HEAD)

### B1. Planner Naming — ❌ CONFIRMED

All evidence from previous audit confirmed at HEAD. Key citations:

| Planner | File:Line | Evidence |
|---|---|---|
| D\*Lite | `dstar_lite.py:21-31` | `class DStarLitePlanner(AdaptiveAStarPlanner)` — only sets `base_interval=6, lookahead_steps=8` |
| AD\* | `ad_star.py:22-32` | `class ADStarPlanner(AdaptiveAStarPlanner)` — only sets `base_interval=4, lookahead_steps=10` |
| DWA | `dwa.py:27-54` | `cur = min(nbrs, key=score)` — greedy 1-step, no velocity space |
| MPPI | `mppi.py:22-28, 190-196` | `_DIRECTIONS = [4 cardinal]`, `best_dir = argmax(dots)` — continuous→discrete collapse |

### B2. Replanning Fairness — ❌ CONFIRMED

Gate at `benchmark.py:466`: `is_adaptive = hasattr(planner, "should_replan") and hasattr(planner, "replan")`

- A\*, Theta\*, DWA, MPPI: `replan` absent → `is_adaptive=False` → no replanning → stuck at 10 → break
- D\*Lite, AD\*: inherited from AdaptiveAStarPlanner → `is_adaptive=True` → full replan infra

### B3. Interdiction Bias — ❌ CONFIRMED

`urban.py:935-1007`: `_init_forced_interdictions()` uses `cfg.interdiction_reference_planner` (default `theta_star`) to compute reference path. Cut points at 30% and 65% of reference path.

### B4. Non-Functional Features — ❌ CONFIRMED

| Feature | Schema location | Runtime effect in `cli/benchmark.py`? |
|---|---|---|
| `comms_dropout_prob` | `schema.py` extra dict, `spec.py:86` | ❌ Label only (benchmark.py:424-426) |
| `comms_latency_steps` | `spec.py:87` | ❌ Not implemented anywhere |
| `solvability_cert_ok` | `schema.py:156` | ❌ Never set to True (default False) |
| `forced_replan_ok` | `schema.py:157` | ❌ Stub returns True always (`solvability.py:237`) |

---

## C. Patch Series Plan

### Commit 1: Planner naming honesty (TASK 1)
**Files changed:**
- `src/uavbench/planners/dstar_lite.py` — rename class + docstring
- `src/uavbench/planners/ad_star.py` — rename class + docstring
- `src/uavbench/planners/dwa.py` — rename class + docstring
- `src/uavbench/planners/mppi.py` — update docstring
- `src/uavbench/planners/__init__.py` — update registry keys + imports
- `tests/test_planner_naming_honesty.py` — new test

### Commit 2: Replanning fairness (TASK 2)
**Files changed:**
- `src/uavbench/cli/benchmark.py` — harness-level replanning for all planners
- `tests/test_replanning_fairness.py` — new test

### Commit 3: Interdiction fairness (TASK 3)
**Files changed:**
- `src/uavbench/envs/urban.py` — chokepoint-based interdiction
- `src/uavbench/scenarios/schema.py` — remove `InterdictionReferencePlanner`
- `tests/test_interdiction_fairness.py` — new test

### Commit 4: Constraint update latency (TASK 4)
**Files changed:**
- `src/uavbench/scenarios/schema.py` — add `constraint_latency_steps`
- `src/uavbench/envs/urban.py` — FIFO buffer for constraint layers
- `tests/test_constraint_latency.py` — new test

### Commit 5: Real comms dropout (TASK 5)
**Files changed:**
- `src/uavbench/scenarios/schema.py` — add `comms_dropout_prob` to schema proper
- `src/uavbench/cli/benchmark.py` — skip replan on dropout, serve stale state
- `tests/test_comms_dropout.py` — new test

### Commit 6: GNSS interference (TASK 6)
**Files changed:**
- `src/uavbench/scenarios/schema.py` — add `gnss_mode`, `gnss_denial_zones`
- `src/uavbench/envs/urban.py` — observation corruption
- `tests/test_gnss_interference.py` — new test

### Commit 7: Scenario YAMLs + paper alignment (TASK 7)
**Files changed:**
- All 9 scenario YAMLs — add new fields
- `PAPER_METHODOLOGY.md` — new file
