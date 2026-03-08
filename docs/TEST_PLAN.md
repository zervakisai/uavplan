# UAVBench — Test Plan

All tests live under `tests/`. Tests are written spec-first (before implementation).
Every contract has a named test file with explicit acceptance criteria.

---

## 1. Test Categories

| Category | Purpose | Speed | CI Gate |
|----------|---------|-------|---------|
| Contract | Architectural invariant enforcement | Medium | Required |
| Unit | Pure function correctness | Fast | Required |
| Integration | Full episode end-to-end | Slow | Nightly |

---

## 2. Contract Tests

Contract tests verify the non-negotiable architectural invariants defined in `CONTRACTS.md`.
Each test file maps to one or more contract IDs.

### 2.1 `contract_test_determinism.py` — DC-1, DC-2

**Contracts tested**: DC-1 (ONE RNG source), DC-2 (bit-identical outputs)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_single_rng_source` | DC-1 | `grep` finds zero `np.random.default_rng` calls outside `reset()`; all components receive spawned child generators |
| `test_no_independent_rng_constructors` | DC-1 | No `np.random.RandomState`, `random.Random`, or bare `np.random.seed` in `src/uavbench/` |
| `test_identical_seed_identical_output` | DC-2 | Two runs with `(osm_penteli_pharma_delivery_medium, astar, seed=42)` produce SHA-256 identical event logs, trajectories, and metrics dicts |
| `test_identical_seed_identical_frames` | DC-2 | Frame hash arrays from two identical runs are element-wise equal |
| `test_different_seed_different_output` | DC-2 | Two runs with seeds 42 and 43 produce different trajectory hashes |

**Evidence**: `outputs/determinism_hashes.json`, `outputs/rng_audit.json`

---

### 2.2 `contract_test_fairness.py` — FC-1, FC-2

**Contracts tested**: FC-1 (BFS reference corridor), FC-2 (equivalent degraded snapshots)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_interdiction_on_corridor` | FC-1 | All corridor interdiction cells (fire closures or vehicle roadblocks) are on the A* shortest path computed on the static grid before any planner runs |
| `test_interdiction_planner_agnostic` | FC-1 | Same `(scenario, seed)` with `astar` vs `periodic_replan` produces identical corridor interdiction coordinates |
| `test_identical_dynamic_state_across_planners` | FC-2 | At each step, `get_dynamic_state()` returns byte-identical arrays for two different planners running the same `(scenario, seed)` |

**Evidence**: `outputs/fairness_audit.json`

---

### 2.3 `contract_test_decision_record.py` — EC-1, EC-2

**Contracts tested**: EC-1 (rejection logging), EC-2 (acceptance logging)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_rejected_move_has_all_fields` | EC-1 | When agent attempts move into a building cell, info dict contains `reject_reason` (RejectReason.BUILDING), `reject_layer` ("building"), `reject_cell` (tuple), `step_idx` (int) |
| `test_reject_reason_is_enum` | EC-1 | `reject_reason` is a `RejectReason` enum member, not a raw string |
| `test_reject_cell_matches_destination` | EC-1 | `reject_cell` equals the attempted destination `(x, y)` |
| `test_all_reject_reasons_exercised` | EC-1 | Over a hard scenario, at least `BUILDING`, `FIRE`, `TRAFFIC_BUFFER` are observed |
| `test_accepted_move_has_fields` | EC-2 | Info dict contains `accepted_move is True` and `dynamics_step` integer |
| `test_dynamics_step_matches_runner` | EC-2 | `dynamics_step` equals the runner's current step counter |

**Evidence**: `outputs/decision_record_sample.json`

---

### 2.4 `contract_test_event_semantics.py` — EV-1

**Contracts tested**: EV-1 (authoritative step_idx)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_all_events_have_step_idx` | EV-1 | Every event in `env.events` has an integer `step_idx` field |
| `test_step_idx_monotonic` | EV-1 | Event `step_idx` values are non-decreasing across the episode |
| `test_step_idx_matches_runner` | EV-1 | For events logged during step N, `event["step_idx"] == N` |
| `test_no_off_by_one` | EV-1 | Reset event has `step_idx=0`; first step event has `step_idx=1` |

**Evidence**: `outputs/event_semantics_audit.json`

---

### 2.5 `contract_test_guardrail.py` — GC-1, GC-2

**Contracts tested**: GC-1 (multi-depth relaxation), GC-2 (infeasible flagging)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_guardrail_logs_depth_and_relaxations` | GC-1 | Info dict includes `guardrail_depth` (int 0-3) and `relaxations` (list of dicts) after a topology-changing event |
| `test_depth1_clears_roadblock_vehicles` | GC-1 | After a vehicle roadblock appears on the path, guardrail at D1 clears it; `relaxations[0]` documents cells freed |
| `test_depth2_shrinks_nfz` | GC-1 | After D1 fails, D2 shrinks NFZ zones; `relaxations` includes NFZ cells removed |
| `test_depth3_emergency_corridor` | GC-1 | After D1+D2 fail, D3 opens corridor; `relaxations` includes corridor cells |
| `test_infeasible_flagged` | GC-2 | When all depths fail, `feasible_after_guardrail=False` in info dict |
| `test_infeasible_rate_in_metrics` | GC-2 | Aggregate metrics contain `infeasible_rate` as float [0,1] |

**Evidence**: `outputs/guardrail_audit.json`

---

### 2.6 `contract_test_mask_parity.py` — MP-1

**Contracts tested**: MP-1 (single blocking mask)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_single_definition` | MP-1 | `grep -r "def compute_blocking_mask" src/uavbench/` returns exactly 1 match |
| `test_step_uses_blocking_mask` | MP-1 | `env.step()` code path calls `compute_blocking_mask()` |
| `test_guardrail_uses_blocking_mask` | MP-1 | `guardrail.check()` code path calls `compute_blocking_mask()` |
| `test_masks_identical_at_every_step` | MP-1 | Over a 50-step dynamic episode, the mask used by step legality == the mask used by guardrail BFS at each step |

**Evidence**: `outputs/mask_parity_audit.json`

---

### 2.7 `contract_test_mission_story.py` — MC-1, MC-2, MC-3, MC-4

**Contracts tested**: MC-1 (objective POI), MC-2 (task completion), MC-3 (HUD fields), MC-4 (termination fields)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_objective_poi_exists` | MC-1 | After reset, info dict contains `objective_poi` as `(int, int)` and `objective_reason` as non-empty str |
| `test_objective_reason_is_human_readable` | MC-1 | `objective_reason` is >= 10 characters and contains no raw IDs |
| `test_task_completion_requires_service_time` | MC-2 | Agent at POI for fewer than `service_time_s` steps does NOT trigger completion event |
| `test_task_completion_logs_event` | MC-2 | After `service_time_s` STAY actions at POI, event log contains `task_completed` event with `task_id` |
| `test_hud_fields_present_every_step` | MC-3 | Info dict contains `mission_domain`, `objective_label`, `distance_to_task`, `task_progress`, `deliverable_name` with non-None values at every step |
| `test_termination_reason_in_final_info` | MC-4 | Final info dict has `termination_reason` (TerminationReason enum) and `objective_completed` (bool) |
| `test_successful_episode_objective_completed` | MC-4 | When agent reaches goal and completes tasks, `objective_completed is True` |

**Evidence**: `outputs/mission_story_sample.json`

---

### 2.8 `contract_test_visual_truth.py` — VC-1, VC-2

**Contracts tested**: VC-1 (path visible), VC-2 (NO_PLAN/STALE badge)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_path_visible_when_plan_exists` | VC-1 | When `plan_len > 1`, rendered frame contains path-colored pixels (cyan `#56B4E9` Okabe-Ito sky blue) |
| `test_no_silent_path_absence` | VC-1 | If plan_len drops to 0 mid-episode, frame contains NO_PLAN badge text |
| `test_no_plan_badge` | VC-2 | When `plan_len <= 1`, HUD text contains "NO PLAN" |
| `test_stale_badge` | VC-2 | When `plan_age_steps > 2 * replan_every_steps`, HUD text contains "STALE" |
| `test_plan_reason_shown` | VC-2 | HUD badge includes `plan_reason` string |

**Evidence**: `outputs/viz_manifest.csv`, `outputs/viz_frame_checks.json`

---

### 2.9 `contract_test_replan_storm_regression.py` — RS-1

**Contracts tested**: RS-1 (replan storm prevention)

**Test cases**:

| Test Name | Contract | Acceptance Criterion |
|-----------|----------|---------------------|
| `test_naive_replan_ratio` | RS-1 | Over a 200-step dynamic episode with `periodic_replan`, `naive_replan_count / total_replan_count <= 0.20` |
| `test_naive_definition` | RS-1 | Naive replan = same position AND identical blocking mask hash as previous replan |
| `test_cooldown_enforced` | RS-1 | No two replans occur within 3 steps of each other (unless forced) |

**Evidence**: `outputs/replan_storm_audit.json`

---

## 3. Unit Tests

Unit tests verify pure functions and data structures. They run fast and have no env/planner dependencies.

| Test File | Module Under Test | Test Cases |
|-----------|------------------|------------|
| `unit_test_scenario_schema.py` | `scenarios/schema.py` | ScenarioConfig construction, validation, frozen enforcement, event_t1/t2 invariants (SC-2, SC-5) |
| `unit_test_scenario_loader.py` | `scenarios/loader.py` | Load all 3 YAMLs, osm map source, invalid YAML rejection (SC-1, SC-3) |
| `unit_test_scenario_registry.py` | `scenarios/registry.py` | Filter by track/mission/difficulty, list all, count assertions (SC-4) |
| `unit_test_env_api.py` | `envs/urban.py` | Action space, obs space, reset/step signatures, info dict keys, agent_xy/goal_xy, export_planner_inputs, get_dynamic_state (EN-1..EN-6, EN-9) |
| `unit_test_env_connectivity.py` | `envs/urban.py` | Start-goal BFS reachability on initial free_mask (EN-7) |
| `unit_test_env_reward.py` | `envs/urban.py` | Step cost, goal bonus, collision penalty ranges (EN-8) |
| `unit_test_fire_ca.py` | `dynamics/fire_ca.py` | CA step determinism, spread rules, smoke generation, force_cell_state |
| `unit_test_traffic.py` | `dynamics/traffic.py` | Vehicle movement, road following, fire avoidance, occupancy mask |
| `unit_test_restriction_zones.py` | `dynamics/restriction_zones.py` | Zone activation, expansion rate, coverage limits |
| `unit_test_interaction_engine.py` | `dynamics/interaction_engine.py` | Fire-traffic coupling, closure mask generation |
| `unit_test_moving_target.py` | `dynamics/moving_target.py` | BFS path following, buffer mask |
| `unit_test_intruder.py` | `dynamics/intruder.py` | Spawn, approach, buffer mask |
| `unit_test_population_risk.py` | `dynamics/population_risk.py` | Diffusion, hazard source, non-blocking assertion |
| `unit_test_blocking_mask.py` | `blocking.py` | Layer merging order, config gating, smoke threshold >= 0.5 (MP-1 support) |
| `unit_test_planner_base.py` | `planners/base.py` | ABC enforcement, PlanResult fields, should_replan default, update default (PL-1, PL-2, PL-4, PL-5) |
| `unit_test_planner_registry.py` | `planners/__init__.py` | Exactly 5 keys in PLANNERS (PL-3) |
| `unit_test_planner_paths.py` | `planners/*.py` | Path format: list[(x,y)], start/goal inclusive (PL-6) |
| `unit_test_astar.py` | `planners/astar.py` | Optimality on known grids, max_expansions cap |
| `unit_test_dstar_lite_api.py` | `planners/dstar_lite.py` | Incremental update correctness, obstacle change handling (PL-5) |
| `unit_test_metrics_schema.py` | `metrics/schema.py` | EpisodeMetrics fields, AggregateMetrics fields, planned vs executed separation (ME-1, ME-4) |
| `unit_test_metrics_aggregate.py` | `metrics/compute.py` | CI calculation, aggregation over 3+ episodes (ME-2) |
| `unit_test_metrics_safety.py` | `metrics/compute.py` | Safety metric non-negativity (ME-3) |
| `unit_test_renderer_modes.py` | `visualization/renderer.py` | paper_min and ops_full construction (VZ-1) |
| `unit_test_guardrail_topology.py` | `guardrail/feasibility.py` | BFS skip when topology unchanged (GR-3) |
| `unit_test_mission_schema.py` | `missions/schema.py` | TaskSpec fields, TaskStatus lifecycle |
| `unit_test_mission_engine.py` | `missions/engine.py` | Task queue, injection, completion, expiration |

---

## 4. Integration Tests

Integration tests verify the full pipeline from CLI to evidence artifacts.

| Test File | Scope | Acceptance Criterion |
|-----------|-------|---------------------|
| `integration_test_runner_e2e.py` | Full episode: scenario → env → planner → metrics | Episode completes for `(osm_penteli_pharma_delivery_medium, astar, seed=42)`; metrics contain all required fields; motion is legal (PC-1); planned vs executed counts differ (PC-2) |
| `integration_test_cli.py` | CLI entry point | `python -m uavbench run --scenarios osm_penteli_pharma_delivery_medium --planners astar --trials 1 --seed-base 42 --output-dir /tmp/v2_test` exits 0 and produces JSON output |
| `integration_test_dynamic_episode.py` | Full dynamic episode | `(osm_penteli_pharma_delivery_medium, periodic_replan, seed=42)` completes; dynamics active; guardrail invoked at least once; events contain step_idx |
| `integration_test_determinism_e2e.py` | End-to-end determinism | Two full CLI runs with identical args produce identical output files (SHA-256 match) |

---

## 5. CI Configuration

### 5.1 Fast Suite (every push)

```bash
pytest tests/unit_test_*.py tests/contract_test_*.py -q --timeout=60
```

- All unit + contract tests
- Timeout: 60s per test
- Expected runtime: < 2 minutes

### 5.2 Slow Suite (nightly / pre-merge)

```bash
pytest tests/ -q --timeout=300
```

- All tests including integration
- Timeout: 300s per test
- Expected runtime: < 10 minutes

### 5.3 Determinism Suite (release gate)

```bash
pytest tests/contract_test_determinism.py tests/integration_test_determinism_e2e.py -q --timeout=600
```

- Full determinism verification
- Timeout: 600s
- Expected runtime: < 5 minutes

---

## 6. Test File Inventory

All test files, ordered by phase introduction:

| Phase | Test File | Category | Contracts |
|-------|-----------|----------|-----------|
| 2 | `contract_test_determinism.py` | Contract | DC-1, DC-2 |
| 2 | `contract_test_mission_story.py` | Contract | MC-1, MC-2, MC-3, MC-4 |
| 2 | `unit_test_scenario_schema.py` | Unit | SC-2, SC-5 |
| 2 | `unit_test_scenario_loader.py` | Unit | SC-1, SC-3 |
| 2 | `unit_test_scenario_registry.py` | Unit | SC-4 |
| 2 | `unit_test_env_api.py` | Unit | EN-1..EN-6, EN-9 |
| 2 | `unit_test_env_connectivity.py` | Unit | EN-7 |
| 2 | `unit_test_env_reward.py` | Unit | EN-8 |
| 2 | `unit_test_mission_schema.py` | Unit | — |
| 2 | `unit_test_mission_engine.py` | Unit | — |
| 3 | `contract_test_decision_record.py` | Contract | EC-1, EC-2 |
| 3 | `contract_test_event_semantics.py` | Contract | EV-1 |
| 4 | `contract_test_mask_parity.py` | Contract | MP-1 |
| 4 | `unit_test_blocking_mask.py` | Unit | MP-1 support |
| 4 | `unit_test_fire_ca.py` | Unit | — |
| 4 | `unit_test_traffic.py` | Unit | — |
| 4 | `unit_test_restriction_zones.py` | Unit | — |
| 4 | `unit_test_interaction_engine.py` | Unit | — |
| 4 | `unit_test_moving_target.py` | Unit | — |
| 4 | `unit_test_intruder.py` | Unit | — |
| 4 | `unit_test_population_risk.py` | Unit | — |
| 5 | `contract_test_fairness.py` | Contract | FC-1, FC-2 |
| 6 | `contract_test_guardrail.py` | Contract | GC-1, GC-2 |
| 6 | `unit_test_guardrail_topology.py` | Unit | GR-3 |
| 7 | `contract_test_replan_storm_regression.py` | Contract | RS-1 |
| 7 | `unit_test_planner_base.py` | Unit | PL-1, PL-2, PL-4, PL-5 |
| 7 | `unit_test_planner_registry.py` | Unit | PL-3 |
| 7 | `unit_test_planner_paths.py` | Unit | PL-6 |
| 7 | `unit_test_astar.py` | Unit | — |
| 7 | `unit_test_dstar_lite_api.py` | Unit | PL-5 |
| 7 | `unit_test_metrics_schema.py` | Unit | ME-1, ME-4 |
| 7 | `unit_test_metrics_aggregate.py` | Unit | ME-2 |
| 7 | `unit_test_metrics_safety.py` | Unit | ME-3 |
| 7 | `integration_test_runner_e2e.py` | Integration | PC-1, PC-2 |
| 8 | `contract_test_visual_truth.py` | Contract | VC-1, VC-2 |
| 8 | `unit_test_renderer_modes.py` | Unit | VZ-1 |
| 9 | `integration_test_cli.py` | Integration | RU-2, RU-5 |
| 9 | `integration_test_dynamic_episode.py` | Integration | — |
| 9 | `integration_test_determinism_e2e.py` | Integration | DC-2 |

---

## 7. Contract Coverage Summary

Every contract maps to at least one test file with explicit acceptance criteria:

| Contract | Test File | Test Count | Phase |
|----------|-----------|------------|-------|
| DC-1 | `contract_test_determinism.py` | 2 | 2 |
| DC-2 | `contract_test_determinism.py` | 3 | 2 |
| FC-1 | `contract_test_fairness.py` | 2 | 5 |
| FC-2 | `contract_test_fairness.py` | 1 | 5 |
| EC-1 | `contract_test_decision_record.py` | 4 | 3 |
| EC-2 | `contract_test_decision_record.py` | 2 | 3 |
| GC-1 | `contract_test_guardrail.py` | 4 | 6 |
| GC-2 | `contract_test_guardrail.py` | 2 | 6 |
| EV-1 | `contract_test_event_semantics.py` | 4 | 3 |
| VC-1 | `contract_test_visual_truth.py` | 2 | 8 |
| VC-2 | `contract_test_visual_truth.py` | 3 | 8 |
| MC-1 | `contract_test_mission_story.py` | 2 | 2 |
| MC-2 | `contract_test_mission_story.py` | 2 | 2 |
| MC-3 | `contract_test_mission_story.py` | 1 | 2 |
| MC-4 | `contract_test_mission_story.py` | 2 | 2 |
| PC-1 | `integration_test_runner_e2e.py` | 1 | 7 |
| PC-2 | `integration_test_runner_e2e.py` | 1 | 7 |
| MP-1 | `contract_test_mask_parity.py` | 4 | 4 |
| RS-1 | `contract_test_replan_storm_regression.py` | 3 | 7 |
