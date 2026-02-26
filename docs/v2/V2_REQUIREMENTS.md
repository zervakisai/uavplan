# UAVBench v2 — Requirements Specification

All requirements use RFC 2119 keywords (SHALL, SHOULD, MAY).
Each requirement has a unique ID, acceptance criterion, and traceability to contracts and tests.

---

## 1. Contract Requirements (Architectural Invariants)

### DC — Determinism Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| DC-1 | `reset(seed=s)` SHALL initialize ALL RNG from ONE `np.random.default_rng(seed)` source | `rng.spawn()` used for all stochastic components; no other RNG constructors in src/uavbench2/ | `contract_test_determinism.py` | 2 |
| DC-2 | Same `(scenario_id, planner_id, seed)` SHALL produce bit-identical event log, trajectory, metrics, and frame hashes | Two runs with identical inputs produce identical SHA-256 hashes of serialized event log, trajectory array, and metrics dict | `contract_test_determinism.py` | 2 |

### FC — Fairness Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| FC-1 | Forced interdictions SHALL be placed on BFS reference corridor, NOT on any planner's actual path | Interdiction cells computed from BFS shortest path on static grid before any planner runs; verified planner-agnostic via two different planners yielding identical interdiction coordinates | `contract_test_fairness.py` | 5 |
| FC-2 | If latency/dropout enabled, all planners SHALL receive equivalent degraded observation snapshots | For same `(scenario, seed)`, `get_dynamic_state()` returns byte-identical arrays regardless of planner identity at each step | `contract_test_fairness.py` | 5 |

### EC — Decision Record Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| EC-1 | Every rejected move SHALL log: `reject_reason` (RejectReason enum), `reject_layer` (str), `reject_cell` (tuple[int,int]), `step_idx` (int) | Info dict contains all 4 fields when `accepted_move is False`; `reject_reason` is a `RejectReason` enum member; `reject_cell` matches the attempted destination | `contract_test_decision_record.py` | 3 |
| EC-2 | Every accepted move SHALL log: `accepted_move=True`, `dynamics_step` counter (int) | Info dict contains `accepted_move is True` and `dynamics_step` integer equal to the runner's step counter | `contract_test_decision_record.py` | 3 |

### GC — Feasibility Guardrail Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| GC-1 | Guardrail SHALL attempt reachability restoration via logged relaxation depths D1→D2→D3 | Info dict includes `guardrail_depth` (int 0-3) and `relaxations` (list of dicts) describing actions taken at each depth | `contract_test_guardrail.py` | 6 |
| GC-2 | If infeasible after all depths, episode SHALL be flagged `infeasible`; exclusion rate reported | Info dict includes `feasible_after_guardrail=False`; aggregate metrics include `infeasible_rate` as float [0,1] | `contract_test_guardrail.py` | 6 |

### EV — Event Semantics Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| EV-1 | Every event SHALL contain authoritative `step_idx`, consistent across runner/env/logger/renderer | All events in `env.events` have integer `step_idx` field; value matches the runner's step counter at the logical time the event occurred; no off-by-one between env and runner | `contract_test_event_semantics.py` | 3 |

### VC — Visual Truth Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| VC-1 | If `plan_len > 1`, planned path overlay SHALL be visible (never silently absent) | Frame pixel analysis confirms path-colored pixels present when `plan_len > 1` | `contract_test_visual_truth.py` | 8 |
| VC-2 | If plan missing/stale, HUD SHALL show `NO_PLAN` or `STALE` badge + `plan_reason` | HUD text extraction confirms badge presence when `plan_len <= 1` or `plan_age_steps > 2 * replan_every_steps` | `contract_test_visual_truth.py` | 8 |
| VC-3 | Forced block lifecycle SHALL be rendered: `TRIGGERED` → `ACTIVE` → `CLEARED` (with reason) | Frame sequence at event_t1 shows lifecycle transitions; HUD badge text matches block state at each step | `contract_test_visual_truth.py` | 8 |

### MC — Mission Story Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| MC-1 | Every episode SHALL have an objective POI with a human-readable `reason` string | Info dict after reset contains `objective_poi` as (int,int) tuple and `objective_reason` as non-empty str | `contract_test_mission_story.py` | 2 |
| MC-2 | Task completion = reaching POI + spending `service_time_s` steps; completion SHALL log an event | Event log contains `task_completed` event with `task_id` after agent position == POI for `service_time_s` consecutive STAY steps | `contract_test_mission_story.py` | 2 |
| MC-3 | HUD SHALL always show: `mission_domain`, `objective_label`, `distance_to_task`, `task_progress`, `deliverable_name` | Info dict contains all 5 keys on every step with non-None values | `contract_test_mission_story.py` | 2 |
| MC-4 | Results SHALL include `termination_reason` + `objective_completed` boolean | Final info dict contains `termination_reason` (TerminationReason enum) and `objective_completed` (bool) | `contract_test_mission_story.py` | 2 |

### PC — Planner-Env Contract

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| PC-1 | Executed motion SHALL be legal under action model (4-connected grid + STAY) | Every step action produces agent displacement of exactly 0 or 1 in Manhattan distance; no teleportation | `integration_test_runner_e2e.py` | 7 |
| PC-2 | Metrics SHALL separate `planned_waypoints_len` vs `executed_steps_len` | Episode metrics dict contains both integer fields; `executed_steps_len >= planned_waypoints_len` for successful episodes | `integration_test_runner_e2e.py` | 7 |

### Cross-Cutting Contracts

| ID   | Requirement | Acceptance Criterion | Test File | Phase |
|------|------------|---------------------|-----------|-------|
| MP-1 | ONE `compute_blocking_mask(state)` function SHALL be used by both step legality and guardrail BFS | `grep -r compute_blocking_mask src/uavbench2/` shows single definition; step() and guardrail both import it; no parallel mask computation | `contract_test_mask_parity.py` | 4 |
| RS-1 | Path-progress tracking SHALL prevent replan storms (≤20% naive replans) | Over a 200-step dynamic episode, `naive_replan_count / total_replan_count <= 0.20` where naive = same position AND identical mask hash as previous replan | `contract_test_replan_storm_regression.py` | 7 |

---

## 2. Scenario Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| SC-1  | v2 SHALL support 9 government mission scenarios matching v1 parameter values | All 9 YAML configs load without validation errors; field values match v1 audit |
| SC-2  | ScenarioConfig SHALL be a frozen dataclass with validation on construction | `ScenarioConfig(invalid_field=...)` raises `ValueError`; frozen instance rejects attribute assignment |
| SC-3  | Scenario loader SHALL support both `osm` and `synthetic` map sources | `load_scenario()` succeeds for configs with `map_source: osm` and `map_source: synthetic` |
| SC-4  | Scenario registry SHALL provide filter functions by mission_type, difficulty, track | `list_scenarios_by_track("dynamic")` returns exactly 6 scenario IDs |
| SC-5  | Each scenario SHALL have `event_t1` and `event_t2` for dynamic track; None for static | Dynamic scenarios have `event_t1 < event_t2`; easy scenarios have both None |

---

## 3. Environment Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| EN-1  | `UrbanEnvV2` SHALL extend `gymnasium.Env` with `reset(seed)` and `step(action)` | Passes `gymnasium.utils.env_checker.check_env()` |
| EN-2  | Action space SHALL be `Discrete(5)`: UP(0), DOWN(1), LEFT(2), RIGHT(3), STAY(4) | `env.action_space == spaces.Discrete(5)` |
| EN-3  | Observation SHALL include agent position, goal position, terrain height | `env.observation_space.shape[0] >= 5` (ax, ay, gx, gy, terrain_h) |
| EN-4  | `max_steps` SHALL equal `4 * map_size` | Episode truncates at exactly step `4 * config.map_size` |
| EN-5  | `agent_xy` and `goal_xy` SHALL be stable public `(x, y)` properties | Properties return `tuple[int, int]`; values consistent with info dict |
| EN-6  | `export_planner_inputs()` SHALL return `(heightmap, no_fly, start_xy, goal_xy)` | Returns tuple of `(ndarray[H,W], ndarray[H,W], tuple, tuple)` after reset |
| EN-7  | Start and goal SHALL be in same connected component of free cells | BFS from start reaches goal on initial free_mask |
| EN-8  | Reward SHALL include step cost (-1.0), progress shaping, safety penalties, and goal bonus (+50.0) | Reward at goal step >= 49.0 (50 bonus - 1 step cost); reward at collision step <= -5.0 |
| EN-9  | `get_dynamic_state()` SHALL return dict with all dynamic layer masks/positions | Keys include `fire_mask`, `smoke_mask`, `traffic_positions`, `forced_block_mask`, `risk_cost_map` (None if layer disabled) |

---

## 4. Planner Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| PL-1  | `PlannerBase` SHALL define abstract `plan(start, goal, cost_map) → PlanResult` | `PlannerBase` is ABC; calling `plan()` on base raises `NotImplementedError` |
| PL-2  | `PlanResult` SHALL be a dataclass with `path`, `success`, `compute_time_ms`, `expansions` | All 4 fields present and typed correctly |
| PL-3  | All 6 paper planners SHALL be registered in `PLANNERS` dict | `set(PLANNERS.keys()) == {"astar", "theta_star", "periodic_replan", "aggressive_replan", "dstar_lite", "mppi_grid"}` |
| PL-4  | `should_replan(pos, path, dyn_state, step) → (bool, str)` SHALL be overridable | Default returns `(False, "")` for one-shot planners; replanning planners return `(True, reason)` |
| PL-5  | `update(dyn_state)` SHALL accept dynamic state for incremental planners | D* Lite uses `update()` to track obstacle changes; A* ignores it |
| PL-6  | Paths SHALL be `list[tuple[int, int]]` in (x, y) order, inclusive of start and goal | `path[0] == start` and `path[-1] == goal` for successful plans |

---

## 5. Metrics Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| ME-1  | Episode metrics SHALL include: `success`, `termination_reason`, `path_length`, `planning_time_ms`, `replans` | All fields present in episode metrics dict |
| ME-2  | Aggregate metrics SHALL include: `success_rate`, `collision_rate`, `path_length_mean`, `path_length_ci_lower/upper` | Aggregate over 3+ episodes produces all fields as floats |
| ME-3  | Safety metrics SHALL count: `nfz_violations`, `collision_count`, `fire_exposure`, `smoke_exposure` | Non-negative integers/floats per episode |
| ME-4  | `planned_waypoints_len` and `executed_steps_len` SHALL be separate metrics (PC-2) | Both present; `executed_steps_len` counts env.step() calls; `planned_waypoints_len` counts planner path nodes |

---

## 6. Runner & CLI Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| RU-1  | Exactly ONE runner at `benchmark/runner.py` | Single file; no `runner2.py` or `benchmark_runner.py` |
| RU-2  | Exactly ONE CLI at `cli/benchmark.py` | Single entry point; `python -m uavbench2` works |
| RU-3  | Runner SHALL orchestrate: scenario load → env reset → plan → step loop → metrics | Full episode completes without error for `(gov_civil_protection_easy, astar, seed=42)` |
| RU-4  | Runner SHALL own authoritative `step_idx` (EV-1) | Runner increments counter; passes to env/logger; no other counter exists |
| RU-5  | CLI SHALL accept `--scenarios`, `--planners`, `--trials`, `--seed-base`, `--output-dir` | `python -m uavbench2 run --help` shows all flags |

---

## 7. Visualization Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| VZ-1  | Renderer SHALL support `paper_min` and `ops_full` modes | `OperationalRenderer(mode="paper_min")` and `mode="ops_full"` both construct without error |
| VZ-2  | Layer z-order SHALL follow the 12-layer stack (see VISUAL_TRUTH_SPEC.md) | No layer overwrites a higher-priority layer |
| VZ-3  | GIF export SHALL be deterministic (same inputs → same file bytes) | Two GIF exports with identical episode data produce identical file hashes |

---

## 8. Guardrail Requirements

| ID    | Requirement | Acceptance Criterion |
|-------|------------|---------------------|
| GR-1  | Guardrail SHALL use `compute_blocking_mask()` for reachability checks (MP-1) | Same function used by step legality |
| GR-2  | Depth 1: clear forced blocks; Depth 2: shrink NFZ + remove closures; Depth 3: emergency corridor | Each depth logged in `relaxations[]` with cells freed/removed |
| GR-3  | Topology change counter SHALL skip BFS when unchanged | `guardrail_bfs_skips > 0` after 100 steps with no dynamics changes |
