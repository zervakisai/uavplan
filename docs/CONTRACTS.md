# UAVBench — Canonical Contracts

These contracts are the source of truth. Every contract maps to at least one test file.
When implementing, reference contracts by ID (e.g., "Enforces DC-1").

---

## DC — Determinism Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| DC-1 | `reset(seed=s)` initializes ALL RNG from ONE source deterministically | `contract_test_determinism.py` |
| DC-2 | Same `(scenario_id, planner_id, seed)` → bit-identical event log, trajectory, metrics, frame hashes | `contract_test_determinism.py` |

## FC — Fairness Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| FC-1 | Forced interdictions placed on BFS reference corridor, NOT on any planner's actual path | `contract_test_fairness.py` |
| FC-2 | If latency/dropout enabled, all planners receive equivalent degraded observation snapshots | `contract_test_fairness.py` |

## EC — Decision Record Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| EC-1 | Every rejected move logs: `reject_reason(enum)`, `reject_layer`, `reject_cell`, `step_idx` | `contract_test_decision_record.py` |
| EC-2 | Every accepted move logs: `move_accepted=True`, dynamics step counter | `contract_test_decision_record.py` |

## GC — Feasibility Guardrail Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| GC-1 | Guardrail attempts to restore reachability via logged relaxation depths; no guarantee of success | `contract_test_guardrail.py` |
| GC-2 | If infeasible after all depths → episode flagged `infeasible`; exclusion rate reported in metrics | `contract_test_guardrail.py` |

## EV — Event Semantics Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| EV-1 | Every event contains authoritative `step_idx`, consistent across runner/env/logger/renderer | `contract_test_event_semantics.py` |

## VC — Visual Truth Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| VC-1 | If `plan_len > 1` → planned path overlay MUST be visible (never silently absent) | `contract_test_visual_truth.py` |
| VC-2 | If plan missing/stale → HUD shows `NO_PLAN` or `STALE` badge + `plan_reason` | `contract_test_visual_truth.py` |
| VC-3 | (Removed — interdictions are now physical fire/traffic events visible in existing overlays) | — |

## MC — Mission Story Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| MC-1 | Every episode has an objective POI with a human-readable `reason` string | `contract_test_mission_story.py` |
| MC-2 | Task completion = reaching POI + spending `service_time_s`; completion logs an event | `contract_test_mission_story.py` |
| MC-3 | HUD always shows: `mission_domain`, `objective_label`, `distance_to_task`, `task_progress`, `deliverable_name` | `contract_test_mission_story.py` |
| MC-4 | Results include `termination_reason` + `objective_completed` boolean | `contract_test_mission_story.py` |

## PC — Planner↔Env Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| PC-1 | Executed motion is legal under action model (4-connected; any-angle planned → explicit grid expansion) | `integration_test_runner_e2e.py` |
| PC-2 | Metrics separate `planned_waypoints_len` vs `executed_steps_len` | `integration_test_runner_e2e.py` |

## FD — Fire Dynamics Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| FD-1 | Cell states: UNBURNED → BURNING → BURNED. BURNING → BURNED after burn_duration. BURNED is permanent. | `unit_test_fire_ca.py` |
| FD-2 | Isotropic spread: equal probability to all 8 neighbors (Moore neighborhood). NO wind parameter. | `unit_test_fire_ca.py` |
| FD-3 | Step timing: planner sees CURRENT fire → validate on CURRENT mask → move → THEN fire advances | `contract_test_fairness.py` |
| FD-4 | `fire_step(state, rng)` is a pure function. Same seed → same fire evolution. | `unit_test_fire_ca.py` |
| FD-5 | Fire CA module MUST NOT have `wind_speed`, `wind_direction`, or any directional bias parameter | `unit_test_fire_ca.py` |

## CC — Calibration Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| CC-1 | Feasibility pre-check: simulate dynamics forward, BFS at each step. Record first_infeasible_step. | `contract_test_calibration.py` |
| CC-2 | Difficulty thresholds: Medium ≥ 50% feasibility, Hard ≥ 15% feasibility over 30 seeds | `contract_test_calibration.py` |
| CC-3 | Civil Protection recalibrated: no wind, reduced knobs until CC-2 passes | `contract_test_calibration.py` |
| CC-4 | Infeasible episodes flagged and excluded from planner comparison. Feasibility rate reported. | `contract_test_calibration.py` |

## SC — Sanity Check Contract
| ID   | Requirement | Test File |
|------|------------|-----------|
| SC-1 | In fire scenarios: best_adaptive_success ≥ best_static_success. Violation → investigate. | `contract_test_sanity.py` |
| SC-2 | For each planner: success(Medium) ≥ success(Hard). Violation → difficulty ordering bug. | `contract_test_sanity.py` |
| SC-3 | (Deferred — no maritime scenarios in v2. Fire adaptive SHOULD beat static is covered by SC-1.) | — |
| SC-4 | D*Lite ≥ A* in all scenarios. If violated → implementation bug (not expected behavior). | `contract_test_sanity.py` |

## Cross-cutting
| ID    | Requirement | Test File |
|-------|------------|-----------|
| MP-1  | ONE `compute_blocking_mask(state)` used by both step legality and guardrail BFS | `contract_test_mask_parity.py` |
| RS-1  | Path-progress tracking prevents replan storms (≤20% naive replans) | `contract_test_replan_storm_regression.py` |
