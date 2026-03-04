# UAVBench — Traceability Matrix

> Req → Module → Test → Evidence

Every requirement traces from specification to code, test, and evidence artifact.
"Phase" indicates when the module is implemented; tests may be written earlier (spec-first).

---

## 1. Contract Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence Artifact | Phase |
|--------|-------------------|------------------|-------------|-------------------|-------|
| DC-1 | ONE RNG source via `reset(seed)` | `envs/urban.py` | `contract_test_determinism.py` | `outputs/rng_audit.json` | 2 |
| DC-2 | Bit-identical outputs for same inputs | `benchmark/runner.py`, `benchmark/determinism.py` | `contract_test_determinism.py` | `outputs/determinism_hashes.json` | 2 |
| FC-1 | Interdictions on BFS corridor, not planner path | `dynamics/forced_block.py` | `contract_test_fairness.py` | `outputs/fairness_audit.json` | 5 |
| FC-2 | Equivalent degraded snapshots across planners | `envs/urban.py` | `contract_test_fairness.py` | `outputs/fairness_audit.json` | 5 |
| EC-1 | Rejected moves log reason/layer/cell/step | `envs/urban.py` | `contract_test_decision_record.py` | `outputs/decision_record_sample.json` | 3 |
| EC-2 | Accepted moves log move_accepted + dynamics counter | `envs/urban.py` | `contract_test_decision_record.py` | `outputs/decision_record_sample.json` | 3 |
| GC-1 | Multi-depth relaxation with logging | `guardrail/feasibility.py` | `contract_test_guardrail.py` | `outputs/guardrail_audit.json` | 6 |
| GC-2 | Infeasible episodes flagged + exclusion rate | `guardrail/feasibility.py`, `metrics/compute.py` | `contract_test_guardrail.py` | `outputs/guardrail_audit.json` | 6 |
| EV-1 | Authoritative step_idx on every event | `benchmark/runner.py`, `envs/urban.py` | `contract_test_event_semantics.py` | `outputs/event_semantics_audit.json` | 3 |
| VC-1 | Planned path visible when plan_len > 1 | `visualization/renderer.py`, `visualization/overlays.py` | `contract_test_visual_truth.py` | `outputs/viz_frame_checks.json` | 8 |
| VC-2 | NO_PLAN/STALE badge when plan missing/stale | `visualization/hud.py` | `contract_test_visual_truth.py` | `outputs/viz_frame_checks.json` | 8 |
| VC-3 | Forced block lifecycle rendered on HUD | `visualization/hud.py`, `visualization/overlays.py` | `contract_test_visual_truth.py` | `outputs/viz_frame_checks.json` | 8 |
| MC-1 | Objective POI with human-readable reason | `missions/engine.py`, `missions/schema.py` | `contract_test_mission_story.py` | `outputs/mission_story_sample.json` | 2 |
| MC-2 | Task completion = reach + service_time | `missions/engine.py` | `contract_test_mission_story.py` | `outputs/mission_story_sample.json` | 2 |
| MC-3 | HUD shows mission_domain, objective_label, etc. | `visualization/hud.py`, `envs/urban.py` | `contract_test_mission_story.py` | `outputs/mission_story_sample.json` | 2 |
| MC-4 | termination_reason + objective_completed in results | `envs/urban.py`, `metrics/schema.py` | `contract_test_mission_story.py` | `outputs/mission_story_sample.json` | 2 |
| PC-1 | Motion legal under 4-connected + STAY | `envs/urban.py`, `benchmark/runner.py` | `integration_test_runner_e2e.py` | `outputs/e2e_episode.json` | 7 |
| PC-2 | planned_waypoints_len vs executed_steps_len | `metrics/schema.py`, `metrics/compute.py` | `integration_test_runner_e2e.py` | `outputs/e2e_episode.json` | 7 |
| MP-1 | ONE compute_blocking_mask for step + guardrail | `blocking.py` | `contract_test_mask_parity.py` | `outputs/mask_parity_audit.json` | 4 |
| RS-1 | Replan storm prevention (<=20% naive) | `planners/base.py`, `benchmark/runner.py` | `contract_test_replan_storm_regression.py` | `outputs/replan_storm_audit.json` | 7 |

---

## 2. Scenario Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| SC-1 | 9 gov scenarios load without errors | `scenarios/configs/*.yaml`, `scenarios/loader.py` | `unit_test_scenario_loader.py` | — |
| SC-2 | ScenarioConfig frozen + validated | `scenarios/schema.py` | `unit_test_scenario_schema.py` | — |
| SC-3 | osm + synthetic map sources supported | `scenarios/loader.py` | `unit_test_scenario_loader.py` | — |
| SC-4 | Registry filter by mission/difficulty/track | `scenarios/registry.py` | `unit_test_scenario_registry.py` | — |
| SC-5 | event_t1/t2 for dynamic; None for static | `scenarios/schema.py` | `unit_test_scenario_schema.py` | — |

---

## 3. Environment Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| EN-1 | gymnasium.Env with reset/step | `envs/urban.py`, `envs/base.py` | `unit_test_env_api.py` | — |
| EN-2 | Discrete(5) action space | `envs/urban.py` | `unit_test_env_api.py` | — |
| EN-3 | Observation includes agent/goal/terrain | `envs/urban.py` | `unit_test_env_api.py` | — |
| EN-4 | max_steps = 4 * map_size | `envs/urban.py` | `unit_test_env_api.py` | — |
| EN-5 | agent_xy/goal_xy stable properties | `envs/urban.py` | `unit_test_env_api.py` | — |
| EN-6 | export_planner_inputs() returns correct tuple | `envs/urban.py` | `unit_test_env_api.py` | — |
| EN-7 | Start and goal in same connected component | `envs/urban.py` | `unit_test_env_connectivity.py` | — |
| EN-8 | Reward structure (step cost, bonuses, penalties) | `envs/urban.py` | `unit_test_env_reward.py` | — |
| EN-9 | get_dynamic_state() returns all layer masks | `envs/urban.py` | `unit_test_env_api.py` | — |

---

## 4. Planner Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| PL-1 | PlannerBase ABC with plan() | `planners/base.py` | `unit_test_planner_base.py` | — |
| PL-2 | PlanResult dataclass with 4 fields | `planners/base.py` | `unit_test_planner_base.py` | — |
| PL-3 | 6 planners in PLANNERS registry | `planners/__init__.py` | `unit_test_planner_registry.py` | — |
| PL-4 | should_replan() overridable | `planners/base.py` | `unit_test_planner_base.py` | — |
| PL-5 | update() for incremental planners | `planners/dstar_lite.py` | `unit_test_dstar_lite_api.py` | — |
| PL-6 | Paths as list[(x,y)] inclusive start+goal | `planners/*.py` | `unit_test_planner_paths.py` | — |

---

## 5. Metrics Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| ME-1 | Episode metrics: success, reason, path_len, etc. | `metrics/schema.py`, `metrics/compute.py` | `unit_test_metrics_schema.py` | — |
| ME-2 | Aggregate metrics with CI | `metrics/compute.py` | `unit_test_metrics_aggregate.py` | — |
| ME-3 | Safety metrics: nfz, collisions, fire, smoke | `metrics/compute.py` | `unit_test_metrics_safety.py` | — |
| ME-4 | planned vs executed step counts separate | `metrics/schema.py` | `unit_test_metrics_schema.py` | — |

---

## 6. Runner & CLI Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| RU-1 | Single runner at benchmark/runner.py | `benchmark/runner.py` | `integration_test_runner_e2e.py` | `grep -c runner src/uavbench/benchmark/` |
| RU-2 | Single CLI at cli/benchmark.py | `cli/benchmark.py` | `integration_test_cli.py` | `python -m uavbench --help` |
| RU-3 | Runner orchestrates full episode | `benchmark/runner.py` | `integration_test_runner_e2e.py` | `outputs/e2e_episode.json` |
| RU-4 | Runner owns authoritative step_idx | `benchmark/runner.py` | `contract_test_event_semantics.py` | `outputs/event_semantics_audit.json` |
| RU-5 | CLI accepts --scenarios, --planners, etc. | `cli/benchmark.py` | `integration_test_cli.py` | `python -m uavbench run --help` |

---

## 7. Visualization Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| VZ-1 | paper_min and ops_full modes | `visualization/renderer.py` | `unit_test_renderer_modes.py` | — |
| VZ-2 | 12-layer z-order stack | `visualization/renderer.py`, `visualization/overlays.py` | `contract_test_visual_truth.py` | `outputs/viz_manifest.csv` |
| VZ-3 | Deterministic GIF export | `visualization/renderer.py` | `contract_test_visual_truth.py` | `outputs/viz_frame_checks.json` |

---

## 8. Guardrail Requirements

| Req ID | Requirement Summary | Source Module(s) | Test File(s) | Evidence |
|--------|-------------------|------------------|-------------|----------|
| GR-1 | Uses compute_blocking_mask() (MP-1) | `guardrail/feasibility.py`, `blocking.py` | `contract_test_mask_parity.py` | `outputs/mask_parity_audit.json` |
| GR-2 | D1→D2→D3 depths logged | `guardrail/feasibility.py` | `contract_test_guardrail.py` | `outputs/guardrail_audit.json` |
| GR-3 | Topology counter skips BFS when unchanged | `guardrail/feasibility.py` | `unit_test_guardrail_topology.py` | — |

---

## Evidence Artifact Inventory

All evidence artifacts live under `outputs/`. They are regenerated by `scripts/export_artifacts.py`.

| Artifact | Contents | Generated By |
|----------|---------|-------------|
| `determinism_hashes.json` | SHA-256 of event log, trajectory, metrics for N=2 identical runs | `contract_test_determinism.py` |
| `rng_audit.json` | RNG spawn tree verification (no independent constructors) | `contract_test_determinism.py` |
| `fairness_audit.json` | BFS corridor cells vs planner paths; interdiction placement proof | `contract_test_fairness.py` |
| `decision_record_sample.json` | Sample episode info dicts showing reject/accept fields | `contract_test_decision_record.py` |
| `guardrail_audit.json` | Relaxation depth logs for blocked episodes | `contract_test_guardrail.py` |
| `event_semantics_audit.json` | step_idx consistency across runner/env/events | `contract_test_event_semantics.py` |
| `mask_parity_audit.json` | step mask == guardrail mask at every step | `contract_test_mask_parity.py` |
| `replan_storm_audit.json` | Naive replan ratio per planner per scenario | `contract_test_replan_storm_regression.py` |
| `mission_story_sample.json` | POI, reason, service_time, completion events | `contract_test_mission_story.py` |
| `viz_manifest.csv` | Frame count, file size, hash per GIF | `contract_test_visual_truth.py` |
| `viz_frame_checks.json` | Per-frame pixel/HUD assertions | `contract_test_visual_truth.py` |
| `e2e_episode.json` | Full episode trace (trajectory, events, metrics) | `integration_test_runner_e2e.py` |
| `repro_manifest.json` | Single-command reproduction proof | `scripts/export_artifacts.py` |
