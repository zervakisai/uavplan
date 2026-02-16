# UAVBench Scenario Pack (34 IDs, Paper Protocol)

This folder contains the canonical 34 scenario YAMLs used by UAVBench.

## Track Design

- `paper_track: static`
  - Designed for near-deterministic solvability and high baseline success.
  - No blocking dynamics.
  - Includes fixed start/goal anchors for reproducibility.

- `paper_track: dynamic`
  - Designed for forced replanning under operationally realistic interactions.
  - Uses deterministic interdiction scheduler (`force_replan_count`, `event_t1`, `event_t2`).
  - Combines dynamic NFZ, traffic closures, and mission-aware hazard coupling.

## Determinism Fields

Every scenario may include:

- `fixed_start_xy`, `fixed_goal_xy`
- `force_replan_count`
- `event_t1`, `event_t2`
- `emergency_corridor_enabled`
- `interdiction_reference_planner`
- `plan_budget_static_ms`, `plan_budget_dynamic_ms`
- `replan_every_steps`, `max_replans_per_episode`

These are consumed by `UrbanEnv` to ensure repeatable track behavior.

## Validation Rules (Implemented)

- Same 34 IDs and filenames are preserved for backward compatibility.
- Track defaults:
  - `naturalistic -> static`
  - `stress_test -> dynamic`
- Dynamic scenarios keep feasibility guardrails enabled (`emergency_corridor_enabled=true`).

## Practical Listing

```bash
python -c "from uavbench.scenarios.registry import list_scenarios; print(len(list_scenarios()))"
python -c "from uavbench.scenarios.registry import list_scenarios_by_track; print('static', len(list_scenarios_by_track('static'))); print('dynamic', len(list_scenarios_by_track('dynamic')))"
```
