# Fairness Contract (FC) — REVISED for v2

## FC-1: Corridor-Aligned Interdictions
Interdictions placed on A* reference corridor, planner-agnostic.
Implementation: AREA INTERDICTIONS
- Compute A* shortest path (reference corridor)
- For each interdiction, block a RECTANGULAR ZONE width=3 cells
  centered on the corridor, perpendicular to corridor direction

## FC-2: Observation Equality
All planners receive identical observation at each step_idx.
If degradation enabled, all planners get same degraded snapshot.

## FC-3: Fire State Consistency
During planner computation, fire state is FROZEN.
Planner plans on TRUE current mask.
Fire advances ONLY in env.step(), AFTER planner returns action.
All planners see same fire state at same step_idx.

## FC-4: Replan Storm Tracking
Track per planner: total_replans, storm_replans (no progress after replan).
Report storm_ratio. Warn if > 0.15.

## Tests (contract_test_fairness.py)
- Assert: interdiction zone ON A* corridor
- Assert: fire state identical across planners at each step
- Assert: blocking mask identical for all planners at each step
