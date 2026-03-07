# Fairness Contract (FC) — REVISED for v2

## FC-1: Corridor-Aligned Interdictions
Interdictions placed on A* reference corridor, planner-agnostic.
Implementation: PHYSICAL INTERDICTIONS
- Compute A* shortest path (reference corridor)
- Fire corridor closures (all scenarios): fire ignited on corridor cells
  via fire_ca.py, creating a physical barrier (extended burnout = permanent)
- Vehicle roadblocks (piraeus, additional): traffic vehicles positioned on
  corridor cells via traffic.py, blocking with physical vehicle occupancy

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
