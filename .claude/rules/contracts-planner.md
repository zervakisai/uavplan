# Planner↔Env Contract (PC)

## 5 Planners (NO MPPI, NO Theta*)
1. A* — static baseline, grid-constrained, never replans
2. Periodic Replan — replans every N steps
3. Aggressive Replan — replans when obstacle mask changes near path
4. D* Lite — incremental graph repair
5. APF — Artificial Potential Field, reactive (replans every step)

Theta* REMOVED: any-angle paths diverge from A* corridor on large grids,
making corridor interdiction unreliable. Single static baseline (A*)
is sufficient for the paper's adaptive-vs-static comparison.

## PC-1: Legal Motion
Executed motion = 4-connected grid.

## PC-2: Separate Metrics
planned_waypoints_len vs executed_steps_len (always separate).

## PC-3: Path-Progress Tracking (fixes replan storms)
Track distance_to_goal per step.
If distance hasn't decreased in N steps AND replan triggered → SUPPRESS.
Log: replan_triggered, replan_suppressed, replan_reason.

## PC-4: D*Lite Documentation
IMPORTANT: D*Lite performs worse than full-replan planners in fire scenarios
because expanding obstacles cause mass cell changes. This is EXPECTED BEHAVIOR
based on literature. Document in paper, do NOT treat as bug.
