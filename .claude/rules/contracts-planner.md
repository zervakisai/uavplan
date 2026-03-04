# Planner↔Env Contract (PC)

## 6 Planners (NO MPPI)
1. A* — static baseline, grid-constrained, never replans
2. Theta* — static any-angle, line-of-sight shortcuts, never replans
   IMPORTANT: Theta* path EXPANDED to grid steps for execution
3. Periodic Replan — replans every N steps
4. Aggressive Replan — replans when obstacle mask changes near path
5. D* Lite — incremental graph repair
6. APF — Artificial Potential Field, reactive (replans every step)

## PC-1: Legal Motion
Executed motion = 4-connected grid. If planner produces any-angle path →
explicit expansion to grid steps.

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
