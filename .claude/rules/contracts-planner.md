# Planner↔Env Contract (PC)

## 5 Planners (NO MPPI, NO Theta*)
1. A* — static baseline, grid-constrained, never replans, ignores cost_map
2. Periodic Replan — replans every N steps, risk-averse (α=5.0)
3. Aggressive Replan — replans when obstacle mask changes, risk-tolerant (β=0.5)
4. Incremental A* (was D*Lite) — incremental graph repair, moderate risk (γ=2.0)
5. APF — Artificial Potential Field, reactive, risk-modulated repulsion (δ=3.0)

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

## PC-4: Incremental A* (was D*Lite) Documentation
IMPORTANT: Incremental A* performs worse than full-replan planners in fire
scenarios because expanding obstacles cause mass cell changes. This is
EXPECTED BEHAVIOR based on literature. Document in paper, do NOT treat as bug.
Registry alias "dstar_lite" preserved for backward compat.
