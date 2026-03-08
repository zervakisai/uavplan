# Historical Bugs — Regression Tests Exist

## BUG-1: Theta* Paradox (RESOLVED — Theta* removed)
Static any-angle planner beating adaptive in dynamic scenarios.
Root cause: any-angle paths diverge from A* corridor on large grids,
bypassing corridor interdictions entirely (deviations up to 41 cells).
Resolution: Theta* removed from planner set. Single static baseline (A*)
is sufficient for adaptive-vs-static comparison.

## BUG-2: Civil Protection Hard = 0/150 (FIXED)
All planners × all seeds = 0% success due to wind + excessive obstacles.
Fix: Feasibility pre-check + calibration protocol (CC-1..4). No wind. Reduced knobs.

## BUG-3: MPPI Dead Code (FIXED)
Registered but never executed. Removed entirely.

## BUG-4: Wind Complexity (SUPERSEDED by FD-5b)
v2 removed wind due to calibration issues (BUG-2). Paper #1 re-introduces
wind as OPTIONAL with Alexandridis 2008 model. wind_speed=0 → isotropic
(backward compat). Wind-driven fire is a core paper contribution.

## BUG-5: Mask Parity (FIXED)
ONE compute_blocking_mask() in blocking.py used everywhere (MP-1).

## BUG-6: Step Counter Drift (FIXED)
Runner owns step_idx, passes to all components (EV-1).

## BUG-7: Two-Leg POI Routing Bypass (FIXED)
Static planners succeeded on dynamic scenarios because two-leg mission
routing (start→POI→goal) bypassed corridor interdictions on start→goal corridor.
Fix: (a) POI snapped to corridor midpoint so agent visits POI on the
start→goal path, (b) initial plan always start→goal, (c) second plan()
call at POI completion removed for static planners (only triggers if
current path doesn't end at goal).

## BUG-8: Fire Guarantee Burnout Gap (FIXED)
Fire guarantee targets (corridor cells that must burn) used re-ignition:
BURNED_OUT → BURNING at each step. This created a 1-step gap between
burnout and re-ignition where fire_mask returned False, allowing A* agents
to advance one cell per burnout cycle (~135 steps).
Fix: Extended burnout (999999 steps) for guarantee targets. Cells never
reach BURNED_OUT, eliminating the gap. No CA state reversal needed.

## BUG-9: POI Unreachability Deadlock (FIXED)
When fire/traffic blocked access to mission POI, agent targeted POI forever,
never switching to goal. Agent got permanently stuck with 0 replans.
Fix: POI stuck detection (30 steps without progress toward POI) → abandon
POI, switch target to goal. No free replan given (fairness preserved).
