# Historical Bugs — Regression Tests Exist

## BUG-1: Theta* Paradox (RESOLVED — Theta* removed)
Static any-angle planner beating adaptive in dynamic scenarios.
Root cause: any-angle paths diverge from A* corridor on large grids,
bypassing forced-block interdictions entirely (deviations up to 41 cells).
Resolution: Theta* removed from planner set. Single static baseline (A*)
is sufficient for adaptive-vs-static comparison.

## BUG-2: Civil Protection Hard = 0/150 (FIXED)
All planners × all seeds = 0% success due to wind + excessive obstacles.
Fix: Feasibility pre-check + calibration protocol (CC-1..4). No wind. Reduced knobs.

## BUG-3: MPPI Dead Code (FIXED)
Registered but never executed. Removed entirely.

## BUG-4: Wind Complexity (FIXED)
Removed. Isotropic fire CA (FD-2). 8-neighbor Moore neighborhood.

## BUG-5: Mask Parity (FIXED)
ONE compute_blocking_mask() in blocking.py used everywhere (MP-1).

## BUG-6: Step Counter Drift (FIXED)
Runner owns step_idx, passes to all components (EV-1).

## BUG-7: Two-Leg POI Routing Bypass (FIXED)
Static planners succeeded on dynamic scenarios because two-leg mission
routing (start→POI→goal) bypassed forced blocks on start→goal corridor.
Fix: (a) POI snapped to corridor midpoint so agent visits POI on the
start→goal path, (b) initial plan always start→goal, (c) second plan()
call at POI completion removed for static planners (only triggers if
current path doesn't end at goal).
