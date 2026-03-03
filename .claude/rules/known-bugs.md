# Historical Bugs — Regression Tests Exist

## BUG-1: Theta* Paradox (FIXED)
Static planner beating adaptive in dynamic scenarios due to single-cell interdictions.
Fix: Area interdictions (FC-1), path-progress tracking (PC-3), frozen fire state (FD-3).
Theta* removed from planner set (only A* as static baseline).

## BUG-2: Civil Protection Hard = 0/150 (FIXED)
All planners × all seeds = 0% success due to wind + excessive obstacles.
Fix: Feasibility pre-check + calibration protocol (CC-1..4). No wind. Reduced knobs.

## BUG-3: MPPI Dead Code (FIXED)
Registered but never executed. Removed entirely. 4 planners only.

## BUG-4: Wind Complexity (FIXED)
Removed. Isotropic fire CA (FD-2). 8-neighbor Moore neighborhood.

## BUG-5: Mask Parity (FIXED)
ONE compute_blocking_mask() in blocking.py used everywhere (MP-1).

## BUG-6: Step Counter Drift (FIXED)
Runner owns step_idx, passes to all components (EV-1).
