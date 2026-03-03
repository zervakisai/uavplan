# Sanity Check Contract (SC) — Detects Benchmark Bugs via Results

## SC-1: Adaptive > Static in Dynamic Scenarios
In ANY scenario with expanding dynamic obstacles (fire CA):
- best_adaptive_success MUST be ≥ best_static_success
- If static > ALL adaptive → SANITY_VIOLATION → investigate

## SC-2: Difficulty Ordering
For each planner: success(Medium) ≥ success(Hard)
Violated → DIFFICULTY_ORDERING_VIOLATION

## SC-3: Cross-Domain Consistency
Maritime (moving obstacles): adaptive MUST beat static
Fire scenarios (expanding obstacles): adaptive SHOULD beat static
If fire shows reversed ranking → likely fairness bug (FC violation)

## SC-4: D*Lite Position
D*Lite SHOULD perform between static and adaptive in fire scenarios.
It's incremental (advantage over static) but struggles with mass changes
(disadvantage vs full replan). This is EXPECTED, not a bug.
D*Lite ≥ A* in all scenarios (otherwise implementation bug).

## Tests (contract_test_sanity.py)
- 10-seed mini run, all planners × all scenarios
- Assert SC-1: no static beats all adaptive in fire
- Assert SC-2: difficulty ordering holds
- Assert SC-4: D*Lite ≥ A* everywhere
