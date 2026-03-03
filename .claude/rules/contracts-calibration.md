# Calibration Contract (CC) — Prevents Unsolvable Scenarios

## CC-1: Feasibility Pre-Check
Before each episode:
1. Simulate fire + dynamics forward for horizon steps (no planner)
2. At each timestep, BFS from start to goal on blocking mask
3. Record: first_infeasible_step (or "always_feasible")
4. Episode is FEASIBLE if path exists for enough steps to reach goal

## CC-2: Difficulty Thresholds
Over 30 seeds per scenario×difficulty:
- "Medium": feasibility_rate ≥ 0.50
- "Hard": feasibility_rate ≥ 0.15
Scenario FAILS calibration if below threshold → recalibrate knobs.

## CC-3: Civil Protection Recalibration
v1 had: 5 ignitions, 4 NFZ, buffer 5, wind 0.7 → 0/150 (unsolvable)
v2 changes: NO wind (isotropic), ignitions TBD, NFZ TBD, buffer TBD
Calibrate knobs until CC-2 passes. Record final values in config.

## CC-4: Reporting
Results MUST show: feasibility_rate alongside success_rate.
Infeasible episodes EXCLUDED from planner comparison.
If feasibility_rate < threshold → MISCALIBRATED flag.

## Tests (contract_test_calibration.py)
- For each scenario config, feasibility pre-check on 10 seeds
- Assert thresholds met
- Assert infeasible episodes flagged
