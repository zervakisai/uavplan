# Coding Standards

- ONE runner (benchmark/runner.py), ONE CLI (cli/benchmark.py)
- ONE compute_blocking_mask(state) used by env.step AND guardrail BFS
- ONE compute_risk_cost_map(state) for planner cost weighting (MP-1)
- No module-level np.random. Only rng instance from reset().
- Structured event dicts, not prints
- TDD: test first, then implement
- All tests under tests/
- Default seed=42 in tests
- No mocks for core logic
- Wind is OPTIONAL (FD-5b): wind_speed=0 → backward compat. No removal.
- NO MPPI module. If found → delete.
- D*Lite renamed to IncrementalAStar. "dstar_lite" is backward compat alias.
