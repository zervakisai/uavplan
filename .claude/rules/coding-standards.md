# Coding Standards

- ONE runner (benchmark/runner.py), ONE CLI (cli/benchmark.py)
- ONE compute_blocking_mask(state) used by env.step AND guardrail BFS
- No module-level np.random. Only rng instance from reset().
- Structured event dicts, not prints
- TDD: test first, then implement
- All tests under tests/
- Default seed=42 in tests
- No mocks for core logic
- NO wind parameters anywhere. If found → delete.
- NO MPPI module. If found → delete.
