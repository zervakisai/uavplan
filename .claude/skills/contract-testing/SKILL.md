---
name: contract-testing
description: Patterns for writing and running contract tests in UAVBench v2. Use when writing tests that verify architectural invariants like determinism, fairness, event semantics, mask parity, or mission story compliance.
---

# Contract Testing Patterns for UAVBench v2

## When to Use
- Writing any test in `tests/contract_test_*.py`
- Verifying an architectural invariant from `docs/CONTRACTS.md`
- Checking phase gate criteria

## Core Pattern: Hash Equality for Determinism
```python
import hashlib, json

def hash_episode_result(result: dict) -> str:
    """Canonical hash of episode output for determinism checks."""
    canonical = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

## Core Pattern: Mask Parity
```python
def test_mask_parity(env):
    """The blocking mask used by step() must equal the one used by guardrail BFS."""
    state = env.get_state()
    step_mask = env.compute_blocking_mask(state)
    guardrail_mask = env.guardrail.compute_blocking_mask(state)
    assert (step_mask == guardrail_mask).all(), "Mask parity violated (MP-1)"
```

## Core Pattern: Event Step Consistency
```python
def test_event_step_idx_consistent(episode_log):
    """Every event must have step_idx matching the runner's authoritative counter."""
    for event in episode_log.events:
        assert hasattr(event, 'step_idx'), f"Event missing step_idx: {event}"
        assert event.step_idx >= 0, f"Invalid step_idx: {event.step_idx}"
    # step_idx must be monotonically non-decreasing
    indices = [e.step_idx for e in episode_log.events]
    assert indices == sorted(indices), "Events not ordered by step_idx"
```

## Core Pattern: Decision Record Completeness
```python
def test_rejected_move_has_full_record(episode_log):
    """EC-1: Every rejected move must log reason, layer, cell, step."""
    for decision in episode_log.decisions:
        if not decision.move_accepted:
            assert decision.reject_reason is not None
            assert decision.reject_layer is not None
            assert decision.reject_cell is not None
            assert decision.step_idx is not None
```

## Anti-patterns to Avoid
- Testing implementation details instead of contracts
- Using random seeds in contract tests (always use fixed seeds)
- Asserting on floating-point equality without tolerance
- Testing planner quality (that's metrics, not contracts)
