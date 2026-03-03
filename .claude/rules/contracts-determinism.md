# Determinism Contract (DC)

## DC-1: Single RNG
reset(seed=s) creates ONE numpy.random.default_rng(seed).
ALL dynamics, ALL random ops use THIS rng. No module-level np.random.

## DC-2: Bit-Identical Replay
Same (scenario_id, planner_id, seed) → identical:
- event log (JSON byte-for-byte)
- trajectory (np.array_equal)
- metrics
- (optional) rendered frame hashes

## Tests (contract_test_determinism.py)
- 2 runs same params → compare all outputs
- Write hashes to outputs/v2/determinism_hashes.json
