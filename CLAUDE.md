# UAVBench

## Project
UAV navigation benchmark framework for reproducible planner evaluation.
Motion planners navigate grid-based urban environments with dynamic obstacles (fire, traffic, NFZs).
The benchmark measures planner performance under identical, reproducible conditions.

## Rules

1. **Single pipeline**: Exactly ONE runner (`benchmark/runner.py`) and ONE CLI (`cli/benchmark.py`). No forks, no duplicates, no dead modules.
2. **Skeptical**: No "first", no "guarantee", no fabricated citations. Use `[CITE]` placeholders in docs.
3. **Dual-use safety**: No weaponization, targeting, tactics. Only benchmark integrity: navigation, safety constraints, fairness, reproducibility.
4. **Determinism**: Same `(scenario_id, planner_id, seed)` → bit-identical outputs. ONE RNG source.

## Architecture

```
src/uavbench/
  cli/benchmark.py              # CLI entry point
  scenarios/                    # schema, loader, registry, calibration, configs/
  missions/                     # schema, engine (mission objective + task queue)
  envs/                         # base, urban (UrbanEnvV2)
  dynamics/                     # fire_ca, traffic, restriction_zones,
                                # interaction_engine, forced_block
  planners/                     # base, astar, theta_star, periodic_replan,
                                # aggressive_replan, dstar_lite, apf
  blocking.py                   # ONE compute_blocking_mask() (MP-1)
  guardrail/feasibility.py      # multi-depth relaxation
  benchmark/                    # runner, determinism, sanity_check
  metrics/                      # schema, compute
  visualization/                # renderer, overlays, hud
```

## Planners (6)
- `astar` — A* static baseline, never replans
- `theta_star` — Theta* any-angle static, never replans
- `periodic_replan` — replans every N steps
- `aggressive_replan` — replans when obstacle mask changes near path
- `dstar_lite` — incremental graph repair
- `apf` — Artificial Potential Field, reactive (replans every step)

## Key Commands
```bash
pip install -e .
pytest tests/ -q                             # run all tests
python -m uavbench run --seed 42             # single run
python scripts/run_paper_experiments.py      # full paper suite
python scripts/export_artifacts.py           # export evidence
```

## Test Strategy
- Contract tests enforce architectural invariants (determinism, fairness, event semantics, etc.)
- Unit tests for pure functions (fire CA, BFS, metrics)
- Run `pytest tests/contract_test_*.py` after every structural change

## Code Style
- Python 3.10+, type hints on all public functions
- Dataclasses for schemas — no raw dicts crossing module boundaries
- Every enum gets its own class (e.g., `RejectReason`, `TerminationReason`)
- Pure functions where possible; side effects only in runner and CLI
- Docstrings on all public classes/methods — include contract references (e.g., "Enforces DC-1")

## Contracts (11 families, 33 contracts)
DC (Determinism), FC (Fairness), EC (Events/Decisions), GC (Guardrail),
EV (Event Semantics), VC (Visual Truth), MC (Mission Story), PC (Planner),
FD (Fire Dynamics), CC (Calibration), SC (Sanity Check), MP (Mask Parity)

See `docs/CONTRACTS.md` for full specification.

## Decisions Log
All assumptions and design decisions go in `docs/DECISIONS.md`.
