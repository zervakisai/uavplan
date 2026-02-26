# UAVBench v2 — Clean-Room Rewrite

## Project
UAV navigation benchmark framework. v2 is a **from-scratch rewrite** under `src/uavbench2/`.
Motion planners navigate grid-based urban environments with dynamic obstacles (fire, traffic, NFZs).
The benchmark measures planner performance under identical, reproducible conditions.

## NON-NEGOTIABLE RULES

1. **Clean-room**: ALL v2 code goes in `src/uavbench2/`. NEVER modify `src/uavbench/` (v1). You may READ v1 to extract requirements only. No copy-paste.
2. **Branch**: Work ONLY on `rebuild-v2-cleanroom`. Never touch main/stable.
3. **Single pipeline**: Exactly ONE runner (`benchmark/runner.py`) and ONE CLI (`cli/benchmark.py`). No forks, no duplicates, no dead modules.
4. **Spec-first**: Write docs and tests BEFORE implementation code. Each phase has a gate.
5. **Skeptical**: No "first", no "guarantee", no fabricated citations. Use `[CITE]` placeholders in docs.
6. **Dual-use safety**: No weaponization, targeting, tactics. Only benchmark integrity: navigation, safety constraints, fairness, reproducibility.
7. **Determinism**: Same `(scenario_id, planner_id, seed)` → bit-identical outputs. ONE RNG source.

## Architecture

```
src/uavbench2/
  cli/benchmark.py              # uavbench2 CLI entry point
  scenarios/                    # schema, loader, registry, configs/
  missions/                     # schema, engine, policies (mission objective + task queue)
  envs/                         # base, urban (UrbanEnvV2)
  dynamics/                     # fire_ca, traffic, restriction_zones, interaction_engine,
                                # moving_target, intruder, population_risk
  planners/                     # base, astar, theta_star, periodic_replan,
                                # aggressive_replan, dstar_lite, mppi_grid
  guardrail/feasibility.py      # multi-depth relaxation
  benchmark/                    # runner, determinism, fairness
  metrics/                      # schema, compute
  visualization/                # renderer, overlays, hud
```

## Key Commands
```bash
git checkout -b rebuild-v2-cleanroom
pip install -e .
pytest -q tests/v2                          # run all v2 tests
python -m uavbench2 run --seed 42           # single run
python scripts/run_v2_paper_n30.sh          # full paper suite
python scripts/export_v2_artifacts.py       # export evidence
```

## Test Strategy
- Contract tests enforce architectural invariants (determinism, fairness, event semantics, etc.)
- Unit tests for pure functions (fire CA, BFS, metrics)
- Integration test: full episode end-to-end
- Run `pytest tests/v2/contract_test_*.py` after every structural change

## Code Style
- Python 3.10+, type hints on all public functions
- Dataclasses or Pydantic for schemas — no raw dicts crossing module boundaries
- Every enum gets its own class (e.g., `RejectReason`, `TerminationReason`)
- Pure functions where possible; side effects only in runner and CLI
- Docstrings on all public classes/methods — include contract references (e.g., "Enforces DC-1")

## Phased Execution — Reference `docs/v2/PHASE_GATES.md`
Do NOT skip phases. Each phase has a gate test. Read `@docs/v2/PHASE_GATES.md` before starting any phase.

## Decisions Log
All assumptions and design decisions go in `docs/v2/V2_DECISIONS.md`. If a question is blocking, ask. Otherwise assume safest default and log it.
