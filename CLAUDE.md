# UAVBench

## Project
UAV navigation benchmark framework for reproducible planner evaluation.
Motion planners navigate grid-based urban environments with dynamic obstacles (fire, traffic, NFZs).
The benchmark measures planner performance under identical, reproducible conditions.

**Paper #1: Hazard-Coupled Planning Under Uncertainty** — planners differentiated by
risk cost functions, wind-driven fire, fog of war, and triage missions.

## Rules

1. **Single pipeline**: Exactly ONE runner (`benchmark/runner.py`) and ONE CLI (`cli/benchmark.py`). No forks, no duplicates, no dead modules.
2. **Skeptical**: No "first", no "guarantee", no fabricated citations. Use `[CITE]` placeholders in docs.
3. **Dual-use safety**: No weaponization, targeting, tactics. Only benchmark integrity: navigation, safety constraints, fairness, reproducibility.
4. **Determinism**: Same `(scenario_id, planner_id, seed)` → bit-identical outputs. ONE RNG source.
5. **Backward compat**: wind_speed=0, enable_fog_of_war=False → bit-identical to v2.

## Architecture

```
src/uavbench/
  cli/benchmark.py              # CLI entry point
  scenarios/                    # schema, loader, registry, calibration, configs/
  missions/                     # schema, engine, triage (TRIAGE mission + survival model)
  envs/                         # base, urban (UrbanEnvV2)
  dynamics/                     # fire_ca (wind-driven), traffic, restriction_zones,
                                # interaction_engine, fog_of_war
  planners/                     # base, astar, periodic_replan, aggressive_replan,
                                # incremental_astar (was dstar_lite), apf
  blocking.py                   # compute_blocking_mask() + compute_risk_cost_map() (MP-1)
  guardrail/feasibility.py      # multi-depth relaxation
  benchmark/                    # runner, determinism, sanity_check
  metrics/                      # schema, compute
  visualization/                # renderer, overlays, hud
```

## Planners (5) — Per-Planner Risk Coefficients
- `astar` — A* static baseline, never replans, ignores cost_map
- `periodic_replan` — replans every N steps, risk-averse (α=5.0)
- `aggressive_replan` — replans when obstacle mask changes, risk-tolerant (β=0.5)
- `incremental_astar` — incremental graph repair, moderate risk (γ=2.0) [alias: dstar_lite]
- `apf` — Artificial Potential Field, reactive, risk-modulated repulsion (δ=3.0)

## Key Features (Paper #1)
- **Wind-driven fire** (FD-5b): Alexandridis 2008 directional modulation
- **Risk cost maps**: continuous [0,1] planner cost from fire/smoke/traffic proximity
- **Fog of war**: partial observability with stale memory outside sensor radius
- **TRIAGE mission**: multi-casualty survival decay coupled with fire proximity

## Key Commands
```bash
pip install -e .
pytest tests/ -q                             # run all tests
python -m uavbench run --seed-base 42         # single run
python scripts/run_paper_experiments.py      # full paper suite
python scripts/export_artifacts.py           # export evidence
```

## Test Strategy
- Contract tests enforce architectural invariants (determinism, fairness, event semantics, etc.)
- Unit tests for pure functions (fire CA, BFS, metrics, wind, fog, triage)
- Run `pytest tests/contract_test_*.py` after every structural change

## Code Style
- Python 3.10+, type hints on all public functions
- Dataclasses for schemas — no raw dicts crossing module boundaries
- Every enum gets its own class (e.g., `RejectReason`, `TerminationReason`, `Severity`)
- Pure functions where possible; side effects only in runner and CLI
- Docstrings on all public classes/methods — include contract references (e.g., "Enforces DC-1")

## Contracts (15 families, 36+ contracts)
DC (Determinism), FC (Fairness), EC (Events/Decisions), GC (Guardrail),
EV (Event Semantics), VC (Visual Truth), MC (Mission Story), PC (Planner),
FD (Fire Dynamics), CC (Calibration), SC (Sanity Check), MP (Mask Parity),
WD (Wind Determinism), FG (Fog), TR (Triage)

See `docs/CONTRACTS.md` for full specification.

## Decisions Log
All assumptions and design decisions go in `docs/DECISIONS.md`.
