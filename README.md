# UAVBench

Operationally-realistic 2D UAV navigation benchmark with deterministic tracks for static control vs dynamic replanning stress.

## What You Get

- 9 scenarios (3 static control, 6 dynamic stress)
- 2 paper tracks: `static` (control), `dynamic` (forced-replan stress)
- 6 planners across 3 families (search static, search adaptive, reactive)
- Deterministic seeding and reproducible evaluation flow
- Fair protocol invariants (shared snapshot, frozen fire state, area interdictions)
- Formal feasibility guardrail with multi-depth relaxation
- Mission objective system with domain-specific tasks
- Paper-oriented export scripts (CSV + LaTeX + manifest)

## Quick Start

```bash
pip install -e .
pytest tests/ -q
```

```bash
python -c "
from uavbench.scenarios.registry import list_scenarios, list_scenarios_by_track
from uavbench.planners import PLANNERS
print('scenarios', len(list_scenarios()))
print('static', len(list_scenarios_by_track('static')))
print('dynamic', len(list_scenarios_by_track('dynamic')))
print('planners', sorted(PLANNERS.keys()))
"
```

## Planners (6)

| Registry Key         | Algorithm              | Family           | Replanning        |
|----------------------|------------------------|------------------|-------------------|
| `astar`              | A* (4-connected)       | Search (static)  | None (one-shot)   |
| `theta_star`         | Theta* (any-angle)     | Search (static)  | None (one-shot)   |
| `periodic_replan`    | A* + periodic replan   | Search (adaptive) | Every N steps    |
| `aggressive_replan`  | A* + mask-change replan| Search (adaptive) | On obstacle change|
| `dstar_lite`         | D* Lite (incremental)  | Search (adaptive) | On path blocked   |
| `apf`                | Artificial Potential Field | Reactive      | Every step        |

## Scenarios (9)

| Scenario ID                          | Mission Type       | Difficulty | Track   |
|--------------------------------------|--------------------|------------|---------|
| `gov_fire_delivery_easy`             | fire_delivery      | easy       | static  |
| `gov_fire_delivery_medium`           | fire_delivery      | medium     | dynamic |
| `gov_fire_delivery_hard`             | fire_delivery      | hard       | dynamic |
| `gov_fire_surveillance_easy`         | fire_surveillance  | easy       | static  |
| `gov_fire_surveillance_medium`       | fire_surveillance  | medium     | dynamic |
| `gov_fire_surveillance_hard`         | fire_surveillance  | hard       | dynamic |
| `gov_flood_rescue_easy`              | flood_rescue       | easy       | static  |
| `gov_flood_rescue_medium`            | flood_rescue       | medium     | dynamic |
| `gov_flood_rescue_hard`              | flood_rescue       | hard       | dynamic |

## Demo Commands

Single run:

```bash
python -m uavbench run --seed 42
```

Static control run:

```bash
python -m uavbench run \
  --track static \
  --planners astar,theta_star \
  --trials 1 \
  --paper-protocol --fail-fast
```

Dynamic stress comparison:

```bash
python -m uavbench run \
  --scenarios gov_fire_delivery_medium \
  --planners astar,periodic_replan,aggressive_replan,dstar_lite,apf \
  --trials 1 \
  --paper-protocol --fail-fast
```

## Paper Workflow

Full paper experiments (all scenarios × all planners × 30 seeds):

```bash
python scripts/run_paper_experiments.py
```

Analyze results:

```bash
python scripts/analyze_paper_results.py
```

Export artifacts (CSV, manifest, determinism hashes):

```bash
python scripts/export_artifacts.py
```

Generate paper GIFs:

```bash
bash scripts/regenerate_paper_gifs.sh
```

## Documentation

- `docs/CONTRACTS.md` — 33 contracts across 11 families (DC, FC, EC, GC, EV, VC, MC, PC, FD, CC, SC, MP)
- `docs/ARCHITECTURE.md` — module map, pipeline, data contracts
- `docs/DECISIONS.md` — design decisions and rationale
- `docs/TECHNICAL_REFERENCE.md` — complete technical specification for paper
- `docs/REQUIREMENTS.md` — v2 requirements traceability
- `docs/TEST_PLAN.md` — test strategy and coverage
