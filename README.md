# UAVBench

Operationally-realistic 2D UAV navigation benchmark with deterministic tracks for static control vs dynamic replanning stress.

## What You Get

- 9 scenarios (3 static control, 6 dynamic stress)
- 2 paper tracks: `static` (control), `dynamic` (forced-replan stress)
- 6 planners (global optimal, any-angle, incremental, anytime, reactive, sampling-based MPC)
- deterministic seeding and reproducible evaluation flow
- fair protocol invariants (shared snapshot, budget, cadence, checker)
- formal feasibility guardrail logs
- interaction causality metrics + ablation variants
- paper-oriented export scripts (CSV + LaTeX + manifest)

## Primary Guide

For full documentation, commands, demo flow, and troubleshooting, use:

- [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md)

## Canonical Evaluation Path

All results reported in the paper are produced through a single canonical pipeline:

```
cli/benchmark.py → benchmark/runner.py → envs/urban.py
```

The `missions/runner_v2.py` module is an **experimental** demo runner
retained for stakeholder visualization only.  It is **not** used for
reported benchmark results.

## Quick Start

```bash
cd /Users/konstantinos/Dev/uavbench
.venv/bin/mypy src tests
.venv/bin/pytest -q
```

```bash
.venv/bin/python - <<'PY'
from uavbench.scenarios.registry import list_scenarios, list_scenarios_by_track
from uavbench.planners import PLANNERS
print('scenarios', len(list_scenarios()))
print('static', len(list_scenarios_by_track('static')))
print('dynamic', len(list_scenarios_by_track('dynamic')))
print('planners', sorted(PLANNERS.keys()))
PY
```

## Fast Demo Commands

Static control run:

```bash
.venv/bin/python -m uavbench.cli.benchmark \
  --track static \
  --planners astar,theta_star \
  --trials 1 \
  --paper-protocol --fail-fast
```

Dynamic stress comparison:

```bash
.venv/bin/python -m uavbench.cli.benchmark \
  --scenarios osm_athens_comms_denied_hard_downtown \
  --planners astar,theta_star,dstar_lite,ad_star,dwa,mppi \
  --trials 1 \
  --paper-protocol \
  --protocol-variant default \
  --fail-fast
```

Mini benchmark demo script:

```bash
.venv/bin/python scripts/demo_benchmark.py
```

## Paper Workflow

Calibration:

```bash
.venv/bin/python scripts/calibrate_paper_tracks.py --seeds 3 --max-iters 10
```

Artifact export:

```bash
.venv/bin/python scripts/export_paper_artifacts.py --seeds 10 --output-root results/paper
```

Best-paper scientific validation pipeline:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/paper_best_paper_validation.py \
  --seeds 30 \
  --episode-horizon 320 \
  --output-root results/paper_scientific_validation_full \
  --strict-fairness
```

Runtime profile for reproducibility:

- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/runtime_profile.json`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/validation_manifest.json`

## Documentation Structure

- `BENCHMARK_GUIDE.md` complete practical guide
- `docs/PAPER_PROTOCOL.md` paper protocol summary
- `docs/API_REFERENCE.md` API details
- `docs/PERFORMANCE.md` performance notes
- `docs/legacy/root_md/` archived old root markdown files
