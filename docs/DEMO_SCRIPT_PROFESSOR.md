# UAVBench Professor Demo Script (10-12 min)

## Goal
Demonstrate that UAVBench is a fair, deterministic, dynamic 2D benchmark with planner-agnostic stress and feasibility guarantees.

## 0) Setup (30s)

```bash
cd /Users/konstantinos/Dev/uavbench
```

## 1) Credibility check (1 min)

```bash
.venv/bin/mypy src tests scripts/paper_best_paper_validation.py
.venv/bin/pytest -q
```

What to say:
- "Type checks and tests are green before any experiment."

## 2) Scenario + planner inventory (1 min)

```bash
.venv/bin/python - <<'PY'
from uavbench.scenarios.registry import list_scenarios, list_scenarios_by_track
from uavbench.planners import PLANNERS
print('total_scenarios', len(list_scenarios()))
print('static_track', len(list_scenarios_by_track('static')))
print('dynamic_track', len(list_scenarios_by_track('dynamic')))
print('planners', sorted(PLANNERS.keys()))
PY
```

What to say:
- "We keep 34 fixed scenario IDs, split into static control and dynamic stress tracks."

## 3) Static control run (1.5 min)

```bash
.venv/bin/python -m uavbench.cli.benchmark \
  --track static \
  --planners astar,theta_star \
  --trials 1 \
  --paper-protocol \
  --fail-fast
```

What to say:
- "Static track validates baseline solvability and map quality."

## 4) Dynamic stress run (2 min)

```bash
.venv/bin/python -m uavbench.cli.benchmark \
  --scenarios osm_athens_comms_denied_hard_downtown \
  --planners astar,dstar_lite,ad_star,dwa,mppi \
  --trials 1 \
  --paper-protocol \
  --protocol-variant default \
  --fail-fast
```

What to say:
- "Same scenario, same seed policy, same budgets, same trigger contract for all planners."
- "We expect planner separation under forced dynamic stress."

## 5) Fairness + protocol proof (2 min)

Open these files:

- `/Users/konstantinos/Dev/uavbench/docs/PAPER_PROTOCOL.md`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/fairness_audit.json`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/interdiction_fairness_summary.json`

What to show:
- protocol invariants (same snapshot/budget/cadence/checker/seed)
- planner-agnostic interdiction definition (reference corridor)
- `interdiction_hit_rate_reference_var_across_planners`

## 6) Feasibility guarantee proof (1.5 min)

Open:

- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/feasibility_proof.csv`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/ablation_deltas.csv`

What to say:
- "With guardrail enabled: no permanent infeasible episodes."
- "Without guardrail ablation: measurable infeasibility appears."
- "Fallback and relaxation are quantified, not hidden."

## 7) Stress and statistical story (2 min)

Open:

- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/figures/figure2_stress_intensity_curves.png`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/statistical_summary.csv`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/effect_sizes.csv`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/significance_table.tex`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/seed_stability_audit.csv`

What to say:
- "Static planners collapse faster with α; incremental/sampling planners degrade more gracefully."
- "Effect sizes and significance quantify separation; seed stability checks ranking robustness."

## 8) Runtime reproducibility (30s)

Open:

- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/runtime_profile.json`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/validation_manifest.json`

What to say:
- "Hardware/software/runtime and experiment manifest are exported for reproducibility."

## 9) Close (20s)

Open:

- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/FINAL_SCIENTIFIC_REPORT.md`
- `/Users/konstantinos/Dev/uavbench/results/paper_scientific_validation_full/reviewer2_verdict.json`

Closing line:
- "UAVBench is designed as deterministic dynamic stress instrumentation with fairness and feasibility contracts, not just another simulator benchmark." 
