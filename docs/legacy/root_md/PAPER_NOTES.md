"""PAPER_NOTES: Evaluation Guidance and Expected Results

This document outlines key scientific claims and the experiments/plots that support them.
"""

# Paper-Ready Claims & Supporting Evidence

## Claim 1: Forced Replanning is Necessary in Dynamic Scenarios

**Evidence**: 
- Plot: Comparison of replans needed (first replan step) across scenarios
- Table: Success rate with replanning enabled vs. disabled

**Experiments**:
```bash
# Run with replanning enabled
uavbench --scenarios osm_athens_wildfire_easy osm_athens_emergency_easy \
  --planners astar adaptive_astar --seeds 0..9

# Expected: adaptive_astar (replanning enabled) outperforms astar on dynamic scenarios
# Metric: success_rate, path_length, replans_count
```

**Plot Format**:
```
X-axis: Scenario
Y-axis: First replan step (if any)
Overlay: scatter with error bars; highlight scenarios where replan is required early (step < 50)
```

## Claim 2: Solvability Certificates Prevent Unsolvable Scenarios

**Evidence**:
- Test: solvability_cert_ok=True for all scenarios at load time
- Figure: Disjoint path visualization (2+ colored corridors) for sample scenarios

**Experiments**:
```python
from uavbench.scenarios.registry import list_scenarios
from uavbench.benchmark.solvability import check_solvability_certificate
from uavbench.scenarios.loader import load_scenario

for scenario_id in list_scenarios()[:3]:  # Show first 3
    cfg = load_scenario(Path(f"src/uavbench/scenarios/configs/{scenario_id}.yaml"))
    ok, reason = check_solvability_certificate(...)
    print(f"{scenario_id}: {ok} ({reason})")
```

**Table Format**:
```
Scenario | Solvable? | Reason | Num Disjoint Paths
osm_athens_wildfire_easy | Yes | 2 disjoint paths verified | 2+
urban_easy | Yes | fully connected grid | 4+
...
```

## Claim 3: Deterministic Seeding Enables Reproducible Benchmarks

**Evidence**:
- Test: Same (scenario, seed) produces identical trajectory hash
- Figure: Comparison plot showing 0 variance between re-runs

**Experiments**:
```bash
# Run same benchmark twice
for i in 1 2; do
  uavbench --scenarios urban_easy --planners astar --seeds 0 > results_run_$i.csv
done

# Compare: should be identical
diff results_run_1.csv results_run_2.csv  # Should be empty
```

**Test Code**:
```python
def test_deterministic_seeding():
    env1 = UrbanEnv(cfg)
    obs1, _ = env1.reset(seed=42)
    traj1 = list(env1.trajectory)
    
    env2 = UrbanEnv(cfg)
    obs2, _ = env2.reset(seed=42)
    traj2 = list(env2.trajectory)
    
    assert traj1 == traj2  # Identical trajectories
```

## Claim 4: Multi-Objective Metrics Reveal Planner Trade-Offs

**Evidence**:
- Pareto front plot: path_length vs. planning_time_ms and path_length vs. fire_exposure
- Table: Efficiency-safety correlation matrix

**Experiments**:
```bash
uavbench \
  --scenarios osm_athens_wildfire_easy osm_athens_wildfire_medium osm_athens_wildfire_hard \
  --planners astar theta_star adaptive_astar \
  --seeds 0 1 2 3 4 5 \
  --output-dir results/
```

**Plot Format**:
```
Subplot 1: Path Length vs. Planning Time
  X-axis: planning_time_ms
  Y-axis: path_length
  Scatter: colored by planner, sized by success_rate
  Interpretation: Is Theta* faster with shorter paths?

Subplot 2: Path Length vs. Fire Exposure
  X-axis: fire_exposure (safety)
  Y-axis: path_length (efficiency)
  Scatter: colored by scenario difficulty
  Interpretation: Do harder scenarios have worse safety-efficiency trade-off?
```

## Claim 5: Oracle Baseline Quantifies Replanning Regret

**Evidence**:
- Regret plot: regret_length for each (scenario, planner) pair
- Table: Oracle vs. Non-Oracle success rate comparison

**Experiments**:
```bash
# Run with oracle horizon
uavbench \
  --scenarios osm_athens_emergency_easy osm_athens_emergency_medium \
  --planners astar adaptive_astar \
  --oracle-horizon 100 \
  --seeds 0..9
```

**Plot Format**:
```
X-axis: Planner
Y-axis: Regret Length [%] (how much longer than oracle)
Bars: colored by scenario
Annotation: Oracle plans with 100-step lookahead; error bars show std over seeds

Interpretation:
- If regret_length ≈ 0% for astar, then oracle and greedy paths are similar
- If regret_length > 10% for astar, then lookahead helps significantly
```

## Claim 6: Stress-Test Regime Challenges All Planners

**Evidence**:
- Success rate comparison: naturalistic vs. stress_test regimes
- Figure: Replanning requirement heatmap (first_replan_step by scenario)

**Experiments**:
```bash
# Naturalistic scenarios (minimal dynamics)
uavbench \
  --scenarios $(grep "regime: naturalistic" *.yaml | head -10) \
  --planners astar theta_star adaptive_astar \
  --seeds 0..4

# Stress-test scenarios (maximum dynamics)
uavbench \
  --scenarios $(grep "regime: stress_test" *.yaml | head -10) \
  --planners astar theta_star adaptive_astar \
  --seeds 0..4
```

**Comparison Table**:
```
Regime | Avg Success Rate | Avg Planning Time | Avg Replans
Naturalistic | 98.5% | 1.2ms | 0.1
Stress-Test | 87.3% | 2.8ms | 1.7

Interpretation: Stress tests significantly increase difficulty
```

## Claim 7: Time Budget Fairness Enforces Comparable Planning

**Evidence**:
- Test: All planners timeout gracefully at max_planning_time_ms
- Table: Max observed planning_time across all runs

**Experiments**:
```python
for planner_id in ["astar", "theta_star", "adaptive_astar"]:
    config.planner_ids = [planner_id]
    config.max_planning_time_ms = 200  # Fairness constraint
    
    # Should not exceed 200ms
    results = runner.run()
    max_time = max(e.planning_time_ms for r in results.values() for e in r)
    assert max_time <= 200 + 10  # Small tolerance
    print(f"{planner_id}: max_time={max_time:.1f}ms")
```

**Plot Format**:
```
Box plot: planning_time_ms distribution by planner
Y-axis: planning_time_ms (log scale)
X-axis: Planner
Horizontal line at 200ms (budget)
All planners should have <5% of runs exceeding budget
```

# Table & Figure Checklist for Publication

## Required Tables

1. **Table 1: Scenario Taxonomy**
   - Mission Type | Difficulty | Regime | Dynamics | Num Scenarios
   - 9 rows (one per mission type)

2. **Table 2: Metrics Definitions**
   - Metric | Category | Definition | Unit
   - ≥15 rows (efficiency, safety, regret, etc.)

3. **Table 3: Baseline Results**
   - Scenario | Planner | Success Rate | Path Length (mean±std) | Plan Time (ms)
   - ≥6 rows (3 scenarios × 2 planners)

4. **Table 4: Solvability Certificates**
   - Scenario | Solvable? | Num Disjoint Paths | Reason
   - ≥5 rows (sample scenarios)

5. **Table 5: Regret Analysis (Oracle)**
   - Scenario | Planner | Success % | Regret Length (%) | Regret Risk (%)
   - ≥4 rows (oracle comparison)

## Required Figures

1. **Figure 1: Scenario Overview**
   - Top-down screenshots of 3 mission types (wildfire, emergency, SAR)
   - Show map, start/goal, dynamic elements (fire, vehicles, targets)
   - 3 subplots (1 per tile: Penteli, Downtown, Piraeus)

2. **Figure 2: Pareto Front (Multi-Objective)**
   - Scatter plot: X = planning_time_ms, Y = path_length, colored by planner
   - Overlay: X = fire_exposure, Y = path_length (second subplot)
   - Size by success_rate
   - Legend: planner names

3. **Figure 3: Replanning Timeline**
   - 4 scenarios (varying difficulty)
   - X-axis: step, Y-axis: path_blocked indicator (0/1 bars)
   - Overlay: first_replan_step line
   - Show forced replan occurs early in dynamic scenarios

4. **Figure 4: Success Rate by Regime**
   - Bar chart: X = planner, Y = success_rate
   - Grouped: Naturalistic (light) vs. Stress-Test (dark)
   - Error bars: 95% CI
   - Annotation: N trials per bar

5. **Figure 5: Oracle Regret**
   - Bar chart: X = planner, Y = regret_length [%]
   - Colored by scenario
   - Error bars: std dev
   - Annotation: oracle_horizon value

6. **Figure 6: Computational Efficiency**
   - X-axis: planning_time_ms (log scale)
   - Y-axis: path_length
   - Scatter colored by planner
   - Regression lines to show trade-off

# Running the Full Paper Benchmark

```bash
# Standard suite
python -m uavbench.benchmark.runner \
  --scenarios $(python -c "from uavbench.scenarios.registry import list_scenarios; print(' '.join(list_scenarios()[:15]))") \
  --planners astar theta_star adaptive_astar \
  --seeds $(seq 0 9) \
  --output-dir results/paper_standard

# Oracle suite (regret analysis)
python -m uavbench.benchmark.runner \
  --scenarios osm_athens_wildfire_easy osm_athens_emergency_easy \
  --planners astar adaptive_astar \
  --oracle-horizon 100 \
  --seeds $(seq 0 9) \
  --output-dir results/paper_oracle

# Ablation: naturalistic vs stress-test
python -c "
from uavbench.scenarios.registry import list_scenarios_by_regime
from uavbench.scenarios.schema import Regime

nat = list_scenarios_by_regime(Regime.NATURALISTIC)
stress = list_scenarios_by_regime(Regime.STRESS_TEST)
print('Naturalistic:', ' '.join(nat[:5]))
print('Stress-Test:', ' '.join(stress[:5]))
"

# Then run for each regime and compare success rates
```

# Expected Results Summary

## Baseline (A* planner, N=5 seeds)

| Scenario | Success % | Path Len (mean±std) | Plan Time (ms) | Replans |
|----------|-----------|-------------------|----------------|---------|
| urban_easy | 100 | 18.5±0.5 | 0.8 | 0 |
| osm_athens_wildfire_easy | 95 | 250±15 | 5.2 | 0 |
| osm_athens_emergency_easy | 98 | 280±20 | 3.1 | 0 |
| osm_athens_wildfire_medium | 92 | 300±25 | 6.5 | 2.1±1.2 |
| osm_athens_emergency_medium | 90 | 320±30 | 4.2 | 1.8±0.9 |

## Comparative (vs. Oracle)

| Scenario | Planner | Success % | Regret Length | Regret Risk |
|----------|---------|-----------|---------------|-------------|
| osm_athens_wildfire_easy | astar | 95 | +8.2% | +5.1% |
| osm_athens_wildfire_easy | adaptive_astar | 97 | +2.1% | +0.8% |
| osm_athens_emergency_easy | astar | 98 | +3.5% | +2.3% |
| osm_athens_emergency_easy | adaptive_astar | 99 | +0.7% | +0.2% |

# Key Insights to Highlight

1. **Forced Replanning Works** — Dynamic scenarios force strategy changes, enabling rigorous evaluation
2. **Solvability is Verifiable** — Guarantees at load time prevent wasted runs on unsolvable problems
3. **Determinism Enables Fair Comparison** — Same seed ⟹ reproducible comparisons
4. **Multi-Objective Trade-Offs** — Path length, planning time, safety form Pareto frontiers
5. **Oracle Baseline is Achievable** — With sufficient lookahead, planners approach oracle performance
6. **Stress Tests Differentiate Planners** — Incremental planning (adaptive) outperforms static under dynamics
7. **Time Budgets Enforce Fairness** — 200ms timeout applies uniformly across all planners

# Recommendations for Authors

1. Use **Table 3** as main results table
2. Use **Figure 2** (Pareto) as centerpiece multi-objective visualization
3. Use **Figure 3** (Replanning Timeline) to motivate forced replanning claim
4. Use **Table 5** (Regret) to demonstrate oracle baseline utility
5. Include **Figure 4** (Success Rate by Regime) to show stress-test effectiveness
6. Cite **test_sanity.py** results in methods to demonstrate reproducibility

# Citation Format for Results

When reporting results, use:
```
"We evaluated {X} planners on {Y} scenarios with {N} seeds using UAVBench v0.2 
(Zervakis, 2026). Solvability certificates verified ≥2 disjoint paths per scenario. 
Time budget: 200ms. Results aggregated with bootstrap 95% CI."
```
