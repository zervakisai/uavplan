# UAVBench — Paper Evaluation Protocol (v2.1)

> **Target venues:** ICRA / IROS / RA-L / NeurIPS Datasets & Benchmarks  
> **Last updated:** 2025-02-20  
> **Codebase SHA:** HEAD (post-sprint-3 — integrity hardening)  
> **Test count:** 268 (all passing)  
> **Source of truth:** this file + `benchmark/theoretical_validation.py`

---

## 0  Notation & Conventions

| Symbol | Meaning |
|--------|---------|
| $\pi_i$ | Planner $i$ |
| $\sigma_j$ | Scenario $j$ (mission × difficulty) |
| $s_k$ | Seed $k$ |
| $N$ | Seeds per (planner, scenario) cell |
| $T$ | Episode horizon (steps) |
| $d_{\text{Cohen}}$ | Cohen's $d$ effect size |
| CI-95 | $\bar{x} \pm 1.96 \cdot \sigma / \sqrt{n}$ |

---

## 1  Experimental Design

### 1.1  Planner Suite

| Registry Key | Class | Category | Algorithm |
|---|---|---|---|
| `astar` | `AStarPlanner` | Global optimal | A* (Hart et al. 1968) |
| `theta_star` | `ThetaStarPlanner` | Any-angle | Theta* (Nash et al. 2007) |
| `periodic_replan` | `PeriodicReplanPlanner` | Periodic re-init | A* from scratch every $K$ steps |
| `aggressive_replan` | `AggressiveReplanPlanner` | Aggressive re-init | A* from scratch at high cadence |
| `greedy_local` | `GreedyLocalPlanner` | Reactive | Greedy 1-step lookahead |
| `grid_mppi` | `GridMPPIPlanner` | Sampling MPC | MPPI (Williams et al. 2017) |
| `incremental_dstar_lite` | `DStarLiteRealPlanner` | **True incremental** | D* Lite (Koenig & Likhachev 2002) |

**Source:** `src/uavbench/planners/__init__.py:10–24`

### 1.2  Scenario Matrix

3 missions × 3 difficulties = **9 scenarios**.

| Mission | Incident Provenance | YAML |
|---|---|---|
| Civil Protection | 2018 Attica Wildfire | `configs/civil_protection_{easy,medium,hard}.yaml` |
| Maritime Domain | 2017 Agia Zoni II Spill | `configs/maritime_domain_{easy,medium,hard}.yaml` |
| Critical Infrastructure | 2021 Athens Bomb Threat | `configs/critical_infrastructure_{easy,medium,hard}.yaml` |

**Source:** `src/uavbench/scenarios/configs/`

### 1.3  Seed Protocol

- **Minimum:** $N = 30$ seeds per (planner, scenario) cell.
- **Recommended for paper:** $N = 50$.
- **Seeds:** $s \in \{0, 1, \ldots, N-1\}$.
- Each seed controls: RNG for dynamics, obstacle schedules, fire spread, traffic patterns.
- **Determinism contract:** same $(s_k, \sigma_j, \pi_i)$ → bit-identical episode trajectory.

**Verification:** `benchmark/fairness_audit.py:34–57` — `audit_seed_reproducibility()`.

### 1.4  Total Experiment Size

| Variant | Planners | Scenarios | Seeds | Episodes |
|---|---|---|---|---|
| Minimum ($N=30$) | 7 | 9 | 30 | **1,890** |
| Recommended ($N=50$) | 7 | 9 | 50 | **3,150** |
| With ablations (×5) | 7 | 9 | 50 | **15,750** |

---

## 2  Fair Evaluation Invariants

Every invariant is enforced in code and logged per-episode.

| # | Invariant | Enforcement Location | Log Field |
|---|---|---|---|
| F1 | Same world snapshot per `plan()`/`replan()` at identical step + seed | `cli/benchmark.py` — all planners receive identical `env.get_observation()` | `snapshot_hash` |
| F2 | Same planning time budget | `ScenarioConfig.plan_budget_{static,dynamic}_ms` | `plan_budget_ms` |
| F3 | Same replanning cadence + max replans | `ScenarioConfig.replan_every_steps`, `max_replans_per_episode` | `replan_every_steps`, `max_replans` |
| F4 | Same collision checker + grid discretization | `BasePlanner` — 4-connected, `block_buildings`, `no_fly` | N/A (structural) |
| F5 | Same seed → identical dynamics | `np.random.default_rng(seed)` throughout | `seed`, `deterministic` |
| F6 | Planner-agnostic interdiction | Cut against **reference corridor** (A*), not per-planner path | `interdiction_reference_planner` |

**Source:** `benchmark/fairness_audit.py:1–12`, `cli/benchmark.py` run loops.

### 2.1  Interdiction Protocol

Interdictions are scheduled against a **reference corridor** computed at reset time.

- **Reference planner:** `astar` (configurable: `interdiction_reference_planner`)
- **Cut points:** 30% and 65% along reference corridor
- **Event times:** `event_t1`, `event_t2` (from YAML)
- **Log fields:** `path_interdiction_1`, `path_interdiction_2`, `forced_replan_triggered`
- **Fairness metric:** `interdiction_hit_rate_reference_var_across_planners` → target: **0.0**

**Source:** `updates/forced_replan.py:1–224`

---

## 3  Feasibility Guarantee

Every episode must be feasible after guardrail application.

### 3.1  Mechanism

At each dynamic update step:

1. **BFS reachability check** from current UAV position to goal (`solvability.py:54–95`).
2. If disconnected → **progressive relaxation:**
   - Clear forced interdiction cells
   - Reduce NFZ growth rate and radius
   - Remove traffic closures
   - Activate emergency corridor
3. If still disconnected for 2+ consecutive ticks → **hard deconfliction fallback.**

### 3.2  Logged Proof Fields

| Field | Type | Meaning |
|---|---|---|
| `reachability_failed_before_relax` | bool | BFS failed before guardrail |
| `relaxation_applied` | dict | What was relaxed (blocks, NFZ, closures) |
| `corridor_fallback_used` | bool | Emergency corridor activated |
| `feasible_after_guardrail` | bool | **Must be `true`** for valid episode |
| `guardrail_activation_rate` | float | Fraction of steps with guardrail active |
| `corridor_fallback_rate` | float | Fraction of steps using corridor |

**Source:** `solvability.py:100–243`, `cli/benchmark.py` feasibility loop.

---

## 4  Claims-Evidence Map (Hypotheses)

Each claim maps to a metric, a comparison protocol, a falsifiability criterion, and exact code locations.

### H1: Adaptive Advantage

| | |
|---|---|
| **Claim** | Dynamic replanning planners achieve higher success rates than static planners in dynamic scenarios. |
| **Metric** | `success_rate` (mean over seeds) |
| **Groups** | A = `{periodic_replan, aggressive_replan, greedy_local, grid_mppi, incremental_dstar_lite}` vs B = `{astar, theta_star}` |
| **Test** | Two-sample t-test, one-sided ($\mu_A > \mu_B$), $\alpha = 0.05$ |
| **Effect size** | $d_{\text{Cohen}} \geq 0.5$ (medium) required |
| **Falsified if** | Static planners achieve $\geq 80\%$ success in dynamic scenarios |
| **Code** | `theoretical_validation.py:52–59`, metric extraction at lines 196–247 |

### H2: Risk Awareness

| | |
|---|---|
| **Claim** | Risk-aware planners produce lower cumulative risk exposure. |
| **Metric** | `risk_exposure_integral` |
| **Groups** | A = `{grid_mppi}` (risk-cost-map-aware) vs B = `{periodic_replan, aggressive_replan, greedy_local}` |
| **Test** | Two-sample t-test, one-sided ($\mu_A < \mu_B$), $\alpha = 0.05$ |
| **Effect size** | $d_{\text{Cohen}} \geq 0.2$ (small) minimum |
| **Falsified if** | Risk-aware planners show $\geq$ risk exposure of non-risk-aware planners |
| **Code** | `theoretical_validation.py:60–67` |

### H3: Behavioral Differentiation

| | |
|---|---|
| **Claim** | Behaviorally distinct adaptive planners produce statistically different replan trigger distributions. |
| **Metric** | `total_replans` |
| **Groups** | Pairwise: `periodic_replan` vs `grid_mppi` vs `incremental_dstar_lite` |
| **Test** | Kruskal-Wallis + post-hoc Dunn's test, $\alpha = 0.05$ |
| **Effect size** | $d_{\text{Cohen}} \geq 0.5$ between any two planner pairs |
| **Falsified if** | $d_{\text{Cohen}} < 0.5$ for all pairwise comparisons |
| **Code** | `theoretical_validation.py:68–75` |

### H4: Path Quality Spread

| | |
|---|---|
| **Claim** | Path length spread meaningfully differentiates planner quality. |
| **Metric** | `path_length` (normalized by BFS optimal) |
| **Groups** | Best-performing vs worst-performing planner (by success rate) |
| **Test** | Mann-Whitney U, $\alpha = 0.05$ |
| **Effect size** | Normalized path length range $\geq 0.1$ |
| **Falsified if** | Range $< 0.1$ across all planners |
| **Code** | `theoretical_validation.py:76–83` |

### H5: Guardrail Necessity

| | |
|---|---|
| **Claim** | Disabling the guardrail increases infeasibility collapse rate. |
| **Metric** | `feasible_after_guardrail` |
| **Groups** | Default variant vs `no_guardrail` ablation |
| **Test** | Fisher's exact test on feasibility counts |
| **Effect size** | $> 5\%$ absolute increase in infeasibility |
| **Falsified if** | `no_guardrail` shows $\leq 5\%$ increase |
| **Code** | `theoretical_validation.py:84–91` |

### H6: Causal Interaction Effects

| | |
|---|---|
| **Claim** | Fire→traffic causal feedback creates measurable interaction effects. |
| **Metric** | `fire_traffic_feedback_rate`, `interaction_fire_nfz_overlap_ratio` |
| **Groups** | Default vs `no_interactions` ablation |
| **Test** | Paired t-test (same seeds), $\alpha = 0.05$ |
| **Falsified if** | `fire_traffic_feedback_rate = 0` in all wildfire scenarios |
| **Code** | `theoretical_validation.py:92–99`, `dynamics/interaction_engine.py:1–185` |

### H7: Sampling MPC vs Reactive

| | |
|---|---|
| **Claim** | MPPI planner reduces replan storm oscillation compared to greedy reactive. |
| **Metric** | `total_replans` |
| **Groups** | `grid_mppi` vs `greedy_local` |
| **Test** | Mann-Whitney U, $\alpha = 0.05$ |
| **Falsified if** | MPPI shows $\geq$ replans as greedy local |
| **Code** | `theoretical_validation.py:100–106` |

### H8: Incremental Efficiency *(NEW — D* Lite real)*

| | |
|---|---|
| **Claim** | True incremental D* Lite performs fewer vertex expansions than from-scratch replanning per replan event. |
| **Metric** | `expansions` (per replan call) |
| **Groups** | `incremental_dstar_lite` vs `periodic_replan` |
| **Test** | Paired Wilcoxon signed-rank (same seed, same scenario, matched replan events) |
| **Effect size** | $d_{\text{Cohen}} \geq 0.8$ (large) expected |
| **Falsified if** | Incremental expansions $\geq$ from-scratch expansions in $> 50\%$ of replan events |
| **Code** | `planners/dstar_lite_real.py:164–190` (compute_shortest_path), `tests/test_dstar_lite_real.py::TestIncrementalEfficiency` |

---

## 5  Metrics Taxonomy

### 5.1  Primary Metrics (Table 1 in paper)

| Metric | Unit | Source | Aggregation |
|---|---|---|---|
| `success_rate` | fraction | `result["success"]` | mean ± CI-95 |
| `path_length` | steps | `result["path_length"]` | mean ± CI-95 |
| `compute_time_ms` | ms | `result["compute_time_ms"]` | median (IQR) |
| `total_replans` | count | `result["total_replans"]` | mean ± CI-95 |
| `risk_exposure_integral` | dimensionless | `result["risk_exposure_integral"]` | mean ± CI-95 |
| `expansions` | count | `result["expansions"]` | mean ± CI-95 |

### 5.2  Fairness & Validity Metrics (Table 2)

| Metric | Target | Source |
|---|---|---|
| `interdiction_hit_rate_reference` | 1.0 | forced_replan.py |
| `interdiction_hit_rate_var` | 0.0 | fairness_audit.py |
| `feasible_after_guardrail` | 1.0 | solvability.py |
| `guardrail_activation_rate` | low | solvability.py |
| `seed_reproducibility` | 100% | fairness_audit.py |

### 5.3  Interaction Metrics (Table 3 / Ablation)

| Metric | Source |
|---|---|
| `interaction_fire_nfz_overlap_ratio` | interaction_engine.py |
| `interaction_fire_road_closure_rate` | interaction_engine.py |
| `interaction_congestion_risk_corr` | interaction_engine.py |
| `dynamic_block_entropy` | interaction_engine.py |

### 5.4  Mission Metrics (if mission track used)

| Metric | Source |
|---|---|
| `mission_score` | missions/engine.py |
| `task_completion_rate` | missions/engine.py |
| `waypoint_visit_rate` | missions/engine.py |
| `task_log` | missions/engine.py — per-task timing |

---

## 6  Ablation Protocols

Five protocol variants test component contributions:

| Variant | What's removed | Tests |
|---|---|---|
| `default` | Nothing (full benchmark) | Baseline |
| `no_interactions` | Fire→NFZ, fire→road, congestion→risk coupling | H6 |
| `no_forced_breaks` | Scheduled interdictions disabled | H1, H3 |
| `no_guardrail` | BFS feasibility + relaxation disabled | H5 |
| `risk_only` | Dynamic obstacles non-blocking, only risk field | H2 |
| `blocking_only` | No risk field, only blocking dynamics | H2 |

**Activation:** `--protocol-variant <name>` in CLI.

---

## 7  Statistical Analysis Protocol

### 7.1  Per-hypothesis testing

1. Check normality (Shapiro-Wilk) per group.
2. If normal → parametric (t-test, ANOVA).  If not → non-parametric (Mann-Whitney, Kruskal-Wallis).
3. Report: test statistic, p-value, effect size ($d_{\text{Cohen}}$ or rank-biserial $r$), CI-95.
4. Multiple comparison correction: Holm-Bonferroni for $m = 8$ hypotheses.
5. Adjusted significance: $\alpha_{\text{adj}} = 0.05 / 8 = 0.00625$ for most conservative.

### 7.2  Reporting format

```
H1: μ_adaptive = 0.87 ± 0.04, μ_static = 0.52 ± 0.06
    d = 1.23 (large), p < 0.001**, supported ✓
```

### 7.3  Automated report generation

```python
from uavbench.benchmark.theoretical_validation import generate_validation_report
report = generate_validation_report(episode_results, Path("results/paper/validation.json"))
```

**Source:** `benchmark/theoretical_validation.py:108–195`

---

## 8  Reproducibility Checklist

| Item | Status | Evidence |
|---|---|---|
| Fixed random seeds | ✅ | `np.random.default_rng(seed)` everywhere |
| Deterministic dynamics | ✅ | `tests/test_vv_contracts.py::TestDeterminism` (5 tests) |
| Pin Python version | ✅ | `pyproject.toml` requires ≥3.10 |
| Pin dependencies | ✅ | `pyproject.toml` with version bounds |
| All tests pass | ✅ | 268 tests, 0 failures |
| Planner-agnostic interdiction | ✅ | Reference corridor (A*), not per-planner |
| Solvability proof per episode | ✅ | BFS + guardrail + logged fields |
| No silent failures | ✅ | Explicit error on infeasibility |
| Honest planner naming | ✅ | Docstrings cite canonical papers |
| True incremental planner | ✅ | `DStarLiteRealPlanner` — Koenig 2002 |
| Mission framework wired | ✅ | `--mode mission` in CLI (`run_mission_episode()`) |
| Realism knobs verified active | ✅ | All 9 YAMLs: top-level `comms_dropout_prob`, `constraint_latency_steps`, `gnss_noise_sigma` |
| Interdiction reference = astar | ✅ | All 9 YAMLs: `interdiction_reference_planner: astar` |

---

## 9  Commands

### Full paper run (recommended)

```bash
cd /path/to/uavbench

# 1. Run all tests
.venv/bin/pytest tests/ -q

# 2. Run full benchmark — navigation track (all planners × all scenarios × 50 seeds)
.venv/bin/python scripts/paper_best_paper_validation.py \
  --seeds 50 \
  --output-dir results/paper/

# 3. Run full benchmark — mission track
uavbench --mode mission --track all \
  --planners astar,theta_star,periodic_replan,aggressive_replan,greedy_local,grid_mppi,incremental_dstar_lite \
  --trials 50 --paper-protocol --save-csv --save-json \
  --output-dir results/paper_mission/

# 4. Run ablations
for variant in no_interactions no_forced_breaks no_guardrail risk_only blocking_only; do
  .venv/bin/python scripts/paper_best_paper_validation.py \
    --seeds 50 \
    --protocol-variant $variant \
    --output-dir results/paper/ablations/
done

# 4. Generate validation report
.venv/bin/python -c "
from pathlib import Path
from uavbench.benchmark.theoretical_validation import generate_validation_report
import json
results = json.loads(Path('results/paper/episodes.jsonl').read_text().strip().split('\n')[-1])
# ... collect all episode results and call generate_validation_report()
"

# 5. Export paper artifacts (tables, figures)
.venv/bin/python scripts/export_paper_artifacts.py --results-dir results/paper/
```

### Smoke test (fast sanity check)

```bash
.venv/bin/python scripts/paper_best_paper_validation.py \
  --seeds 3 \
  --planners astar,periodic_replan,incremental_dstar_lite \
  --scenarios civil_protection_easy \
  --output-dir results/smoke/
```

---

## 10  Paper Table Templates

### Table 1: Main Results

```
| Planner               | Success ↑ | Path Len ↓ | Replans | Time (ms) ↓ | Risk ↓ |
|-----------------------|-----------|------------|---------|-------------|--------|
| A*                    | ...       | ...        | 0       | ...         | ...    |
| Θ*                    | ...       | ...        | 0       | ...         | ...    |
| Periodic Replan       | ...       | ...        | ...     | ...         | ...    |
| Aggressive Replan     | ...       | ...        | ...     | ...         | ...    |
| Greedy Local          | ...       | ...        | ...     | ...         | ...    |
| Grid MPPI             | ...       | ...        | ...     | ...         | ...    |
| D* Lite (incremental) | ...       | ...        | ...     | ...         | ...    |
```

### Table 2: Ablation Study (Δ from default)

```
| Variant            | Δ Success | Δ Replans | Δ Risk  | Guardrail % |
|--------------------|-----------|-----------|---------|-------------|
| no_interactions    | ...       | ...       | ...     | ...         |
| no_forced_breaks   | ...       | ...       | ...     | ...         |
| no_guardrail       | ...       | ...       | ...     | ...         |
| risk_only          | ...       | ...       | ...     | ...         |
| blocking_only      | ...       | ...       | ...     | ...         |
```

### Table 3: Hypothesis Validation Summary

```
| ID | Claim (abbreviated)      | d_Cohen | p-value | Supported |
|----|--------------------------|---------|---------|-----------|
| H1 | Adaptive > Static        | ...     | ...     | ✓/✗       |
| H2 | Risk-aware < Non-risk    | ...     | ...     | ✓/✗       |
| H3 | Behavioral diff.         | ...     | ...     | ✓/✗       |
| H4 | Path quality spread      | ...     | ...     | ✓/✗       |
| H5 | Guardrail necessity      | ...     | ...     | ✓/✗       |
| H6 | Causal interactions      | ...     | ...     | ✓/✗       |
| H7 | MPPI < Reactive replans  | ...     | ...     | ✓/✗       |
| H8 | Incremental efficiency   | ...     | ...     | ✓/✗       |
```

---

## 11  Reviewer FAQ

**Q: Are the planner names honest?**  
A: Yes. All renamed planners have docstrings citing the canonical paper and disclosing simplifications. See `tests/test_planner_naming_honesty.py` (15 tests).

**Q: Is there a true incremental planner?**  
A: Yes. `DStarLiteRealPlanner` implements D* Lite (Koenig & Likhachev 2002) with incremental g/rhs updates. See `planners/dstar_lite_real.py` (532 lines, 28 tests).

**Q: Are scenarios grounded in real incidents?**  
A: Yes. Each mission cites a real-world event. See `docs/SCENARIO_CARDS.md`.

**Q: How do you ensure planner-agnostic evaluation?**  
A: Interdictions target a reference corridor (A*), not the per-planner path. Variance across planners is logged and audited. See `benchmark/fairness_audit.py`.

**Q: What if the map becomes infeasible?**  
A: A BFS feasibility check runs every step. If disconnected, progressive relaxation + emergency corridor restore feasibility. The field `feasible_after_guardrail` is logged per-episode. See `benchmark/solvability.py`.

---

*Supersedes `docs/PAPER_PROTOCOL_v1.md`.*
