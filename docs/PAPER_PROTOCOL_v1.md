# UAVBench Paper Protocol (v1.1)

## 1) Scope

This protocol defines a fair, deterministic, operationally-realistic 2D benchmark for UAV routing under dual-use constraints.

- `static` track: control conditions where static planners should be near-perfect.
- `dynamic` track: forced-replanning pressure with causal interactions and feasibility guardrails.

## 2) Fair Evaluation Protocol

### 2.1 Invariants (must hold for every planner)

1. Same world snapshot per `plan()/replan()` call at identical simulation step and seed.
2. Same per-call planning budget (ms) for all planners in a scenario.
3. Same replanning cadence (`replan_every_steps`) and same max replans (`max_replans_per_episode`).
4. Same collision checker and grid discretization.
5. Same seed reproduces identical dynamics, events, and interaction masks.

### 2.2 Config fields

- `plan_budget_static_ms`
- `plan_budget_dynamic_ms`
- `replan_every_steps`
- `max_replans_per_episode`
- `interdiction_reference_planner`

## 3) Planner-Agnostic Forced Replan Protocol

Interdictions are scheduled against a **reference corridor** at reset time, not against each planner’s own path.

- Reference planner: `interdiction_reference_planner` (`astar|theta_star`)
- Cut points: 30% and 65% along reference corridor
- Event times: `event_t1`, `event_t2`
- Required logs:
  - `path_interdiction_1`
  - `path_interdiction_2`
  - `forced_replan_triggered`
- Fairness metrics:
  - `interdiction_hit_rate_reference`
  - `interdiction_hit_rate_reference_var_across_planners`

Interpretation:
- Low variance across planners on reference-hit metric indicates planner-agnostic stress instrumentation.

## 3.1 Replanning Trigger Contract (identical for all adaptive planners)

At each simulation step, replanning can be triggered only by:

1. `path invalidation` (upcoming corridor intersects dynamic blocking layers)
2. `forced event` (scheduled interdiction activation)
3. `cadence` (`replan_every_steps`)

Additional safety fallback:
- `stuck_fallback` may trigger replanning when repeated move rejection occurs.

Global hard limit:
- `max_replans_per_episode` (same for all planners in scenario).

Logged counters per episode:
- `replan_trigger_path_invalidation_count`
- `replan_trigger_forced_event_count`
- `replan_trigger_cadence_count`
- `replan_trigger_stuck_fallback_count`

## 4) Formal Feasibility Guarantee

### 4.1 Mechanism

After each dynamic update:

1. Reachability check (BFS) from current UAV cell to goal.
2. If disconnected:
  - relax forced interdiction cells
  - reduce NFZ growth / radii and relax traffic closures
  - activate emergency corridor deconfliction
3. If still disconnected on consecutive ticks:
  - hard deconfliction fallback (controlled emergency opening)

### 4.2 Logged proof fields

- `reachability_failed_before_relax`
- `relaxation_applied`:
  - `forced_blocks_cleared`
  - `nfz_rate_delta`
  - `nfz_radius_delta`
  - `closures_removed`
  - optional `hard_deconfliction`
- `corridor_fallback_used`
- `feasible_after_guardrail` (target: always `true` after guardrail)
- `guardrail_activation_rate`
- `corridor_fallback_rate` (target: low)
- `relaxation_magnitude`

## 5) Dynamic Interaction Specification + Metrics

### 5.1 Causal rules

- Fire -> NFZ pressure (expansion coupling)
- Fire -> road closures (front-adjacent roads)
- Traffic congestion -> risk amplification
- Pedestrian/population map -> non-blocking risk field

### 5.2 Exported interaction metrics

- `interaction_fire_nfz_overlap_ratio`
- `interaction_fire_road_closure_rate`
- `interaction_congestion_risk_corr`
- `dynamic_block_entropy`
- `interdiction_hit_rate_reference`

## 6) Dual-Use Operational Constraints

- Dynamic restricted airspace (NFZ/deconfliction)
- Safety buffers around traffic, crowds, intruders
- Emergency corridor as managed fallback lane
- Mission-level behavior under time-critical and safety-critical objectives

## 7) Planner Buckets

- Global optimal: `astar`
- Any-angle: `theta_star`
- Incremental: `dstar_lite`
- Anytime incremental: `ad_star`
- Reactive: `dwa`
- Sampling-based MPC: `mppi`

## 8) Ablation Protocols

Supported variants (`--protocol-variant`):

- `no_interactions`
- `no_forced_breaks`
- `no_guardrail`
- `risk_only`
- `blocking_only`

Exporter produces:

- `results/paper/ablations/*_episodes.csv`
- `results/paper/ablations/*_aggregates.csv`
- `results/paper/tables/ablation_*.tex`
- `results/paper/tables/ablation_interactions.tex`

## 9) Commands

```bash
cd /Users/konstantinos/Dev/uavbench
.venv/bin/pytest -q
.venv/bin/mypy src tests
```

```bash
.venv/bin/python -m uavbench.cli.benchmark \
  --track dynamic \
  --planners astar,theta_star,dstar_lite,ad_star,dwa,mppi \
  --trials 1 \
  --paper-protocol \
  --protocol-variant default \
  --fail-fast
```

```bash
.venv/bin/python scripts/export_paper_artifacts.py --seeds 10 --output-root results/paper
```

## 10) Runtime Reproducibility Profile (Mac M1-friendly)

For scientific runs, export runtime context in:

- `runtime_profile.json`
- `validation_manifest.json`

Required fields:

- `cpu_model`
- `machine`
- `platform`
- `python`
- `wall_clock_seconds`
- `seeds`
- `stats_planners`
- `stress_planners`
- `episode_horizon`
