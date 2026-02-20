# UAVBench — Full Academic Reviewer Audit

**Audit quality standard:** ICRA / IROS / RA-L / NeurIPS Datasets & Benchmarks  
**Auditor:** Code-level static analysis (every .py file, every scenario config)  
**Date:** 2025-01-XX  
**Codebase snapshot:** 61 source files, 17,039 LoC (src/), 3,372 LoC (tests/)

---

## A. System Capability Map

### A1. What the benchmark *actually* does

| Layer | What exists | What runs |
|-------|-------------|-----------|
| **Environment** | `UrbanEnv` (Gymnasium): 500×500 OSM grids, 4-connected agent, 6-action Discrete space. 3 real-world tiles (Penteli, Piraeus, Downtown Athens) from OSM via `.npz` packs (heightmap, roads, landuse, risk, NFZ). | ✅ Fully functional |
| **Planners (7)** | A\*, Theta\*, AdaptiveA\*, D\*Lite, AD\*, DWA, MPPI | ✅ All run — but see §B1 |
| **Dynamics (7)** | FireSpread, TrafficFlow, MovingTarget, IntruderUAV, DynamicNFZ, AdversarialUAV, PopulationRisk | ✅ All active in dynamic track |
| **InteractionEngine** | Fire→NFZ expansion, fire→road closures, fire↔traffic feedback, smoke density | ✅ 3 causal couplings verified |
| **Metrics** | `operational.py` (safety/efficiency/feasibility), `comprehensive.py` (EpisodeMetrics/AggregateMetrics) | ✅ Computed and exported |
| **Visualization** | `OperationalRenderer` (1,600 LoC, 12-layer z-order), `StakeholderRenderer`, `BasemapStyle`, icon library (20 vector SVG-style icons) | ✅ GIF/frame export |
| **Scenarios** | 12 YAML configs (3 missions × {easy,medium,hard} + extras). Registry with `paper_track` tags (static/dynamic). | ✅ |
| **Updates** | `UpdateBus` (pub/sub), `ConflictDetector`, `PlannerAdapter` | ✅ Wired in env.step() |
| **V&V** | Solvability certificate (≥2 disjoint BFS paths), forced-replan certificate (placeholder), theoretical validation | ⚠️ See §B4 |
| **Paper pipeline** | `paper_best_paper_validation.py` (1,935 LoC), `export_paper_artifacts.py` (465 LoC) | ✅ CSV/LaTeX/figures |

### A2. Scenario inventory

| Scenario ID | Domain | Difficulty | Track | Map | Real incident |
|---|---|---|---|---|---|
| `gov_civil_protection_easy` | civil_protection | easy | static | penteli | 2018 Attica |
| `gov_civil_protection_medium` | civil_protection | medium | dynamic | penteli | 2018 Attica |
| `gov_civil_protection_hard` | civil_protection | hard | dynamic | penteli | 2018 Attica |
| `gov_maritime_domain_easy` | maritime_domain | easy | static | piraeus | — |
| `gov_maritime_domain_medium` | maritime_domain | medium | dynamic | piraeus | — |
| `gov_maritime_domain_hard` | maritime_domain | hard | dynamic | piraeus | — |
| `gov_critical_infrastructure_easy` | critical_infrastructure | easy | static | downtown | — |
| `gov_critical_infrastructure_medium` | critical_infrastructure | medium | dynamic | downtown | — |
| `gov_critical_infrastructure_hard` | critical_infrastructure | hard | dynamic | downtown | — |

Plus additional test/smoke scenarios. Total: 12 registered YAML configs.

---

## B. Critical P0 Findings

### B1. Planner Naming Honesty — ❌ FAIL (4/4 contested)

#### B1a. D\*Lite → NOT canonical

**File:** `src/uavbench/planners/dstar_lite.py` (30 LoC)

```python
class DStarLitePlanner(AdaptiveAStarPlanner):
    def __init__(self, heightmap, no_fly, config=None):
        super().__init__(heightmap, no_fly, config or AdaptiveAStarConfig(
            base_interval=6, lookahead_steps=8
        ))
```

**Canonical D\*Lite requires:** rhs values, key-ordered priority queue, backward search from goal to start, efficient incremental updates via `UpdateVertex()`. See: Koenig & Likhachev, *Artificial Intelligence*, 2002.

**What this is:** AdaptiveA\* with `base_interval=6`. It re-runs forward A\* from scratch at each replan step. No incremental data structures whatsoever.

**Verdict:** Mislabelled. Should be called `AdaptiveAStarV2` or `PeriodicReplanAStar(interval=6)`.

#### B1b. AD\* → NOT canonical

**File:** `src/uavbench/planners/ad_star.py` (30 LoC)

```python
class ADStarPlanner(AdaptiveAStarPlanner):
    def __init__(self, heightmap, no_fly, config=None):
        super().__init__(heightmap, no_fly, config or AdaptiveAStarConfig(
            base_interval=4, lookahead_steps=10
        ))
```

**Canonical AD\* requires:** Anytime D\* with ε-suboptimality bound, iterative refinement from ε_0→1.0, backward search, rhs values. See: Likhachev et al., *NeurIPS*, 2005.

**What this is:** AdaptiveA\* with `base_interval=4`. Same forward A\* from scratch. No suboptimality bounds.

**Verdict:** Mislabelled. Should be called `AdaptiveAStarV3` or `PeriodicReplanAStar(interval=4)`.

#### B1c. DWA → NOT canonical

**File:** `src/uavbench/planners/dwa.py` (66 LoC)

```python
def plan(self, start, goal, cost_map=None) -> PlanResult:
    # ...
    while cur != goal and expansions < self.cfg.max_steps:
        nbrs = self._neighbors(cur)
        def score(p):
            risk = float(cost_map[p[1], p[0]]) if cost_map is not None else 0.0
            revisit = 0.5 if p in visited else 0.0
            return self._heuristic(p, goal) + risk + revisit
        cur = min(nbrs, key=score)
```

**Canonical DWA requires:** Velocity space (v, ω), dynamic window determined by acceleration limits, trajectory simulation per (v, ω) sample, objective function over simulated arcs. See: Fox, Burgard & Thrun, *IEEE RA Magazine*, 1997.

**What this is:** Greedy 1-step local search picking the best 4-connected neighbor by `heuristic + risk + revisit_penalty`. Falls back to A\* on local minima. No velocity space, no trajectory simulation, no dynamics.

**Verdict:** Mislabelled. Should be called `GreedyLocalSearch` or `GreedyBestFirst`.

#### B1d. MPPI → Partially canonical (discrete-grid variant)

**File:** `src/uavbench/planners/mppi.py` (206 LoC)

The MPPI planner does sample N=256 trajectories over horizon=12, computes costs with obstacle/risk/goal penalties, weights by `exp(-cost/λ)`, and computes a weighted-average first displacement. This is the canonical MPPI algorithm (Williams et al., 2017).

**However:**
1. The weighted-average displacement is **discretized to 4 cardinal directions** via `dots = _DIRECTIONS @ weighted_disp; best_dir = argmax(dots)`. This collapses the continuous control output into the same 4-connected grid movement that every other planner uses.
2. Falls back to A\* after 5 stuck steps, and also at end of episode if goal not reached.
3. No replanning interface (`should_replan`/`replan` absent) — treated as static planner in the benchmark loop.

**Verdict:** Partially honest. The sampling/weighting core is canonical MPPI, but the output is discretized to 4 cardinal moves, eliminating the continuous-control advantage that motivates MPPI. Paper should note "grid-discretized MPPI" and acknowledge the continuous→discrete collapse.

### B1 Summary Table

| Name in code | Canonical? | What it actually is | Required fix |
|---|---|---|---|
| D\*Lite | ❌ No | AdaptiveA\*(interval=6) | Rename or reimplement |
| AD\* | ❌ No | AdaptiveA\*(interval=4) | Rename or reimplement |
| DWA | ❌ No | GreedyLocalSearch + A\* fallback | Rename or reimplement |
| MPPI | ⚠️ Partial | Canonical sampling but 4-direction discretization | Document "grid-MPPI" |
| A\* | ✅ Yes | Standard A\* with 4-connected, Manhattan heuristic | — |
| Theta\* | ✅ Yes | Any-angle with Bresenham line-of-sight, 8-connected | — |
| AdaptiveA\* | ✅ Yes | A\* with periodic replanning + dynamic obstacle merge | — |

---

### B2. Replanning Fairness — ❌ FAIL (structural asymmetry)

**Evidence:** `cli/benchmark.py` lines 466–713

```python
is_adaptive = hasattr(planner, "should_replan") and hasattr(planner, "replan")
```

| Planner | Has `should_replan`/`replan`? | Gets replanning? | Stuck handling |
|---|---|---|---|
| A\* | ❌ | ❌ (follows initial path blindly) | Break at `stuck_counter >= 10` |
| Theta\* | ❌ | ❌ (follows initial path blindly) | Break at `stuck_counter >= 10` |
| DWA | ❌ | ❌ (follows initial path blindly) | Break at `stuck_counter >= 10` |
| MPPI | ❌ | ❌ (follows initial path blindly) | Break at `stuck_counter >= 10` |
| AdaptiveA\* | ✅ | ✅ (full replan infrastructure) | Replans on stuck ≥ 3 |
| D\*Lite | ✅ (inherited) | ✅ (full replan infrastructure) | Replans on stuck ≥ 3 |
| AD\* | ✅ (inherited) | ✅ (full replan infrastructure) | Replans on stuck ≥ 3 |

**Impact:** In the dynamic track, A\*/Theta\*/DWA/MPPI plan once at t=0 and follow the path blindly. When fire/interdictions/NFZ block their path, they bump into walls until `stuck_counter >= 10` and then the episode terminates. Meanwhile AdaptiveA\*/D\*Lite/AD\* detect the blockage, merge dynamic obstacles, and re-route.

**This is not a fair comparison.** The paper cannot claim "A\* fails in dynamic scenarios" as evidence that adaptive replanning matters — it's actually evidence that the benchmark harness gives replanning *only* to one planner family. A fair comparison would give all planners the same benchmark-level replanning harness (replan on path invalidation) or document the asymmetry as a design choice.

**Recommendation:** Either:
1. Give all planners equal-opportunity benchmark-level replanning (detect path invalidation → re-invoke `planner.plan()` from current position), or
2. Clearly disclose in the paper: "A\*/Theta\*/DWA/MPPI are evaluated without replanning; AdaptiveA\*/D\*Lite/AD\* receive benchmark-level replanning support."

---

### B3. Forced Interdiction Bias — ❌ FAIL (planner-dependent placement)

**File:** `src/uavbench/envs/urban.py` lines 935–1000

Interdiction blocks are placed at **30% and 65% of a reference planner's path**:

```python
cfg.interdiction_reference_planner  # default: InterdictionReferencePlanner.THETA_STAR
```

The reference planner (Theta\* by default) computes a path at scenario reset. Cut points are placed at 30% and 65% of that reference path with configurable radii.

**Impact:** If the test planner's path happens to overlap with the reference planner's path at those cut points, the interdiction is maximally effective. If the test planner takes a different route, the interdiction may have no effect at all. This creates a systematic bias:
- **Theta\* is maximally penalized** (its own path is used as the reference)
- **A\* is likely penalized** (similar route on 4-connected vs 8-connected)
- **DWA/MPPI may escape** (greedy/stochastic paths may diverge from reference)

**Mitigation exists:** `cfg.interdiction_reference_planner` can be set per-scenario, and the `interdiction_hit_rate` metric is tracked. But the default (Theta\*) creates unfairness unless the paper explicitly acknowledges and controls for it.

**Recommendation:** Either:
1. Compute interdiction placement per-planner (use each planner's own initial path), or
2. Place interdictions at map-level chokepoints (topological analysis, not path-dependent), or
3. Report `interdiction_hit_rate` for each planner and discuss the bias explicitly.

---

### B4. Cosmetic / Non-Functional Features — ⚠️ WARNING

#### B4a. `comms_dropout_prob` — Cosmetic in benchmark loop

**Defined in:** `missions/spec.py` (DifficultyKnobs), scenario YAML `extra:` section.
**Used in:** `missions/engine.py` line 172 — sets `events["comms_dropout"] = True` when random draw < probability.

**But:** The benchmark loop in `cli/benchmark.py` **never imports or uses MissionEngine**. The `comms_dropout_prob` value is only used for:
1. A label on the renderer (`"COMMS DROP 15%"`) — line 424–426
2. Nothing else. The replanning logic in the benchmark loop does NOT check comms dropout events. The planner always receives fresh `get_dynamic_state()` data regardless.

**Verdict:** The comms dropout is **cosmetic-only in the main benchmark path**. It only has runtime effect in the separate `missions/runner.py` and `missions/runner_v2.py` pipelines, which are NOT used by `cli/benchmark.py`.

#### B4b. `comms_latency_steps` — Not implemented anywhere

**Defined in:** `missions/spec.py` DifficultyKnobs.
**Used:** Only in the DifficultyKnobs constructor presets. No runtime code checks `comms_latency_steps` to delay risk map updates or stale observations.

**Verdict:** Pure schema decoration. Not implemented.

#### B4c. `solvability_cert_ok` — Schema field, never set to True at runtime

**Defined in:** `scenarios/schema.py` line 156: `solvability_cert_ok: bool = False`
**Runtime usage:** Neither `registry.py` nor `urban.py` nor `benchmark.py` ever calls `check_solvability_certificate()` to set this flag. The flag stays at its default `False` for all scenarios.

**The `check_solvability_certificate()` function exists** in `benchmark/solvability.py` and works correctly (BFS + node-disjoint path finding). But it is never called in the production path. It is only called in tests.

**Verdict:** The docs claim "All 34 scenarios have `solvability_cert_ok=True`" but the runtime never sets this. The actual certificate function is sound; it's just not wired into the scenario loader.

#### B4d. `check_forced_replan_certificate()` — Placeholder stub

**File:** `benchmark/solvability.py` lines 224–243:

```python
def check_forced_replan_certificate(...) -> Tuple[bool, str]:
    # Placeholder: in full implementation, this would simulate dynamics forward
    return True, "forced_replan check deferred to runtime validation"
```

**Verdict:** Always returns `True`. No actual validation.

---

### B5. Export / Provenance Integrity — ✅ ADEQUATE

**What exists:**
- `export_paper_artifacts.py`: Git commit hash (`_git_commit()`), scenario YAML SHA-256 checksums (`_scenario_checksums()`), bootstrap CIs, paired statistics.
- `operational_renderer.py` line 200: Version string with git hash in rendered frames.
- `schema.py` lines 128–133: `incident_name`, `incident_year`, `incident_summary`, `incident_refs` fields per scenario — grounding scenarios in real events.

**What's missing:**
- No Python version / package version lockfile hash in export.
- No random seed chain verification (seeds are set but not cryptographically committed).
- No environment (OS, NumPy version) fingerprint in artifacts.

**Verdict:** Adequate for a systems paper. For a dataset/benchmark track (NeurIPS D&B), would want full environment fingerprint.

---

## C. Novelty Features Assessment

### C1. Comms Dropout / Latency — ❌ NOT FUNCTIONAL in benchmark

As shown in §B4a–B4b:
- `comms_dropout_prob` is cosmetic in the benchmark loop (label only).
- `comms_latency_steps` is completely unimplemented.
- The separate `missions/runner.py` pipeline uses `MissionEngine` which does process comms dropout, but this pipeline is NOT used for paper results.

**Recommendation:** Either wire comms dropout into the benchmark loop (skip replanning when dropout fires) or remove the claim from the paper.

### C2. GNSS Drift / Localization Noise — ❌ NOT PRESENT

grep for `gnss|gps_noise|position_noise|localization` found only `"localization_time": step` in `missions/runner.py` and `runner_v2.py` — which is just a timestamp field, not a noise model.

No GNSS degradation, position uncertainty, or localization error model exists in the codebase.

### C3. Interaction Engine — ✅ GENUINE NOVELTY

The `InteractionEngine` (`dynamics/interaction_engine.py`) implements 3 verified causal couplings:
1. **Fire → NFZ expansion:** Fire masks expand dynamic NFZ boundaries
2. **Fire → Road closures:** Fire proximity triggers traffic closures on road cells
3. **Fire ↔ Traffic feedback:** Congestion increases fire risk (positive feedback loop)

These run every step in `urban.py` step function. Metrics are tracked:
- `interaction_fire_nfz_overlap_ratio`
- `interaction_fire_road_closure_rate`
- `interaction_congestion_risk_corr`
- `dynamic_block_entropy` / `dynamic_block_entropy_env`

**Verdict:** This is a genuine and well-implemented novelty. The causal interaction model is the strongest differentiator vs. existing benchmarks.

---

## D. Mission Identity Beyond Start→Goal

### D1. Mission framework architecture

The codebase has a sophisticated mission framework:
- **3 mission types:** Civil Protection, Maritime Domain, Critical Infrastructure
- **`MissionSpec`:** Labels, difficulty knobs, initial tasks, product types, utility decay
- **`TaskSpec`:** POI positions, weights, time windows, service times, categories
- **`DifficultyKnobs`:** num_tasks (4/6/8), injection rates, dynamics intensity, comms parameters
- **`MissionEngine`:** Task injection scheduler, completion tracking, risk/energy budgets, violation counting
- **`MissionProduct`:** Operational deliverables (GeoJSON, CSV reports)
- **3 builders:** `build_civil_protection()`, `build_maritime_domain()`, `build_critical_infrastructure()`

### D2. Is the mission framework used in the benchmark? — ❌ NO

**Critical finding:** `cli/benchmark.py` does **not import any mission module**. The benchmark loop:
1. Loads scenario YAML → creates `ScenarioConfig`
2. Creates `UrbanEnv` with the config
3. Calls `planner.plan(start, goal)` → follow path → done

The missions framework (`builders.py`, `engine.py`, `spec.py`) is only used by:
- `missions/runner.py` — separate mission-oriented runner
- `missions/runner_v2.py` — v2 of the same
- Unit tests

**The benchmark produces start→goal navigation results.** The multi-task, time-windowed, service-time, injection-scheduled mission layer exists but is **not exercised by the paper pipeline**.

**What the benchmark actually evaluates:** "Can planner X navigate from (50,50) to (450,450) through dynamic obstacles on map Y?"

**Recommendation:** Either:
1. Integrate `MissionEngine` into the benchmark loop so that multi-POI, time-windowed tasks actually affect the evaluation, or
2. Remove mission-layer claims from the paper and frame honestly as "single-query navigation in dynamic environments."

---

## E. Guardrail Confound Control

### E1. What the guardrail does

**File:** `urban.py` `_enforce_feasibility_guardrail()` (lines 1086–1200)

A 3-depth relaxation cascade triggered every step when the agent cannot reach the goal:

| Depth | Action | What it clears |
|---|---|---|
| 0 | No action | Path is feasible |
| 1 | Clear forced interdictions | `forced_block_mask[:] = False` |
| 2 | Reduce NFZ expansion + clear closures | NFZ rate ×0.7, radii −2, traffic closures cleared |
| 3 | Emergency corridor fallback | Open corridor mask, hard deconfliction (all blocks cleared) |

### E2. Is it controllable for ablation? — ✅ YES

```python
# cli/benchmark.py
if variant == "no_guardrail":
    extra["disable_feasibility_guardrail"] = True
```

When disabled, the environment still tracks reachability but does not relax obstacles. The flag `guardrail_disabled: True` is set in the status dict.

### E3. Is activation tracked? — ✅ YES

```python
"guardrail_activation_count": int(guardrail_activation_count),
"guardrail_activation_rate": float(guardrail_activation_count / max(episode_steps, 1)),
"corridor_fallback_count": int(corridor_fallback_count),
"relaxation_magnitude": float(relaxation_magnitude),
"guardrail_depth_distribution": _compute_guardrail_depth_distribution(events),
```

### E4. Is it a confound? — ⚠️ POTENTIALLY

The guardrail differentially benefits adaptive planners (who replan after guardrail clears obstacles) vs static planners (who still follow their original, now-invalidated path). But since the guardrail is ablatable via `protocol_variant="no_guardrail"`, this can be controlled experimentally.

**Recommendation:** Report results with and without guardrail (the ablation infrastructure exists). If guardrail activation rate > 5%, flag it in the paper.

---

## F. Patch Series Plan

### Priority 0 — Must fix before submission

| # | What | Where | Fix |
|---|---|---|---|
| F1 | **Rename D\*Lite** | `planners/dstar_lite.py`, `planners/__init__.py` | Rename to `AdaptiveAStarV2` or implement canonical D\*Lite |
| F2 | **Rename AD\*** | `planners/ad_star.py`, `planners/__init__.py` | Rename to `AdaptiveAStarV3` or implement canonical AD\* |
| F3 | **Rename DWA** | `planners/dwa.py`, `planners/__init__.py` | Rename to `GreedyLocal` or implement canonical DWA |
| F4 | **Document MPPI discretization** | `planners/mppi.py` docstring, paper | Note "grid-discretized MPPI (4-cardinal)" |
| F5 | **Disclose replanning asymmetry** | Paper methodology section | Table showing which planners get replanning |
| F6 | **Disclose interdiction bias** | Paper methodology section | Report `interdiction_hit_rate` per planner |

### Priority 1 — Should fix before submission

| # | What | Where | Fix |
|---|---|---|---|
| F7 | **Wire solvability cert** | `scenarios/loader.py` or `registry.py` | Call `check_solvability_certificate()` at load time, set flag |
| F8 | **Remove or implement comms dropout** | `cli/benchmark.py` | Either skip replanning on dropout, or remove from paper claims |
| F9 | **Remove `comms_latency_steps`** | `missions/spec.py` | Not implemented; remove from schema or implement |
| F10 | **Fix `check_forced_replan_certificate`** | `benchmark/solvability.py` | Implement or remove the placeholder stub |

### Priority 2 — Nice to have

| # | What | Where | Fix |
|---|---|---|---|
| F11 | **Equal-opportunity replanning** | `cli/benchmark.py` | Give all planners benchmark-level replan-on-invalidation |
| F12 | **Per-planner interdiction** | `envs/urban.py` | Compute interdiction cut points per planner's own path |
| F13 | **Integrate mission framework** | `cli/benchmark.py` | Use `MissionEngine` for multi-POI evaluation |
| F14 | **Environment fingerprint** | `export_paper_artifacts.py` | Add Python/NumPy/OS version to artifacts |
| F15 | **GNSS noise model** | `envs/urban.py` | If claimed, implement position noise |

---

## G. Paper Artifact Checklist

### G1. What the export pipeline produces

**Script:** `scripts/paper_best_paper_validation.py` (1,935 LoC)  
**Script:** `scripts/export_paper_artifacts.py` (465 LoC)

| Artifact | File | Status |
|---|---|---|
| Raw episode results | `episodes.jsonl` | ✅ |
| Aggregate CSV | `aggregates.csv` | ✅ |
| LaTeX table | `tables/*.tex` | ✅ |
| Bootstrap CIs | computed inline | ✅ |
| Paired statistics | `paired_stats.json` | ✅ |
| Git commit hash | in manifest | ✅ |
| Scenario checksums | SHA-256 per YAML | ✅ |
| Ablation variants | `{no_interactions, no_forced_breaks, no_guardrail, risk_only, blocking_only}` | ✅ |
| Stress-α curves | via `_apply_stress_alpha()` | ✅ |
| Sensitivity heatmaps | `paper_best_paper_validation.py` | ✅ |
| Figures (PNG) | `figures/` directory | ✅ |
| Reproducibility manifest | `reproducibility_manifest.json` | ✅ |

### G2. What's missing for NeurIPS D&B

| Missing | Impact |
|---|---|
| Datasheet for Datasets (Gebru et al.) | Required for D&B track |
| Environment fingerprint (Python, NumPy, OS versions) | Reproducibility |
| Pre-computed canonical results for unit testing | Regression testing |
| Docker/container for exact reproduction | Reproducibility |
| License compatibility audit (OSM data vs code license) | Legal |

### G3. Ablation variant coverage

| Variant | What it disables | Implemented? |
|---|---|---|
| `default` | Nothing | ✅ |
| `no_interactions` | InteractionEngine couplings | ✅ |
| `no_forced_breaks` | Forced interdictions, event times | ✅ |
| `risk_only` | Blocking + interdictions, keeps risk maps | ✅ |
| `blocking_only` | Population risk, adversarial UAV | ✅ |
| `no_guardrail` | Feasibility guardrail | ✅ |

**Verdict:** Ablation infrastructure is solid and well-designed. 6 variants cover the major axes.

---

## Summary Verdict

### Strengths (what reviewers will praise)
1. **InteractionEngine** — Genuine novelty. Causal coupling between dynamics is unique in the UAV benchmark space.
2. **Real-world grounding** — OSM tiles from actual Greek locations, incident provenance metadata.
3. **Ablation infrastructure** — 6 protocol variants, stress-α curves, sensitivity analysis.
4. **Guardrail system** — 3-depth feasibility cascade with full tracking/ablation.
5. **Visualization quality** — Publication-ready 12-layer operational renderings with GIF export.
6. **Reproducibility pipeline** — Git hash, YAML checksums, bootstrap CIs, paired statistics.

### Weaknesses (what reviewers will attack)

| Severity | Finding | §Ref |
|---|---|---|
| 🔴 P0 | 3 of 7 planners are mislabelled (D\*Lite, AD\*, DWA) | B1 |
| 🔴 P0 | Replanning is structurally asymmetric (only adaptive planners get it) | B2 |
| 🔴 P0 | Interdiction placement biases against reference planner | B3 |
| 🟡 P1 | Comms dropout is cosmetic in benchmark loop | B4a, C1 |
| 🟡 P1 | Solvability certificate never runs in production | B4c |
| 🟡 P1 | Mission framework not used in benchmark | D2 |
| 🟢 P2 | MPPI discretization not documented | B1d |
| 🟢 P2 | Forced-replan certificate is a stub | B4d |
| 🟢 P2 | No GNSS noise model | C2 |

### Bottom line

The benchmark has **strong infrastructure** (environment, dynamics, interactions, visualization, ablations) but **3 critical honesty issues** (planner naming, replanning fairness, interdiction bias) that would likely result in **major revision or reject** at top venues unless addressed. The fixes are straightforward: rename planners, document the replanning protocol, and address interdiction fairness. The non-functional features (comms, solvability, missions) should either be wired in or their claims removed.

The **InteractionEngine** is the paper's strongest contribution and should be front-and-centre. The planner comparison is secondary and needs the most cleanup.
