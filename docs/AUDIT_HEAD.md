# AUDIT_HEAD.md — HEAD Reality Check (v2)

**Date:** 2025-02-20  
**Commit:** HEAD (post-sprint-3 — integrity hardening)  
**Auditor:** Automated V&V audit agent  
**Scope:** Full re-audit from scratch — zero trust in prior summaries  
**Baseline:** 268 tests passing (37 s)

---

## 1. Codebase Census

| Category | Files | Lines of Code |
|---|---|---|
| `src/uavbench/` | 63 `.py` | **19,278** |
| `tests/` | 22 `.py` | **4,659** |
| **Total** | 85 | **23,937** |

### Key source files (audit-critical)

| File | LoC | Role |
|---|---|---|
| `cli/benchmark.py` | 1,884 | Main CLI: `run_planner_once`, `run_dynamic_episode`, `run_mission_episode`, `main()` |
| `envs/urban.py` | ~1,400 | Gymnasium UrbanEnv — 500×500 OSM grid, Discrete(6) |
| `visualization/operational_renderer.py` | ~1,930 | Operational render pipeline |
| `planners/dstar_lite_real.py` | 531 | True D* Lite (Koenig & Likhachev 2002) |
| `planners/adaptive_astar.py` | 241 | AdaptiveAStarPlanner (replan heuristics) |
| `planners/__init__.py` | 57 | Registry: 11 keys (7 honest + 4 legacy aliases) |
| `scenarios/schema.py` | ~270 | ScenarioConfig dataclass + realism fields |
| `scenarios/loader.py` | ~140 | YAML → ScenarioConfig parser |
| `missions/engine.py` | 358 | MissionEngine (task tracking, injection, utility) |
| `missions/builders.py` | 335 | 3 mission builders with injection schedules |

---

## 2. Planner Suite — Naming Honesty Verified

### 2.1 Registry (11 keys)

| Registry Key | Class | File | Honest? |
|---|---|---|---|
| `astar` | `AStarPlanner` | `planners/astar.py` | ✅ |
| `theta_star` | `ThetaStarPlanner` | `planners/theta_star.py` | ✅ |
| `periodic_replan` | `PeriodicReplanPlanner` | `planners/dstar_lite.py` | ✅ |
| `aggressive_replan` | `AggressiveReplanPlanner` | `planners/ad_star.py` | ✅ |
| `greedy_local` | `GreedyLocalPlanner` | `planners/dwa.py` | ✅ |
| `grid_mppi` | `GridMPPIPlanner` | `planners/mppi.py` | ✅ |
| `incremental_dstar_lite` | `DStarLiteRealPlanner` | `planners/dstar_lite_real.py` | ✅ |
| `dstar_lite` | → `PeriodicReplanPlanner` | alias | ⚠️ Legacy |
| `ad_star` | → `AggressiveReplanPlanner` | alias | ⚠️ Legacy |
| `dwa` | → `GreedyLocalPlanner` | alias | ⚠️ Legacy |
| `mppi` | → `GridMPPIPlanner` | alias | ⚠️ Legacy |

### 2.2 Algorithmic Differentiation

| Planner | Search | Replan Strategy | True Incremental? |
|---|---|---|---|
| `astar` | A* | None (single-shot) | N/A |
| `theta_star` | Theta* (any-angle) | None (single-shot) | N/A |
| `periodic_replan` | A* from scratch | Every 6 steps + proximity trigger | ❌ |
| `aggressive_replan` | A* from scratch | Every 4 steps + proximity trigger | ❌ |
| `greedy_local` | Greedy local-best | Per-step (reactive) | ❌ |
| `grid_mppi` | Sampling-based (MPPI) | Per-step (trajectory rollout) | ❌ |
| `incremental_dstar_lite` | D* Lite (LPA*-based) | Incremental g/rhs updates on edge changes | ✅ |

**Key finding:** `DStarLiteRealPlanner` (531 LoC) implements true Koenig & Likhachev 2002 with priority queue, `g`/`rhs` values, `update_vertex()`, `compute_shortest_path()`, and `notify_edge_changes()`. It subclasses `BasePlanner`. This is the **only** genuinely incremental planner.

### 2.3 Architectural Note

`AdaptiveAStarPlanner` (adaptive_astar.py:30) does **NOT** subclass `BasePlanner` — it is a standalone class with its own `plan()`/`replan()`/`should_replan()` interface. `PeriodicReplanPlanner` and `AggressiveReplanPlanner` both inherit from it. The benchmark loop duck-types via `hasattr(planner, 'replan')`. Functional but inconsistent.

---

## 3. P0 Bugs Found & Fixed This Sprint

### 3.1 🔴 P0-FIX: Realism Knobs Were Dead at Runtime

**Bug:** All 9 scenario YAMLs stored `comms_dropout_prob` and `comms_latency_steps` inside the `extra:` block. The loader (`scenarios/loader.py:120`) reads them from **top-level** YAML keys:

```python
comms_dropout_prob=float(data.get("comms_dropout_prob", 0.0))
```

Since the keys were nested under `extra:`, `data.get()` returned `0.0` for every scenario. **All realism features (comms dropout, constraint latency, GNSS noise) were silently disabled at runtime** — even though scenario cards and docs claimed 15% dropout on hard scenarios.

**Fix applied to all 9 YAMLs:**
- Moved `comms_dropout_prob` to top-level
- Renamed `comms_latency_steps` → `constraint_latency_steps` (matching schema field name)
- Added `gnss_noise_sigma` (was entirely missing from YAMLs)

**Values per difficulty tier (verified in all 9 files):**

| Field | Easy | Medium | Hard |
|---|---|---|---|
| `comms_dropout_prob` | 0.0 | 0.05 | 0.15 |
| `constraint_latency_steps` | 0 | 2 | 4 |
| `gnss_noise_sigma` | 0.0 | 1.0 | 2.0 |

**Files changed:**
- `src/uavbench/scenarios/configs/gov_civil_protection_{easy,medium,hard}.yaml`
- `src/uavbench/scenarios/configs/gov_maritime_domain_{easy,medium,hard}.yaml`
- `src/uavbench/scenarios/configs/gov_critical_infrastructure_{easy,medium,hard}.yaml`

### 3.2 🔴 P0-FIX: Interdiction Reference Planner Defaulted to theta_star

**Bug:** `scenarios/schema.py:145` defines:
```python
interdiction_reference_planner: InterdictionReferencePlanner = InterdictionReferencePlanner.THETA_STAR
```

No YAML overrode this default, so all scenarios used `theta_star` as interdiction reference — contradicting docs that claim `astar`. While interdiction now uses BFS shortest-path (planner-agnostic), the logged `reference_planner` field was misleading.

**Fix:** Added `interdiction_reference_planner: astar` to all 9 YAMLs. Updated test assertion in `tests/test_fair_protocol.py`.

### 3.3 🔴 P0-FIX: `run_mission_episode()` Not Callable from CLI

**Bug:** `run_mission_episode()` existed at `cli/benchmark.py:977` (400+ LoC, fully implemented with MissionEngine integration) but `main()` had no way to invoke it. The dispatch at line ~1700 only chose between `run_dynamic_episode` and `run_planner_once`.

**Fix:** Added two CLI arguments:
- `--mode nav|mission` (default: `nav`) — selects execution mode
- `--policy greedy|lookahead` (default: `greedy`) — mission-layer task-selection policy

When `--mode mission`, dispatch calls `run_mission_episode()` with the selected policy. Backward compatible: `--mode nav` (default) preserves existing behavior.

---

## 4. Fairness Verification

### 4.1 Replanning Fairness

All planners enter the same replanning block (`cli/benchmark.py`). Two paths:
- **Native replan:** `planner.replan()` for adaptive planners (`hasattr(planner, 'replan')`)
- **Harness replan:** `planner.plan()` from scratch for non-adaptive planners

Both paths subject to identical `plan_budget_dynamic_ms` enforcement.

### 4.2 Interdiction Fairness

Interdiction placement uses BFS shortest-path (`envs/urban.py:929-1049`) — completely planner-agnostic. No planner module imported for interdiction decisions.

### 4.3 Realism Fairness

All planners face identical:
- Constraint latency (delayed dynamic state via FIFO buffer)
- Comms dropout (stale snapshot with probability `comms_dropout_prob`)
- GNSS noise (Gaussian perturbation on planner's perceived position)

These are applied in the benchmark loop *before* the planner sees state, ensuring equal degradation.

---

## 5. Scenario Registry Verification

**9 YAML scenarios** — 3 missions × 3 difficulties:

| # | Scenario ID | Track | Realism Active? |
|---|---|---|---|
| 1 | `gov_civil_protection_easy` | static | ✅ (0/0/0) |
| 2 | `gov_civil_protection_medium` | dynamic | ✅ (0.05/2/1.0) |
| 3 | `gov_civil_protection_hard` | dynamic | ✅ (0.15/4/2.0) |
| 4 | `gov_maritime_domain_easy` | static | ✅ (0/0/0) |
| 5 | `gov_maritime_domain_medium` | dynamic | ✅ (0.05/2/1.0) |
| 6 | `gov_maritime_domain_hard` | dynamic | ✅ (0.15/4/2.0) |
| 7 | `gov_critical_infrastructure_easy` | static | ✅ (0/0/0) |
| 8 | `gov_critical_infrastructure_medium` | dynamic | ✅ (0.05/2/1.0) |
| 9 | `gov_critical_infrastructure_hard` | dynamic | ✅ (0.15/4/2.0) |

**Format:** (dropout / latency_steps / gnss_sigma)

All 9 have `interdiction_reference_planner: astar`.

---

## 6. CLI Capabilities (post-fix)

| Mode | Flag | Function | Status |
|---|---|---|---|
| Static navigation | `--mode nav` (default) | `run_planner_once()` | ✅ |
| Dynamic navigation | `--mode nav` + dynamic scenario | `run_dynamic_episode()` | ✅ |
| Multi-task mission | `--mode mission` | `run_mission_episode()` | ✅ NEW |
| Paper protocol | `--paper-protocol` | Fixed scoring labels | ✅ |
| Track filter | `--track static\|dynamic` | Filter by paper track | ✅ |
| Ablation | `--protocol-variant` | 6 variants | ✅ |

---

## 7. Remaining Gaps (Ordered by Severity)

| # | Severity | Gap | Status |
|---|---|---|---|
| G1 | 🟡 MEDIUM | `AdaptiveAStarPlanner` not a `BasePlanner` subclass | Known, functional via duck-typing |
| G2 | 🟡 MEDIUM | `PlannerAdapter` not the primary code path | Benchmark uses own replan logic |
| G3 | 🟢 LOW | Legacy alias keys still in registry | Backward compat, documented |
| G4 | 🟢 INFO | `MissionRunnerV2` standalone runner exists parallel to `run_mission_episode` | Two mission execution paths |

**All P0 bugs are now fixed.** No critical gaps remain.

---

## 8. Test Status

| Suite | Count | Status |
|---|---|---|
| Full test suite | 268 | ✅ All passing (37 s) |
| `test_fair_protocol.py` | 2 | ✅ Confirms `astar` reference |
| `test_planning_correctness.py` | ~50 | ✅ |
| `test_mission_bank.py` | ~30 | ✅ |
| `test_dstar_lite_real.py` | 28 | ✅ |

---

## 9. Files Changed This Sprint

| File | Change |
|---|---|
| `scenarios/configs/gov_civil_protection_{easy,medium,hard}.yaml` | P0: realism knobs to top-level + astar + gnss |
| `scenarios/configs/gov_maritime_domain_{easy,medium,hard}.yaml` | P0: realism knobs to top-level + astar + gnss |
| `scenarios/configs/gov_critical_infrastructure_{easy,medium,hard}.yaml` | P0: realism knobs to top-level + astar + gnss |
| `cli/benchmark.py` | P0: `--mode mission` + `--policy` CLI wiring |
| `tests/test_fair_protocol.py` | Updated assertion: theta_star → astar |
