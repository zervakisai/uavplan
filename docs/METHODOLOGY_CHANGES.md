# UAVBench — Post-Audit Methodology Changes

This document summarises every code change made in the audit patch series,
cross-referenced to the AUDIT_HEAD.md findings, for inclusion in the paper's
methodology section or supplementary material.

---

## Commit 1: Planner Naming Honesty (P0-B1)

**Finding:** Four planner classes carried names of canonical algorithms they do
not implement (D\* Lite, AD\*, DWA, MPPI).

**Fix:**

| Old Name          | New (Honest) Name          | What It Actually Does                        | Canonical Ref (NOT implemented)          |
|-------------------|----------------------------|----------------------------------------------|------------------------------------------|
| `DStarLitePlanner`| `PeriodicReplanPlanner`    | Full A\* replan every 6 steps                | Koenig & Likhachev 2002, D\* Lite       |
| `ADStarPlanner`   | `AggressiveReplanPlanner`  | Full A\* replan every 4 steps                | Likhachev et al. 2005, AD\*             |
| `DWAPlanner`      | `GreedyLocalPlanner`       | Greedy 1-step local + A\* fallback           | Fox, Burgard & Thrun 1997, DWA          |
| `MPPIPlanner`     | `GridMPPIPlanner`          | MPPI sampling → 4-cardinal discretization    | Williams et al. 2017, MPPI              |

- Legacy class names kept as aliases (`DStarLitePlanner = PeriodicReplanPlanner`).
- Legacy registry keys kept alongside honest keys.
- Each class docstring now cites the canonical paper it does NOT implement.
- **39 automated tests** verify aliases, registry keys, and docstring honesty.

**Files changed:** `planners/dstar_lite.py`, `planners/ad_star.py`,
`planners/dwa.py`, `planners/mppi.py`, `planners/__init__.py`,
`tests/test_planner_naming_honesty.py`, `tests/test_planner_suite_expanded.py`.

---

## Commit 2: Replanning Fairness (P0-B2)

**Finding:** The benchmark harness only offered replanning infrastructure to
"adaptive" planners (those with `.should_replan()` + `.replan()` methods).
Static planners (A\*, Theta\*) were forced to follow their initial path; if
blocked, they hit `stuck_counter >= 10` and terminated.  This made A\*/Theta\*
appear artificially weak on dynamic scenarios.

**Fix:** The replanning gate in `benchmark.py` now applies to **all planners**:

- **Adaptive planners** (PeriodicReplan, AggressiveReplan): use their native
  `.replan()` method (unchanged behaviour).
- **Non-adaptive planners** (A\*, Theta\*, GreedyLocal, GridMPPI): receive
  **harness-level replanning** — `planner.plan(current_pos, goal)` is called
  when the same triggers fire (path invalidation, cadence, stuck, forced event).

This ensures a fair comparison: every planner has the same opportunity to
recover from dynamic obstacles.

A new result field `replan_mode` records `"native"` or `"harness_replan"` per
episode, enabling the paper to report which mode was used.

**Files changed:** `cli/benchmark.py`.

---

## Commit 3: Interdiction Fairness (P0-B3)

**Finding:** Path interdiction placement used a reference planner (default
Theta\*) to compute the path to block.  This biased interdictions specifically
against the reference planner's path.

**Fix:** Interdiction placement now uses a **planner-agnostic BFS shortest
path** on the traversable grid (buildings + NFZ removed).  BFS is the
graph-theoretic shortest path on an unweighted 4-connected grid — it is not
tied to any named planning algorithm.

- The `interdiction_reference_planner` config field is still accepted for
  backward compatibility but is **no longer used** for placement.
- All interdiction events now log `reference_planner: "bfs_shortest_path"`.
- The `ThetaStarPlanner` import was removed from `urban.py`.

**Files changed:** `envs/urban.py`, `tests/test_fair_protocol.py`.

---

## Commits 4–6: P1 Realism Features

### 4. Constraint Update Latency

**Feature:** `constraint_latency_steps` (default 0) introduces a FIFO delay
buffer.  When > 0, the planner sees a dynamic snapshot that is *k* steps old,
simulating real-world sensor/comms processing latency.

### 5. Communications Dropout

**Feature:** `comms_dropout_prob` (default 0.0) is the per-step probability
that the planner receives a *stale* dynamic snapshot (the last successfully
received one) instead of the current one.  Models packet-loss degradation.

### 6. GNSS Interference

**Feature:** `gnss_noise_sigma` (default 0.0) adds Gaussian noise (σ in grid
cells) to the agent position observed by the planner.  Simulates GNSS
degradation in urban canyons.  Noise is clamped to grid bounds.

**All three** use a deterministic RNG seeded from `seed ^ 0x50_1A_C1_00` to
preserve reproducibility.

**Schema fields + validation** added to `ScenarioConfig`.
**Loader** updated to parse new fields from YAML.
**10 automated tests** cover schema validation + benchmark result presence.

**Files changed:** `scenarios/schema.py`, `scenarios/loader.py`,
`cli/benchmark.py`, `tests/test_p1_realism.py`.

---

## Test Suite Summary

| Metric                  | Before Audit | After Audit |
|-------------------------|-------------|-------------|
| Total tests             | 292         | 345         |
| Passing                 | 292         | 345         |
| New test files           | —           | 2 (`test_planner_naming_honesty.py`, `test_p1_realism.py`) |
| Updated test files       | —           | 2 (`test_planner_suite_expanded.py`, `test_fair_protocol.py`) |

---

## Paper Section Recommendations

1. **Methodology §Planners:** Replace "D\* Lite", "AD\*", "DWA" with the
   honest names.  Cite the canonical algorithms as "not implemented" baselines.
2. **Methodology §Evaluation Protocol:** Note that all planners receive
   identical replanning opportunities via the harness (with replan_mode field
   distinguishing native vs harness-level).
3. **Methodology §Interdiction:** State that interdiction placement uses the
   BFS shortest path (planner-agnostic).
4. **Limitations / Future Work:** Mention the three P1 realism knobs as
   available for ablation studies (constraint latency, comms dropout, GNSS
   noise), with defaults at 0 (baseline = ideal conditions).
