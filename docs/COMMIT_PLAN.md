# UAVBench — Ordered Commit Plan

> **Purpose:** Atomic, reviewable commits that bring HEAD from prior audit state to paper-ready.  
> **Date:** 2025-07-14  
> **Test baseline:** 388 tests, all passing  
> **Rule:** Each commit must leave tests green. No commit > 600 LoC diff.

---

## Commit 1: `feat(missions): wire MissionEngine into CLI`

**Files:**
- `M` `src/uavbench/cli/benchmark.py` — add `run_mission_episode()` + imports
- `A` `tests/test_mission_episode.py` — 12 tests

**What:** Adds `run_mission_episode()` as a parallel entry point to `run_dynamic_episode()`.
Integrates `MissionEngine` (tasks, scoring, products) with the existing `UrbanEnv` dynamic loop.
Returns a superset of dynamic episode keys plus `mission_score`, `task_completion_rate`, `task_log`, etc.

**Backward compat:** ✅ `run_dynamic_episode()` and `run_planner_once()` untouched.  
**Tests after:** 357 → 369 (12 new).

---

## Commit 2: `feat(planners): add true D* Lite incremental planner`

**Files:**
- `A` `src/uavbench/planners/dstar_lite_real.py` — 532 lines, `DStarLiteRealPlanner`
- `M` `src/uavbench/planners/__init__.py` — register `incremental_dstar_lite`
- `A` `tests/test_dstar_lite_real.py` — 28 tests across 9 classes

**What:** Implements D* Lite (Koenig & Likhachev 2002) with true incremental g/rhs updates,
priority queue with lazy deletion, and `notify_edge_changes()` for dynamic obstacle integration.
Subclasses `BasePlanner`. Provides harness-compatible `should_replan()` / `replan()` API.

**Backward compat:** ✅ New registry key only. Existing planners untouched.  
**Tests after:** 369 → 388 (28 new + auto-parametrized picks up new key in existing tests).

---

## Commit 3: `test: update registry assertions for 11 planners`

**Files:**
- `M` `tests/test_planner_suite_expanded.py` — update `test_six_planners_registered()` to 11
- `M` `tests/test_planner_naming_honesty.py` — add `DStarLiteRealPlanner` import + `incremental_dstar_lite` to registry key test

**What:** Existing tests that assert exact planner count / registry keys are updated to
include the new `incremental_dstar_lite` entry.

**Note:** Can be squashed into Commit 2 if preferred.

---

## Commit 4: `docs: add AUDIT_HEAD.md — codebase re-audit`

**Files:**
- `A` `docs/AUDIT_HEAD.md`

**What:** Gap analysis of HEAD. LoC counts, test counts, verified patches from prior sprint,
remaining gaps identified (MissionEngine unwired, no true incremental planner — both now fixed).

---

## Commit 5: `docs: add PROJECT_CAPABILITIES.md — system capability map`

**Files:**
- `A` `docs/PROJECT_CAPABILITIES.md`

**What:** Every testable capability with `file:line` citations and status indicators (✅/⚠️/🔲).
Covers: environment, planners, dynamics, missions, scenarios, benchmark infrastructure,
visualization, metrics, and updates subsystems.

---

## Commit 6: `docs: add SCENARIO_CARDS.md — incident-grounded scenario storytelling`

**Files:**
- `A` `docs/SCENARIO_CARDS.md`

**What:** All 9 scenarios documented with:
- Real-world incident provenance (2018 Attica Wildfire, 2017 Agia Zoni II, 2021 Athens Bomb Threat)
- YAML field mapping with exact parameter values per difficulty tier
- Causal dynamics chain (fire→NFZ, spill→vessel, threat→restriction zones)
- Expected planner behavior under each scenario

---

## Commit 7: `docs: upgrade PAPER_PROTOCOL.md to v2.0`

**Files:**
- `M` `docs/PAPER_PROTOCOL.md` — full rewrite (v1 preserved as `PAPER_PROTOCOL_v1.md`)

**What:** Complete evaluation protocol for ICRA/IROS/RA-L/NeurIPS D&B with:
- 8 falsifiable hypotheses (H1–H8) with claims-evidence map
- Statistical analysis protocol (Shapiro-Wilk → parametric/non-parametric, Holm-Bonferroni)
- Metrics taxonomy (primary, fairness, interaction, mission)
- 5 ablation variants with hypothesis linkage
- Paper table templates
- Reproducibility checklist
- Reviewer FAQ

---

## Commit Dependency Graph

```
C1 (missions)  →  independent
C2 (D* Lite)   →  independent
C3 (test fix)  →  depends on C2
C4 (AUDIT)     →  independent (docs only)
C5 (CAPS)      →  independent (docs only)
C6 (CARDS)     →  independent (docs only)
C7 (PROTOCOL)  →  logically after C1, C2 (references both)
```

**Recommended merge order:** C4 → C5 → C6 → C1 → C2+C3 → C7

---

## Post-merge Verification

```bash
# 1. Full test suite
.venv/bin/pytest tests/ -v  # expect 388 passed

# 2. Import smoke test
.venv/bin/python -c "
from uavbench.planners import PLANNERS, DStarLiteRealPlanner
from uavbench.cli.benchmark import run_mission_episode
print(f'Planners: {len(PLANNERS)}')  # 11
print(f'D* Lite: {DStarLiteRealPlanner.__name__}')
print('All imports OK')
"

# 3. Determinism check
.venv/bin/python -c "
from uavbench.planners.dstar_lite_real import DStarLiteRealPlanner
import numpy as np
h = np.zeros((20,20)); nfz = np.zeros((20,20), dtype=bool)
p = DStarLiteRealPlanner(h, nfz)
r1 = p.plan((0,0),(19,19))
p2 = DStarLiteRealPlanner(h, nfz)
r2 = p2.plan((0,0),(19,19))
assert r1.path == r2.path, 'Determinism broken!'
print(f'Path length: {len(r1.path)}, deterministic: ✓')
"
```

---

## Summary

| Commit | Type | Files | Tests Δ | LoC Δ (est) |
|---|---|---|---|---|
| C1 | feat | 2 | +12 | +140 |
| C2 | feat | 3 | +28 | +590 |
| C3 | test | 2 | 0 | +6 |
| C4 | docs | 1 | 0 | +80 |
| C5 | docs | 1 | 0 | +200 |
| C6 | docs | 1 | 0 | +350 |
| C7 | docs | 1 | 0 | +320 |
| **Total** | | **11** | **+40** | **~1,690** |
