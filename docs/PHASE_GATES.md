# UAVBench — Phase Gates

Each phase has a GATE. Do not proceed to Phase N+1 until Phase N gate passes.
When starting a phase, announce: "Starting Phase N: [name]".
When a gate passes, announce: "Gate N PASSED" and commit.

---

## Phase 0 — Baseline Extraction (NO v1 edits, NO v2 code)
**Do**: Read `src/uavbench/` to extract requirements, scenario taxonomy, planner APIs, known failure modes.
**Decide or ASSUME**: horizon policy, OSM distribution, 2D vs 2.5D, planner suite (default 6).
**Write**: `docs/V2_DECISIONS.md`, `outputs/rebuild_audit.json`
**Gate 0**: All 8 contracts (DC/FC/EC/GC/EV/VC/MC/PC) written as testable requirements in V2_REQUIREMENTS.md.

## Phase 1 — Spec-First Docs (NO implementation code)
**Write**:
- `docs/V2_REQUIREMENTS.md` — SHALL requirements + acceptance criteria
- `docs/V2_ARCHITECTURE.md` — module map, single pipeline, data contracts
- `docs/V2_TRACEABILITY.md` — Req → Code → Tests → Evidence
- `docs/V2_TEST_PLAN.md` — unit/contract/integration + CI split
- `docs/MISSION_CARDS.md` — mission story cards per scenario family
- `docs/VISUAL_TRUTH_SPEC.md` — HUD tokens, planned path rules, event semantics
**Gate 1**: Every contract has named test files + acceptance criteria in V2_TEST_PLAN.md.

## Phase 2 — Minimal Scaffold ("Hello Mission")
**Build**:
- `uavbench` CLI stub
- Minimal `UrbanEnvV2` with deterministic `reset(seed)`/`step()`, info dict with `step_idx`
- Minimal mission: single-goal POI + `service_time_s` + completion event
- Contract test: `contract_test_determinism.py` (two runs → hash-identical)
- Contract test: `contract_test_mission_story.py` (objective exists, completion event)
**Gate 2**: `pytest tests/ -q` passes. Two identical seed runs produce identical hashes.

## Phase 3 — Decision Record + Event Semantics
**Build**:
- `RejectReason` enum + `reject_layer` + `reject_cell` + `step_idx` on every rejected move
- `move_accepted=True` + dynamics counter on every accepted move
- Authoritative `step_idx` owned by runner, passed into env/dynamics/logger/renderer
**Gate 3**: `contract_test_decision_record.py` + `contract_test_event_semantics.py` pass.

## Phase 4 — Dynamics with Mask Parity
**Build**:
- Fire CA, traffic, restriction zones, interaction engine
- ONE `compute_blocking_mask(state)` used by BOTH step legality AND guardrail BFS
**Gate 4**: `contract_test_mask_parity.py` passes (no missing blockers between step and guardrail).

## Phase 5 — Corridor Interdictions + Fairness
**Build**:
- A* reference corridor (planner-agnostic)
- Physical corridor interdictions: fire corridor closures (penteli, downtown) and vehicle roadblocks (piraeus)
**Gate 5**: `contract_test_fairness.py` passes. Interdiction timelines identical across planners for same seed.

## Phase 6 — Feasibility Guardrail
**Build**:
- Multi-depth relaxation: D1 clear roadblock vehicles → D2 NFZ erosion → D3 traffic removal → D4 corridor (optional)
- Log: `guardrail_depth`, `relaxations[]`, `feasible_after_guardrail`
- Infeasible episodes flagged; exclusion rate reported
**Gate 6**: `contract_test_guardrail.py` passes.

## Phase 7 — Planner Suite (5 only) + Replan Storm Regression
**Build**: astar, periodic_replan, aggressive_replan, dstar_lite, apf
- All implement `PlannerBase` interface
- Path-progress tracking to prevent replan storms
**Gate 7**: `contract_test_replan_storm_regression.py` passes (≤20% naive replans).
           `unit_test_dstar_lite_api.py` passes.

## Phase 8 — Visualization Truth + Mission HUD
**Build**:
- Renderer modes: `paper_min`, `ops_full`
- HUD always shows: mission domain, objective label, distance_to_task, task_progress
- Planned path overlay: never silently absent. If missing → "NO PLAN" badge.
- Physical interdiction events visible in fire/traffic overlays
**Gate 8**: `contract_test_visual_truth.py` passes. `outputs/viz_manifest.csv` + `viz_frame_checks.json` written.

## Phase 9 — Export & Reproducibility
**Build**:
- `scripts/export_v2_artifacts.py` — all evidence JSONs
- `scripts/regenerate_v2_paper_gifs.sh` — gif pack
- `outputs/repro_manifest.json` — one command regenerates everything
**Gate 9**: Single command produces deterministic artifacts. `outputs/determinism_hashes.json` matches.
