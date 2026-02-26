# UAVBench v2 — Claude Code Launch Prompt

## How to Use This File

**This is your launch prompt.** Copy the section below into Claude Code to start the build.
The supporting files (CLAUDE.md, slash commands, subagents, skills, docs) must already be in your repo.

### Setup Checklist
1. Copy these files into your repo root:
   - `CLAUDE.md` → repo root
   - `.claude/commands/` → all 4 slash commands
   - `.claude/agents/` → all 3 subagents
   - `.claude/skills/contract-testing/` → testing skill
   - `docs/v2/PHASE_GATES.md` → phase gates reference
   - `docs/v2/CONTRACTS.md` → canonical contracts
2. Start Claude Code in your repo root
3. Paste the launch prompt below

---

## 🚀 LAUNCH PROMPT — Copy everything below this line into Claude Code

```
Think hard about this. You are building UAVBench v2, a clean-room rewrite of a UAV navigation
benchmark framework. Read CLAUDE.md, then read docs/v2/PHASE_GATES.md and docs/v2/CONTRACTS.md
to understand the full scope.

CRITICAL RULES (non-negotiable):
1. All v2 code goes in src/uavbench2/. NEVER touch src/uavbench/ (v1).
2. Work on branch rebuild-v2-cleanroom only.
3. Spec-first: write docs and tests BEFORE implementation.
4. Phased execution: complete each phase gate before moving to the next.
5. Single pipeline: ONE runner, ONE CLI, no duplicates.
6. Determinism: same (scenario, planner, seed) → bit-identical outputs.

Start Phase 0 now. Use the v1-reader subagent to scan src/uavbench/ and extract:
- Scenario taxonomy and configs
- Planner interface (methods, input/output types)
- Environment API (reset/step signatures, observation/action spaces)
- Known failure modes (plan disappearing, step_idx off-by-one, mask mismatches, replan storms)
- Metrics schema
- Visualization approach

Then make these decisions (or mark as ASSUMPTION in docs/v2/V2_DECISIONS.md):
- Q1: Horizon policy → default: env-enforced at 4×map_size
- Q2: OSM distribution → default: deterministic fetch+rasterize pipeline (ODbL compliant)
- Q3: Action model → default: 2D grid (2.5D as optional extension)
- Q4: Planner suite → default 6: astar, theta_star, periodic_replan, aggressive_replan, dstar_lite, mppi_grid
- Q5: v2 independence → default: v2 is independent, no v1 schema compatibility required

Write:
- docs/v2/V2_DECISIONS.md with all decisions/assumptions
- outputs/v2/rebuild_audit.json with extracted findings

Gate 0 check: all 8 contracts (DC/FC/EC/GC/EV/VC/MC/PC) must be written as testable
requirements. Run /gate-check to verify.

Do NOT write any implementation code yet. Phase 0 is read-only analysis.
Report your findings and Gate 0 status when done.
```

---

## Phase-by-Phase Follow-up Prompts

After Gate 0 passes, use these prompts to advance through phases:

### Phase 1 (Spec Docs)
```
Gate 0 passed. Start Phase 1: write all spec docs. Use /phase 1.
Write V2_REQUIREMENTS.md, V2_ARCHITECTURE.md, V2_TRACEABILITY.md, V2_TEST_PLAN.md,
MISSION_CARDS.md, and VISUAL_TRUTH_SPEC.md under docs/v2/.
Every contract must have named tests and acceptance criteria.
Do NOT write implementation code. Run /gate-check when done.
```

### Phase 2 (Minimal Scaffold)
```
Gate 1 passed. Start Phase 2: minimal v2 scaffold with "hello mission".
Use the test-writer subagent to write contract_test_determinism.py and
contract_test_mission_story.py FIRST (TDD). Then implement the minimal:
- uavbench2 CLI stub
- UrbanEnvV2 with deterministic reset/step
- Single-goal mission with service_time_s + completion event
Run pytest tests/v2/ -q. Then run /determinism-check. Run /gate-check.
```

### Phase 3 (Decision Record)
```
Gate 2 passed. Start Phase 3: decision record + event semantics.
Write contract_test_decision_record.py and contract_test_event_semantics.py FIRST.
Then implement RejectReason enum, reject_layer/cell logging, authoritative step_idx
owned by runner. Run /gate-check.
```

### Phase 4 (Dynamics)
```
Gate 3 passed. Start Phase 4: dynamics with mask parity.
Write contract_test_mask_parity.py and unit_test_fire_ca.py FIRST.
Implement fire CA, traffic, restriction zones, interaction engine.
Critical: ONE compute_blocking_mask() shared by step legality AND guardrail BFS.
Run /gate-check.
```

### Phase 5 (Fairness)
```
Gate 4 passed. Start Phase 5: forced interdictions + fairness.
Write contract_test_fairness.py FIRST.
Implement BFS reference corridor, t1/t2 interdictions with lifecycle.
Verify interdiction timelines identical across planners for same seed.
Run /gate-check.
```

### Phase 6 (Guardrail)
```
Gate 5 passed. Start Phase 6: feasibility guardrail.
Write contract_test_guardrail.py FIRST.
Implement multi-depth relaxation (D1→D2→D3→D4).
Log guardrail_depth, relaxations, feasible_after_guardrail.
Flag infeasible episodes. Run /gate-check.
```

### Phase 7 (Planners)
```
Gate 6 passed. Start Phase 7: planner suite.
Write contract_test_replan_storm_regression.py and unit_test_dstar_lite_api.py FIRST.
Implement exactly 6 planners: astar, theta_star, periodic_replan, aggressive_replan,
dstar_lite, mppi_grid. All implement PlannerBase.
Path-progress tracking to prevent replan storms (≤20% naive replans).
Run /gate-check.
```

### Phase 8 (Visualization)
```
Gate 7 passed. Start Phase 8: visualization truth + mission HUD.
Write contract_test_visual_truth.py FIRST.
Implement renderer (paper_min + ops_full modes).
HUD always shows mission objective + plan status. Planned path never silently absent.
Generate viz_manifest.csv and viz_frame_checks.json.
Run /gate-check.
```

### Phase 9 (Export)
```
Gate 8 passed. Start Phase 9: export + reproducibility.
Build export_v2_artifacts.py, regenerate_v2_paper_gifs.sh.
Write repro_manifest.json. Verify one command regenerates everything deterministically.
Run /determinism-check. Run /gate-check. Run /verify-contracts.
```

---

## Recovery Prompts

### If context gets long
```
/clear
Read CLAUDE.md. I'm on Phase N of UAVBench v2. Read docs/v2/PHASE_GATES.md.
Run /gate-check to determine current status, then continue from where we left off.
```

### If tests are failing
```
Think harder. The contract tests are failing. Read the failing test output carefully.
Read docs/v2/CONTRACTS.md for the contract being tested.
Use the contract-reviewer subagent to audit the implementation.
Fix the root cause, don't patch the test. Run /gate-check after fixing.
```

### If architecture is drifting
```
Stop. Use the contract-reviewer subagent to audit all implemented code against
docs/v2/CONTRACTS.md. Report any compliance gaps. Do not write new code until
all existing contracts are satisfied. Run /verify-contracts.
```
