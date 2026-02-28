# UAVBench v1 → Mission-Driven Upgrade — Superprompt

## CONTEXT FOR CLAUDE CODE

You are upgrading an existing, **working** UAV navigation benchmark (v1) with mission-driven concepts from a v2 prototype. The v1 codebase has battle-tested planners, dynamics, OSM maps, and visualization that took months to develop. The v2 prototype defined 20 formal contracts and mission concepts but its planners/dynamics had bugs. 

**Strategy: Keep v1's working engine. Layer v2's mission concepts on top.**

DO NOT rewrite planners, dynamics, envs, or the runner from scratch. Instead, **augment** them.

---

## WHAT V1 ALREADY HAS (DO NOT BREAK)

### Working Planners (src/uavbench/planners/)
- `astar.py` — A* baseline (no replan)
- `theta_star.py` — Any-angle (no replan)  
- `dstar_lite.py` / `dstar_lite_real.py` — Incremental replanning
- `mppi.py` — Sampling-based (MPPI)
- `ad_star.py` — Anytime D* (replanning)
- `dwa.py` — Dynamic Window Approach
- `adaptive_astar.py` — Adaptive A*

### Working Dynamics (src/uavbench/dynamics/)
- `fire_spread.py` — CA fire model with wind
- `traffic.py` — Vehicle traffic on roads
- `restriction_zones.py` — Dynamic NFZ
- `dynamic_nfz.py` — No-fly zone management
- `interaction_engine.py` — Cross-dynamic coupling (fire→roads)
- `intruder.py`, `moving_target.py`, `population_risk.py`, `adversarial_uav.py`

### Working Infrastructure
- `envs/urban.py` — UrbanEnv with OSM raster maps
- `benchmark/runner.py` — Episode runner
- `scenarios/` — 9 scenarios (3 families × 3 difficulties) on real Athens maps
- `visualization/` — Stakeholder renderer, overlays, basemap styles
- `missions/` — **Already has mission system** (spec.py, engine.py, builders.py, policies.py, runner.py)
- `data/maps/` — Real OSM maps: downtown, penteli, piraeus (.npz + .geojson)
- `results/paper/` — Existing experiment results

### Working Tests (tests/)
- tests/unit/ — planners, dynamics, missions, visualization
- tests/integration/ — env behavior, V&V contracts
- tests/benchmark/ — full pipeline

---

## WHAT TO ADD FROM V2 (THE UPGRADE)

### Phase 1: Mission Briefing System
**Goal**: Every episode starts with a clear, human-readable mission briefing.

Add to `src/uavbench/missions/`:

```python
@dataclass
class MissionBriefing:
    mission_type: str        # "fire_delivery", "infrastructure_inspection", "maritime_rescue"
    domain: str              # "civil_protection", "critical_infrastructure", "maritime"
    origin_name: str         # "Athens Fire Station #3"
    destination_name: str    # "Evacuation Zone Alpha"
    objective: str           # "Deliver medical supplies to evacuation zone"
    deliverable: str         # "Thermal-sealed medical kit"
    constraints: list[str]   # ["Avoid active fire zones", "Maintain altitude >50m"]
    service_time_steps: int  # Time to spend at destination
    priority: str            # "critical", "high", "routine"
    max_time_steps: int      # Episode timeout
```

**Implementation**:
1. Add `MissionBriefing` dataclass to `missions/spec.py`
2. Add `generate_briefing()` to `missions/engine.py` that creates a briefing from scenario config
3. Log the briefing as an event at step_idx=0 in every episode
4. Create MISSION_CARDS.md in docs/ with briefing templates per scenario family

**Map scenario families to mission stories**:
- `gov_civil_protection_*` → "Emergency medical delivery during wildfire crisis"
- `gov_critical_infrastructure_*` → "Critical infrastructure inspection under restricted airspace"  
- `gov_maritime_domain_*` → "Maritime search and rescue in multi-hazard zone"

### Phase 2: Decision Record (Reject/Accept Logging)
**Goal**: Every move the drone makes (or fails to make) is fully explained.

Add to `src/uavbench/envs/base.py`:

```python
class RejectReason(Enum):
    BUILDING = "building"
    NO_FLY = "no_fly_zone"
    FIRE = "active_fire"
    TRAFFIC = "traffic_vehicle"
    RESTRICTION_ZONE = "restriction_zone"
    FORCED_BLOCK = "forced_interdiction"
    OUT_OF_BOUNDS = "out_of_bounds"
    INTRUDER = "intruder_proximity"
```

**Implementation**:
1. Add `RejectReason` enum to `envs/base.py`
2. Modify `envs/urban.py` step() to return `reject_reason`, `reject_layer`, `reject_cell` in info dict
3. When a move is accepted: log `{"action": action, "accepted": True, "step_idx": N}`
4. When a move is rejected: log `{"action": action, "accepted": False, "reject_reason": reason, "reject_layer": layer, "reject_cell": (r,c), "step_idx": N}`
5. DO NOT change the step() return signature — add to info dict only

### Phase 3: HUD (Heads-Up Display)
**Goal**: Visualization always shows mission context.

Add `src/uavbench/visualization/hud.py`:

```
MISSION: [objective] → [destination_name]
STATUS: EN_ROUTE | SERVICING | COMPLETED | FAILED  |  Dist: [N] cells
PLAN: ACTIVE | NO_PLAN | STALE
STEP: [N] / [max_time_steps]
```

**Implementation**:
1. Create `hud.py` in visualization/
2. Integrate with existing `stakeholder_renderer.py` and `operational_renderer.py`
3. HUD is ALWAYS visible — never hidden
4. If no plan exists → show "NO_PLAN" badge (not blank)
5. If plan exists but path is stale (>N steps old) → show "STALE"

### Phase 4: Forced Interdictions on BFS Corridor (Fairness)
**Goal**: Test all planners against the SAME obstacles, placed on a planner-agnostic reference path.

**Implementation**:
1. Add `src/uavbench/dynamics/forced_block.py` (port from v2 if useful)
2. At episode start: compute BFS shortest path from start→goal on the STATIC map (before dynamics)
3. Place forced interdictions on BFS corridor interior (not on start/goal cells)
4. Interdiction lifecycle: PENDING → ACTIVE → CLEARED (with configurable timing)
5. ALL planners see the SAME interdictions for the SAME seed
6. This is already partially in `updates/forced_replan.py` — check if it can be extended

### Phase 5: Feasibility Guardrail
**Goal**: When all paths are blocked, progressively relax constraints instead of instant failure.

Add `src/uavbench/benchmark/guardrail.py`:

```
Relaxation depths:
  D1: Remove forced blocks
  D2: Erode NFZ boundaries  
  D3: Remove traffic closures
  D4: Open corridor (last resort)
```

**Implementation**:
1. Create guardrail.py in benchmark/
2. After each step, if planner returns no path: try relaxation D1→D2→D3→D4
3. Log: `guardrail_depth`, `relaxations[]`, `feasible_after_guardrail`
4. If still infeasible after D4 → flag episode as infeasible, report exclusion rate

### Phase 6: Contract Tests
**Goal**: Formalize the 20 contracts as runnable tests.

Create `tests/contracts/` with:

```
test_determinism.py      — DC-1, DC-2: same seed → identical outputs
test_fairness.py         — FC-1, FC-2: planner-agnostic interdictions
test_decision_record.py  — EC-1, EC-2: reject/accept logging
test_guardrail.py        — GC-1, GC-2: relaxation, infeasible flagging  
test_event_semantics.py  — EV-1: authoritative step_idx
test_visual_truth.py     — VC-1, VC-2, VC-3: path overlay, plan badges
test_mask_parity.py      — MP-1: one blocking mask
test_replan_storm.py     — RS-1: ≤20% naive replans
test_mission_story.py    — MC-1–MC-4: objective, completion, HUD, briefing
test_planner_contract.py — PC-1, PC-2: legal moves, planned≠executed
```

**IMPORTANT**: These tests should use the EXISTING v1 planners and scenarios. Do not create new planners.

### Phase 7: Paper Experiment Pipeline
**Goal**: Run all experiments and generate publication figures.

Upgrade existing scripts:
1. `scripts/run_v2_paper_experiments.py` → adapt to use v1 runner + v1 planners
2. `scripts/analyze_v2_paper_results.py` → keep as-is (reads CSV)
3. `scripts/generate_paper_snapshots.py` → adapt to use v1 renderer

**Experiment matrix**:
- Planners: astar, theta_star, dstar_lite, mppi, ad_star, dwa (6 from v1)
- Scenarios: 9 existing (3 families × 3 difficulties)
- Seeds: 30 per combination
- Total: 6 × 9 × 30 = 1,620 episodes

---

## THE 20 CONTRACTS (Acceptance Criteria)

Every contract must have a passing test before the upgrade is complete.

### Determinism Contracts
- **DC-1**: One master seed → child RNGs for each subsystem. No global random state.
- **DC-2**: Same (scenario, planner, seed) → bit-identical episode outputs.

### Fairness Contracts  
- **FC-1**: Forced interdictions placed on BFS corridor, NOT on any planner's specific path.
- **FC-2**: If observation noise exists, all planners receive identical noise for same seed.

### Decision Record Contracts
- **EC-1**: Every rejected move logged with: RejectReason enum, reject_layer, reject_cell.
- **EC-2**: Every accepted move logged with step_idx and dynamics state counter.

### Guardrail Contracts
- **GC-1**: Relaxation proceeds D1→D2→D3→D4, each logged.
- **GC-2**: If infeasible after all depths → episode flagged, exclusion rate reported.

### Event Semantics Contracts
- **EV-1**: One authoritative step_idx owned by the runner, passed to all subsystems.

### Visual Truth Contracts
- **VC-1**: If a planned path exists, it is ALWAYS rendered (never silently absent).
- **VC-2**: If no plan exists, HUD shows NO_PLAN or STALE badge.
- **VC-3**: Forced block lifecycle (TRIGGERED→ACTIVE→CLEARED) visible in visualization.

### Mission Story Contracts
- **MC-1**: Every episode has a mission objective (POI + human-readable reason).
- **MC-2**: Completion = arrival at POI + service_time countdown.
- **MC-3**: HUD always shows mission, status, distance.
- **MC-4**: Results include termination_reason + objective_completed.

### Planner↔Env Contracts
- **PC-1**: Every executed move is legal in the current action model.
- **PC-2**: Metrics distinguish planned_waypoints from executed_steps.

### Cross-cutting Contracts
- **MP-1**: ONE compute_blocking_mask() function used by BOTH step legality AND guardrail BFS.
- **RS-1**: Path-progress tracking ensures ≤20% naive replans for replanning planners.

---

## EXECUTION ORDER

```
Phase 1 → Mission Briefings      (add MissionBriefing, generate_briefing(), MISSION_CARDS.md)
Phase 2 → Decision Record        (add RejectReason enum, modify step() info dict)
Phase 3 → HUD                    (add hud.py, integrate with renderers)
Phase 4 → Forced Interdictions   (add/upgrade forced_block.py, BFS corridor)
Phase 5 → Feasibility Guardrail  (add guardrail.py)
Phase 6 → Contract Tests         (create tests/contracts/ with 20 contract tests)
Phase 7 → Paper Pipeline         (adapt experiment scripts to v1 engine)
```

**Gate per phase**: ALL existing tests must still pass + new contract tests for that phase pass.

---

## NON-NEGOTIABLE RULES

1. **DO NOT rewrite planners** — v1 planners work. Add to them, don't replace.
2. **DO NOT rewrite dynamics** — v1 fire_spread, traffic, etc. work. Augment only.
3. **DO NOT rewrite the runner** — modify it to log more, don't rebuild.
4. **DO NOT change envs/urban.py step() return signature** — add info to info dict.
5. **ALL existing tests must pass** after every phase. Run `pytest tests/` before committing.
6. **One commit per phase** with message: "mission-upgrade phase N: [description]"
7. **Branch**: `mission-upgrade` off current main.
8. **Test before commit**: `pytest tests/ -x -q` must show 0 failures.

---

## WHAT TO CHECK FIRST

Before starting, run:
```bash
pytest tests/ -x -q 2>&1 | tail -5
```
Report how many tests pass. This is the baseline — it must never decrease.

Then examine:
1. `src/uavbench/missions/spec.py` — what mission data already exists?
2. `src/uavbench/missions/engine.py` — what does the mission engine already do?
3. `src/uavbench/benchmark/runner.py` — how does the runner loop work?
4. `src/uavbench/envs/urban.py` — what does step() return in info?
5. `src/uavbench/scenarios/configs/gov_civil_protection_hard.yaml` — what config fields exist?

Report findings before writing any code.

---

## OUTCOME

When complete:
- 20 contract tests pass
- All pre-existing tests still pass
- Every episode has a mission briefing at step 0
- Every move has a decision record (accept/reject with reason)
- HUD shows mission context in all visualizations
- Forced interdictions are planner-agnostic
- Guardrail prevents unnecessary infeasible episodes
- Paper experiment pipeline runs 1,620 episodes with v1's battle-tested engine
- Results show meaningful differentiation between planners
